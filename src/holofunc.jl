using CUDA
using CUDA.CUFFT

export cu_transfer_sqrt_arr, cu_transfer, cu_phase_retrieval_holo, cu_get_reconst_vol, cu_get_reconst_xyprojectin, cu_get_reconst_vol_and_xyprojection

function _cu_transfer_sqrt_arr!(Plane, datLen, wavLen, dx)
    x = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if x <= datLen && y <= datLen
        @inbounds Plane[x,y] = 1.0 - ((x-datLen/2)*wavLen/datLen/dx)^2 - ((y-datLen/2)*wavLen/datLen/dx)^2
    end
    return nothing
end

"""
    cu_transfer_sqrt_arr(datlen, wavlen, dx)

Create a CuArray of size `datlen` x `datlen` with the values of the square-root part of the transfer function.

# Arguments
- `datlen::Int`: The size of the CuArray.
- `wavlen::AbstractFloat`: The wavelength of the light.
- `dx::AbstractFloat`: The pixel size of the hologram.

# Returns
- `CuArray{Float32,2}`: The square-root part of the transfer function.
"""
function cu_transfer_sqrt_arr(datlen::Int, wavlen::AbstractFloat, dx::AbstractFloat)
    Plane = CuArray{Float32}(undef, datlen, datlen)
    threads = (32, 32)
    blocks = cld.((datlen, datlen), threads)
    @cuda threads=threads blocks=blocks _cu_transfer_sqrt_arr!(Plane, datlen, wavlen, dx)
    return Plane
end

function _cu_transfer!(Plane, z0, datLen, wavLen, d_sqr)
    x = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if x <= datLen && y <= datLen
        @inbounds Plane[x,y] = exp(2im*pi*z0/wavLen*sqrt(d_sqr[x,y]))
    end
    return nothing
end

"""
    cu_transfer(z0, datLen, wavLen, d_sqr)

Create a CuArray of size `datLen` x `datLen` with the values of the transfer function for a given propagated distance z0. `d_sqr` can be obtained with `cutransfersqrtarr(datlen, wavlen, dx)`.

# Arguments
- `z0::AbstractFloat`: The distance to propagate the wave.
- `datLen::Int`: The size of the CuArray.
- `wavLen::AbstractFloat`: The wavelength of the light.
- `d_sqr::CuArray{Float32,2}`: The square of the distance from the center of the hologram, obtained with `cutransfersqrtarr(datlen, wavlen, dx)`.

# Returns
- `CuArray{ComplexF32,2}`: The transfer function for the propagation.
"""
function cu_transfer(z0::AbstractFloat, datLen::Int, wavLen::AbstractFloat, d_sqr::CuArray{Float32,2})
    Plane = CuArray{ComplexF32}(undef, datLen, datLen)
    threads = (32, 32)
    blocks = cld.((datLen, datLen), threads)
    @cuda threads=threads blocks=blocks _cu_transfer!(Plane, z0, datLen, wavLen, d_sqr)
    return Plane
end

"""
    cu_phase_retrieval_holo(holo1, holo2, transfer, invtransfer, priter, datlen)

Perform the Gerchberg-Saxton algorithm-based phase retrieving on two holograms and return the retrieved light field at the z-coordinate point of `holo1`. The algorithm is repeated `priter` times. `holo1` and `holo2` are the holograms (I = |phi|^2) of the object at two different z-coordinates. `transfer` and `invtransfer` are the transfer functions for the propagation from `holo1` to `holo2` and vice versa. `datlen` is the size of the holograms.

# Arguments
- `holo1::CuArray{Float32,2}`: The hologram at the z-cordinate of closer to the object.
- `holo2::CuArray{Float32,2}`: The hologram at the z-coordinate of further from the object.
- `transfer::CuArray{ComplexF32,2}`: The transfer function from `holo1` to `holo2`.
- `invtransfer::CuArray{ComplexF32,2}`: The transfer function from `holo2` to `holo1`.
- `priter::Int`: The number of iterations to perform the algorithm.
- `datlen::Int`: The size of the holograms.

# Returns
- `CuArray{ComplexF32,2}`: The retrieved light field at the z-coordinate of `holo1`.
"""
function cu_phase_retrieval_holo(holo1::CuArray{Float32,2}, holo2::CuArray{Float32,2}, transfer::CuArray{ComplexF32,2}, invtransfer::CuArray{ComplexF32,2}, priter::Int, datlen::Int)
    @assert size(holo1) == size(holo2) == size(transfer) == size(invtransfer) == (datlen, datlen) "All arrays must have the same size as ($datlen, $datlen). Got $(size(holo1)), $(size(holo2)), $(size(transfer)), $(size(invtransfer))."

    light1 = CuArray{ComplexF32}(undef, datlen, datlen)
    light2 = CuArray{ComplexF32}(undef, datlen, datlen)
    phi1 = CuArray{Float32}(undef, datlen, datlen)
    phi2 = CuArray{Float32}(undef, datlen, datlen)
    sqrtI1 = sqrt.(holo1)
    sqrtI2 = sqrt.(holo2)

    light1 .= sqrtI1 .+ 0.0im

    for _ in 1:priter
        # STEP1
        light2 .= CUFFT.ifft(CUFFT.fftshift(CUFFT.fftshift(CUFFT.fft(light1)).*transfer))
        phi2 .= angle.(light2)

        # STEP2
        light2 .= sqrtI2.*exp.(1.0im.*phi2)

        # STEP3
        light1 .= CUFFT.ifft(CUFFT.fftshift(CUFFT.fftshift(CUFFT.fft(light2)).*invtransfer))
        phi1 .= angle.(light1)

        # STEP4
        light1 .= sqrtI1.*exp.(1.0im.*phi1)
    end

    return light1
end

"""
    cu_get_reconst_vol(holo, transfer_front, transfer_dz, slices)

Reconstruct the observation volume from the light field `light_field` using the transfer functions `transfer_front` and `transfer_dz`. `transfer_front` propagates the light field to the front of the volume, and `transfer_dz` propagates the light field between the slices. `slices` is the number of slices in the volume.

# Arguments
- `light_field::CuArray{ComplexF32,2}`: The light_field to reconstruct. In Gabor's holography, this is the square root of the hologram.
- `transfer_front::CuArray{ComplexF32,2}`: The transfer function to propagate the light field to the front of the volume.
- `transfer_dz::CuArray{ComplexF32,2}`: The transfer function to propagate the light field between the slices.
- `slices::Int`: The number of slices in the volume.

# Returns
- `CuArray{Float32,3}`: The reconstructed volume.
"""
function cu_get_reconst_vol(light_field::CuArray{ComplexF32,2}, transfer_front::CuArray{ComplexF32,2}, transfer_dz::CuArray{ComplexF32,2}, slices::Int) 
    @assert size(light_field) == size(transfer_front) == size(transfer_dz) "All arrays must have the same size. Got $(size(light_field)), $(size(transfer_front)), $(size(transfer_dz))."

    vol = CuArray{Float32}(undef, size(light_field)..., slices)
    
    fftholo = CUFFT.fftshift(CUFFT.fft(light_field))
    fftholo .= fftholo.*transfer_front

    vol[:,:,1] .= fftholo |> CUFFT.ifftshift |> CUFFT.ifft |> (x -> x .* conj.(x)) .|> abs .|> Float32

    for i in 2:slices
        fftholo .= fftholo.*transfer_dz
        vol[:,:,i] .= fftholo |> CUFFT.ifftshift |> CUFFT.ifft |> (x -> x .* conj.(x)) .|> abs .|> Float32
    end

    return vol
end

function _cu_get_xy_projection_from_vol!(Plane, vol, datlen, slices)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if x <= datlen && y <= datlen
        min_val = vol[x, y, 1]
        for z in 2:slices
            @inbounds val = vol[x, y, z]
            min_val = val < min_val ? val : min_val
        end
        Plane[x, y] = min_val
    end

    return nothing
end

"""
    cu_get_reconst_xyprojectin(light_field, transfer_front, transfer_dz, slices)

Get the XY projection of the reconstructed volume from the light field `light_field` using the transfer functions `transfer_front` and `transfer_dz`. `transfer_front` propagates the light field to the front of the volume, and `transfer_dz` propagates the light field between the slices. `slices` is the number of slices in the volume.

# Arguments
- `light_field::CuArray{ComplexF32,2}`: The light_field to reconstruct. In Gabor's holography, this is the square root of the hologram.
- `transfer_front::CuArray{ComplexF32,2}`: The transfer function to propagate the light field to the front of the volume.
- `transfer_dz::CuArray{ComplexF32,2}`: The transfer function to propagate the light field between the slices.
- `slices::Int`: The number of slices in the volume.

# Returns
- `CuArray{Float32,2}`: The XY projection of the reconstructed volume.
"""
function cu_get_reconst_xyprojection(light_field::CuArray{ComplexF32,2}, transfer_front::CuArray{ComplexF32,2}, transfer_dz::CuArray{ComplexF32,2}, slices::Int) 
    @assert size(light_field) == size(transfer_front) == size(transfer_dz) "All arrays must have the same size. Got $(size(light_field)), $(size(transfer_front)), $(size(transfer_dz))."
    
    vol = CuArray{Float32}(undef, size(light_field)..., slices)
    
    fftholo = CUFFT.fftshift(CUFFT.fft(light_field))
    fftholo .= fftholo.*transfer_front

    vol[:,:,1] .= fftholo |> CUFFT.ifftshift |> CUFFT.ifft |> (x -> x .* conj.(x)) .|> abs .|> Float32

    for i in 2:slices
        fftholo .= fftholo.*transfer_dz
        vol[:,:,i] .= fftholo |> CUFFT.ifftshift |> CUFFT.ifft |> (x -> x .* conj.(x)) .|> abs .|> Float32
    end

    xyprojection = CuArray{Float32}(undef, size(light_field)...)
    threads = (32, 32)
    blocks = cld.((size(light_field)[1], size(light_field)[2]), threads)
    @cuda threads=threads blocks=blocks _cu_get_xy_projection_from_vol!(xyprojection, vol, size(light_field, 1), slices)

    return xyprojection
end

"""
    cu_get_reconst_vol_and_xyprojection(light_field, transfer_front, transfer_dz, slices)

Reconstruct the observation volume from the light field `light_field` and get the XY projection of the volume using the transfer functions `transfer_front` and `transfer_dz`. `transfer_front` propagates the light field to the front of the volume, and `transfer_dz` propagates the light field between the slices. `slices` is the number of slices in the volume.

# Arguments
- `light_field::CuArray{ComplexF32,2}`: The light_field to reconstruct. In Gabor's holography, this is the square root of the hologram.
- `transfer_front::CuArray{ComplexF32,2}`: The transfer function to propagate the light field to the front of the volume.
- `transfer_dz::CuArray{ComplexF32,2}`: The transfer function to propagate the light field between the slices.
- `slices::Int`: The number of slices in the volume.

# Returns
- `CuArray{Float32,3}`: The reconstructed volume.
- `CuArray{Float32,2}`: The XY projection of the reconstructed volume.
"""
function cu_get_reconst_vol_and_xyprojection(light_field::CuArray{ComplexF32,2}, transfer_front::CuArray{ComplexF32,2}, transfer_dz::CuArray{ComplexF32,2}, slices::Int)
    @assert size(light_field) == size(transfer_front) == size(transfer_dz) "All arrays must have the same size. Got $(size(light_field)), $(size(transfer_front)), $(size(transfer_dz))."

    vol = CuArray{Float32}(undef, size(light_field)..., slices)
    
    fftholo = CUFFT.fftshift(CUFFT.fft(light_field))
    fftholo .= fftholo.*transfer_front

    vol[:,:,1] .= fftholo |> CUFFT.ifftshift |> CUFFT.ifft |> (x -> x .* conj.(x)) .|> abs .|> Float32

    for i in 2:slices
        fftholo .= fftholo.*transfer_dz
        vol[:,:,i] .= fftholo |> CUFFT.ifftshift |> CUFFT.ifft |> (x -> x .* conj.(x)) .|> abs .|> Float32
    end

    xyprojection = CuArray{Float32}(undef, size(light_field)...)
    threads = (32, 32)
    blocks = cld.((size(light_field)[1], size(light_field)[2]), threads)
    @cuda threads=threads blocks=blocks _cu_get_xy_projection_from_vol!(xyprojection, vol, size(light_field, 1), slices)
    
    return vol, xyprojection
end

