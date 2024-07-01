using CUDA
using CUDA.CUFFT

export cu_transfer_sqrt_arr, cu_transfer, cu_gabor_wavefront, cu_phase_retrieval_holo, cu_get_reconst_vol, cu_get_reconst_xyprojection, cu_get_reconst_vol_and_xyprojection

# Not tested
export cu_get_reconst_complex_vol, cu_get_reconst_complex_vol, cu_get_reconst_xyprojection, cu_get_reconst_vol_and_xyprojection, cu_get_reconst_vol_lpf, cu_get_reconst_complex_vol_lpf

function _cu_transfer_sqrt_arr!(Plane, datLen, wavLen, dx)
    x = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if x <= datLen && y <= datLen
        @inbounds Plane[x,y] = 1.0 - ((x-datLen/2 - 1.0)*wavLen/datLen/dx)^2 - ((y-datLen/2 - 1.0)*wavLen/datLen/dx)^2
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
- `CuTransferSqrtPart{Float32}`: The square-root part of the transfer function. See [`CuTransferSqrtPart`](@ref).
"""
function cu_transfer_sqrt_arr(datlen::Int, wavlen::AbstractFloat, dx::AbstractFloat)
    Plane = CuArray{Float32}(undef, datlen, datlen)
    threads = (32, 32)
    blocks = cld.((datlen, datlen), threads)
    @cuda threads=threads blocks=blocks _cu_transfer_sqrt_arr!(Plane, datlen, wavlen, dx)
    return CuTransferSqrtPart(Plane)
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
- `d_sqr::::CuTransferSqrtPart{Float32}`: The square of the distance from the center of the hologram, obtained with `cutransfersqrtarr(datlen, wavlen, dx)`.

# Returns
- `CuTransfer{Float32}`: The transfer function for the propagation. See [`CuTransfer`](@ref).
"""
function cu_transfer(z0::AbstractFloat, datLen::Int, wavLen::AbstractFloat, d_sqr::CuTransferSqrtPart{Float32})
    Plane = CuArray{ComplexF32}(undef, datLen, datLen)
    threads = (32, 32)
    blocks = cld.((datLen, datLen), threads)
    @cuda threads=threads blocks=blocks _cu_transfer!(Plane, z0, datLen, wavLen, d_sqr.data)
    return CuTransfer(Plane)
end

"""
    cu_gabor_wavefront(holo)

Create a wavefront from single hologram `holo`. This is for Gabor holography. The wavefront is created by taking the square root of the hologram and casting it to a complex number.

# Arguments
- `holo::CuArray{Float32,2}`: The hologram to create the wavefront from.

# Returns
- `CuWavefront{ComplexF32}`: The wavefront created from the hologram. See [`CuWavefront`](@ref).
"""
function cu_gabor_wavefront(holo::CuArray{Float32,2})
    return CuWavefront(ComplexF32.(sqrt.(holo)))
end

"""
    cu_phase_retrieval_holo(holo1, holo2, transfer, invtransfer, priter, datlen)

Perform the Gerchberg-Saxton algorithm-based phase retrieving on two holograms and return the retrieved wavefront at the z-coordinate point of `holo1`. The algorithm is repeated `priter` times. `holo1` and `holo2` are the holograms (I = |phi|^2) of the object at two different z-coordinates. `transfer` and `invtransfer` are the transfer functions for the propagation from `holo1` to `holo2` and vice versa. `datlen` is the size of the holograms.

# Arguments
- `holo1::CuArray{Float32,2}`: The hologram at the z-cordinate of closer to the object.
- `holo2::CuArray{Float32,2}`: The hologram at the z-coordinate of further from the object.
- `transfer::CuTransfer{ComplexF32}`: The transfer function from `holo1` to `holo2`.
- `invtransfer::CuTransfer{ComplexF32}`: The transfer function from `holo2` to `holo1`.
- `priter::Int`: The number of iterations to perform the algorithm.
- `datlen::Int`: The size of the holograms.

# Returns
- `CuWavefront{ComplexF32}`: The retrieved wavefront at the z-coordinate of `holo1`. See [`CuWavefront`](@ref).
"""
function cu_phase_retrieval_holo(holo1::CuArray{Float32,2}, holo2::CuArray{Float32,2}, transfer::CuTransfer{ComplexF32}, invtransfer::CuTransfer{ComplexF32}, priter::Int, datlen::Int)
    @assert size(holo1) == size(holo2) == size(transfer.data) == size(invtransfer.data) == (datlen, datlen) "All arrays must have the same size as ($datlen, $datlen). Got $(size(holo1)), $(size(holo2)), $(size(transfer.data)), $(size(invtransfer.data))."

    light1 = CuArray{ComplexF32}(undef, datlen, datlen)
    light2 = CuArray{ComplexF32}(undef, datlen, datlen)
    phi1 = CuArray{Float32}(undef, datlen, datlen)
    phi2 = CuArray{Float32}(undef, datlen, datlen)
    sqrtI1 = sqrt.(holo1)
    sqrtI2 = sqrt.(holo2)

    light1 .= sqrtI1 .+ 0.0im

    for _ in 1:priter
        # STEP1
        light2 .= CUFFT.ifft(CUFFT.ifftshift(CUFFT.fftshift(CUFFT.fft(light1)).*transfer.data))
        phi2 .= angle.(light2)

        # STEP2
        light2 .= sqrtI2.*exp.(1.0im.*phi2)

        # STEP3
        light1 .= CUFFT.ifft(CUFFT.ifftshift(CUFFT.fftshift(CUFFT.fft(light2)).*invtransfer.data))
        phi1 .= angle.(light1)

        # STEP4
        light1 .= sqrtI1.*exp.(1.0im.*phi1)
    end

    return CuWavefront(light1)
end

########################  Recontruction functions  ########################

# function _ifft_and_abs(fftholo; lpf::CuLowPassFilter=nothing, return_type::Symbol=:Float32)
#     if return_type in [:Float32, :Float64, :Float16, :N0f8]
#         if lpf === nothing
#             return fftholo |> CUFFT.ifftshift |> CUFFT.ifft .|> (x -> abs(x)^2) .|> (x -> clamp(x, 0, 1)) .|> x -> convert(return_type, x)
#         else
#             return fftholo |> (x -> x .* lpf.data) |> CUFFT.ifftshift |> CUFFT.ifft .|> (x -> abs(x)^2) .|> Float32
#         end
#     else
#         if lpf === nothing
#             return fftholo |> CUFFT.ifftshift |> CUFFT.ifft
#         else
#             return fftholo |> (x -> x .* lpf.data) |> CUFFT.ifftshift |> CUFFT.ifft
#         end
#     end


# end


"""
    cu_get_reconst_vol(holo, transfer_front, transfer_dz, slices)

Reconstruct the observation volume from the `wavefront` using the transfer functions `transfer_front` and `transfer_dz`. `transfer_front` propagates the wavefront to the front of the volume, and `transfer_dz` propagates the wavefront between the slices. `slices` is the number of slices in the volume.

# Arguments
- `wavefront::CuArray{ComplexF32,2}`: The wavefront to reconstruct. In Gabor's holography, this is the square root of the hologram.
- `transfer_front::CuTransfer{ComplexF32}`: The transfer function to propagate the wavefront to the front of the volume. See [`CuTransfer`](@ref).
- `transfer_dz::CuTransfer{ComplexF32}`: The transfer function to propagate the wavefront between the slices.
- `slices::Int`: The number of slices in the volume.

# Returns
- `CuArray{Float32,3}`: The reconstructed intensity volume.
"""
function cu_get_reconst_vol(wavefront::CuWavefront{ComplexF32}, transfer_front::CuTransfer{ComplexF32}, transfer_dz::CuTransfer{ComplexF32}, slices::Int) 
    @assert size(wavefront.data) == size(transfer_front.data) == size(transfer_dz.data) "All arrays must have the same size. Got $(size(wavefront.data)), $(size(transfer_front.data)), $(size(transfer_dz.data))."

    vol = CuArray{Float32}(undef, size(wavefront.data)..., slices)
    
    fftholo = CUFFT.fftshift(CUFFT.fft(wavefront.data))
    fftholo .= fftholo.*transfer_front.data

    vol[:,:,1] .= fftholo |> CUFFT.ifftshift |> CUFFT.ifft .|> (x -> abs(x)^2) .|> Float32

    for i in 2:slices
        fftholo .= fftholo.*transfer_dz.data
        vol[:,:,i] .= fftholo |> CUFFT.ifftshift |> CUFFT.ifft .|> (x -> abs(x)^2) .|> Float32
    end

    return vol
end

"""
    cu_get_reconst_complex_vol(holo, transfer_front, transfer_dz, slices)

Reconstruct the observation volume from the `wavefront` using the transfer functions `transfer_front` and `transfer_dz` and return the complex amplitude volume. `transfer_front` propagates the wavefront to the front of the volume, and `transfer_dz` propagates the wavefront between the slices. `slices` is the number of slices in the volume.

# Arguments
- `wavefront::CuArray{ComplexF32,2}`: The wavefront to reconstruct. In Gabor's holography, this is the square root of the hologram.
- `transfer_front::CuTransfer{ComplexF32}`: The transfer function to propagate the wavefront to the front of the volume. See [`CuTransfer`](@ref).
- `transfer_dz::CuTransfer{ComplexF32}`: The transfer function to propagate the wavefront between the slices.
- `slices::Int`: The number of slices in the volume.

# Returns
- `CuArray{ComplexF32,3}`: The reconstructed complex amplitude volume.
"""
function cu_get_reconst_complex_vol(wavefront::CuWavefront{ComplexF32}, transfer_front::CuTransfer{ComplexF32}, transfer_dz::CuTransfer{ComplexF32}, slices::Int) 
    @assert size(wavefront.data) == size(transfer_front.data) == size(transfer_dz.data) "All arrays must have the same size. Got $(size(wavefront.data)), $(size(transfer_front.data)), $(size(transfer_dz.data))."

    vol = CuArray{ComplexF32}(undef, size(wavefront.data)..., slices)
    
    fftholo = CUFFT.fftshift(CUFFT.fft(wavefront.data))
    fftholo .= fftholo.*transfer_front.data

    vol[:,:,1] .= fftholo |> CUFFT.ifftshift |> CUFFT.ifft

    for i in 2:slices
        fftholo .= fftholo.*transfer_dz.data
        vol[:,:,i] .= fftholo |> CUFFT.ifftshift |> CUFFT.ifft
    end

    return vol
end

"""
    cu_get_reconst_xyprojectin(wavefront, transfer_front, transfer_dz, slices)

Get the XY projection of the reconstructed volume from the `wavefront` using the transfer functions `transfer_front` and `transfer_dz`. `transfer_front` propagates the wavefront to the front of the volume, and `transfer_dz` propagates the wavefront between the slices. `slices` is the number of slices in the volume.

# Arguments
- `wavefront::CuWavefront{ComplexF32}`: The wavefront to reconstruct. In Gabor's holography, this is the square root of the hologram.
- `transfer_front::CuTransfer{ComplexF32}`: The transfer function to propagate the wavefront to the front of the volume. See [`CuTransfer`](@ref).
- `transfer_dz::CuTransfer{ComplexF32}`: The transfer function to propagate the wavefront between the slices.
- `slices::Int`: The number of slices in the volume.

# Returns
- `CuArray{Float32,2}`: The XY projection of the reconstructed volume.
"""
function cu_get_reconst_xyprojection(wavefront::CuWavefront{ComplexF32}, transfer_front::CuTransfer{ComplexF32}, transfer_dz::CuTransfer{ComplexF32}, slices::Int) 
    @assert size(wavefront.data) == size(transfer_front.data) == size(transfer_dz.data) "All arrays must have the same size. Got $(size(wavefront.data)), $(size(transfer_front.data)), $(size(transfer_dz.data))."
    
    proj = CuArray{Float32}(undef, size(wavefront.data)...)
    projtmp = CuArray{Float32}(undef, size(wavefront.data)...)

    fftholo = CUFFT.fftshift(CUFFT.fft(wavefront.data))
    fftholo .= fftholo.*transfer_front.data

    proj .= fftholo |> CUFFT.ifftshift |> CUFFT.ifft |> (x -> x .* conj.(x)) .|> abs .|> Float32

    for i in 2:slices
        fftholo .= fftholo.*transfer_dz.data
        projtmp .= fftholo |> CUFFT.ifftshift |> CUFFT.ifft |> (x -> x .* conj.(x)) .|> abs .|> Float32 
        proj .= CUDA.min.(proj, projtmp)
    end

    return proj
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
    cu_get_reconst_vol_and_xyprojection(wavefront, transfer_front, transfer_dz, slices)

Reconstruct the observation volume from the `wavefront` and get the XY projection of the volume using the transfer functions `transfer_front` and `transfer_dz`. `transfer_front` propagates the wavefront to the front of the volume, and `transfer_dz` propagates the wavefront between the slices. `slices` is the number of slices in the volume.

# Arguments
- `wavefront::CuWavefront{ComplexF32}`: The wavefront to reconstruct. In Gabor's holography, this is the square root of the hologram.
- `transfer_front::CuTransfer{ComplexF32}`: The transfer function to propagate the wavefront to the front of the volume. See [`CuTransfer`](@ref).
- `transfer_dz::CuTransfer{ComplexF32}`: The transfer function to propagate the wavefront between the slices.
- `slices::Int`: The number of slices in the volume.

# Returns
- `CuArray{Float32,3}`: The reconstructed volume.
- `CuArray{Float32,2}`: The XY projection of the reconstructed volume.
"""
function cu_get_reconst_vol_and_xyprojection(wavefront::CuWavefront{ComplexF32}, transfer_front::CuTransfer{ComplexF32}, transfer_dz::CuTransfer{ComplexF32}, slices::Int)
    @assert size(wavefront.data) == size(transfer_front.data) == size(transfer_dz.data) "All arrays must have the same size. Got $(size(wavefront.data)), $(size(transfer_front.data)), $(size(transfer_dz.data))."

    vol = CuArray{Float32}(undef, size(wavefront.data)..., slices)
    
    fftholo = CUFFT.fftshift(CUFFT.fft(wavefront.data))
    fftholo .= fftholo.*transfer_front.data

    vol[:,:,1] .= fftholo |> CUFFT.ifftshift |> CUFFT.ifft |> (x -> x .* conj.(x)) .|> abs .|> Float32

    for i in 2:slices
        fftholo .= fftholo.*transfer_dz.data
        vol[:,:,i] .= fftholo |> CUFFT.ifftshift |> CUFFT.ifft |> (x -> x .* conj.(x)) .|> abs .|> Float32
    end

    xyprojection = CuArray{Float32}(undef, size(wavefront.data)...)
    threads = (32, 32)
    blocks = cld.((size(wavefront.data)[1], size(wavefront.data)[2]), threads)
    @cuda threads=threads blocks=blocks _cu_get_xy_projection_from_vol!(xyprojection, vol, size(wavefront.data, 1), slices)
    
    return vol, xyprojection
end

########################  Recontruction functions with low pass filter  ########################

"""
    cu_get_reconst_vol(holo, transfer_front, transfer_dz, slices)

Reconstruct the observation volume from the `wavefront` using the transfer functions `transfer_front` and `transfer_dz`. `transfer_front` propagates the wavefront to the front of the volume, and `transfer_dz` propagates the wavefront between the slices. `slices` is the number of slices in the volume.

# Arguments
- `wavefront::CuArray{ComplexF32,2}`: The wavefront to reconstruct. In Gabor's holography, this is the square root of the hologram.
- `transfer_front::CuTransfer{ComplexF32}`: The transfer function to propagate the wavefront to the front of the volume. See [`CuTransfer`](@ref).
- `transfer_dz::CuTransfer{ComplexF32}`: The transfer function to propagate the wavefront between the slices.
- `slices::Int`: The number of slices in the volume.
- `lpf::CuLowPassFilter{Float32}`: The low pass filter to apply to the reconstructed volume.

# Returns
- `CuArray{Float32,3}`: The reconstructed intensity volume.
"""
function cu_get_reconst_vol_lpf(wavefront::CuWavefront{ComplexF32}, transfer_front::CuTransfer{ComplexF32}, transfer_dz::CuTransfer{ComplexF32}, slices::Int, lpf::CuLowPassFilter{Float32})
    @assert size(wavefront.data) == size(transfer_front.data) == size(transfer_dz.data) "All arrays must have the same size. Got $(size(wavefront.data)), $(size(transfer_front.data)), $(size(transfer_dz.data))."

    vol = CuArray{Float32}(undef, size(wavefront.data)..., slices)
    
    fftholo = CUFFT.fftshift(CUFFT.fft(wavefront.data))
    fftholo .= fftholo.*transfer_front.data

    vol[:,:,1] .= fftholo |> (x -> x .* lpf.data) |> CUFFT.ifftshift |> CUFFT.ifft |> (x -> x .* conj.(x)) .|> abs .|> Float32

    for i in 2:slices
        fftholo .= fftholo.*transfer_dz.data
        vol[:,:,i] .= fftholo |> (x -> x .* lpf.data) |> CUFFT.ifftshift |> CUFFT.ifft |> (x -> x .* conj.(x)) .|> abs .|> Float32
    end

    return vol
end

"""
    cu_get_reconst_complex_vol(holo, transfer_front, transfer_dz, slices)

Reconstruct the observation volume from the `wavefront` using the transfer functions `transfer_front` and `transfer_dz` and return the complex amplitude volume. `transfer_front` propagates the wavefront to the front of the volume, and `transfer_dz` propagates the wavefront between the slices. `slices` is the number of slices in the volume.

# Arguments
- `wavefront::CuArray{ComplexF32,2}`: The wavefront to reconstruct. In Gabor's holography, this is the square root of the hologram.
- `transfer_front::CuTransfer{ComplexF32}`: The transfer function to propagate the wavefront to the front of the volume. See [`CuTransfer`](@ref).
- `transfer_dz::CuTransfer{ComplexF32}`: The transfer function to propagate the wavefront between the slices.
- `slices::Int`: The number of slices in the volume.
- `lpf::CuLowPassFilter{Float32}`: The low pass filter to apply to the reconstructed volume.

# Returns
- `CuArray{ComplexF32,3}`: The reconstructed complex amplitude volume.
"""
function cu_get_reconst_complex_vol_lpf(wavefront::CuWavefront{ComplexF32}, transfer_front::CuTransfer{ComplexF32}, transfer_dz::CuTransfer{ComplexF32}, slices::Int, lpf::CuLowPassFilter{Float32})
    @assert size(wavefront.data) == size(transfer_front.data) == size(transfer_dz.data) "All arrays must have the same size. Got $(size(wavefront.data)), $(size(transfer_front.data)), $(size(transfer_dz.data))."

    vol = CuArray{ComplexF32}(undef, size(wavefront.data)..., slices)
    
    fftholo = CUFFT.fftshift(CUFFT.fft(wavefront.data))
    fftholo .= fftholo.*transfer_front.data

    vol[:,:,1] .= fftholo |> (x -> x .* lpf.data) |> CUFFT.ifftshift |> CUFFT.ifft

    for i in 2:slices
        fftholo .= fftholo.*transfer_dz.data
        vol[:,:,i] .= fftholo |> (x -> x .* lpf.data) |> CUFFT.ifftshift |> CUFFT.ifft
    end

    return vol
end

"""
    cu_get_reconst_xyprojectin(wavefront, transfer_front, transfer_dz, slices)

Get the XY projection of the reconstructed volume from the `wavefront` using the transfer functions `transfer_front` and `transfer_dz`. `transfer_front` propagates the wavefront to the front of the volume, and `transfer_dz` propagates the wavefront between the slices. `slices` is the number of slices in the volume.

# Arguments
- `wavefront::CuWavefront{ComplexF32}`: The wavefront to reconstruct. In Gabor's holography, this is the square root of the hologram.
- `transfer_front::CuTransfer{ComplexF32}`: The transfer function to propagate the wavefront to the front of the volume. See [`CuTransfer`](@ref).
- `transfer_dz::CuTransfer{ComplexF32}`: The transfer function to propagate the wavefront between the slices.
- `slices::Int`: The number of slices in the volume.
- `lpf::CuLowPassFilter{Float32}`: The low pass filter to apply to the reconstructed volume.

# Returns
- `CuArray{Float32,2}`: The XY projection of the reconstructed volume.
"""
function cu_get_reconst_xyprojection_lpf(wavefront::CuWavefront{ComplexF32}, transfer_front::CuTransfer{ComplexF32}, transfer_dz::CuTransfer{ComplexF32}, slices::Int, lpf::CuLowPassFilter{Float32}) 
    @assert size(wavefront.data) == size(transfer_front.data) == size(transfer_dz.data) "All arrays must have the same size. Got $(size(wavefront.data)), $(size(transfer_front.data)), $(size(transfer_dz.data))."
    
    proj = CuArray{Float32}(undef, size(wavefront.data)...)
    projtmp = CuArray{Float32}(undef, size(wavefront.data)...)

    fftholo = CUFFT.fftshift(CUFFT.fft(wavefront.data))
    fftholo .= fftholo.*transfer_front.data

    proj .= fftholo |> (x -> x .* lpf.data) |> CUFFT.ifftshift |> CUFFT.ifft |> (x -> x .* conj.(x)) .|> abs .|> Float32

    for i in 2:slices
        fftholo .= fftholo.*transfer_dz.data
        projtmp .= fftholo |> (x -> x .* lpf.data) |> CUFFT.ifftshift |> CUFFT.ifft |> (x -> x .* conj.(x)) .|> abs .|> Float32 
        proj .= CUDA.min.(proj, projtmp)
    end

    return proj
end

"""
    cu_get_reconst_vol_and_xyprojection(wavefront, transfer_front, transfer_dz, slices, lpf)

Reconstruct the observation volume from the `wavefront` and get the XY projection of the volume using the transfer functions `transfer_front` and `transfer_dz`. `transfer_front` propagates the wavefront to the front of the volume, and `transfer_dz` propagates the wavefront between the slices. `slices` is the number of slices in the volume.

# Arguments
- `wavefront::CuWavefront{ComplexF32}`: The wavefront to reconstruct. In Gabor's holography, this is the square root of the hologram.
- `transfer_front::CuTransfer{ComplexF32}`: The transfer function to propagate the wavefront to the front of the volume. See [`CuTransfer`](@ref).
- `transfer_dz::CuTransfer{ComplexF32}`: The transfer function to propagate the wavefront between the slices.
- `slices::Int`: The number of slices in the volume.
- `lpf::CuLowPassFilter{Float32}`: The low pass filter to apply to the reconstructed volume.

# Returns
- `CuArray{Float32,3}`: The reconstructed volume.
- `CuArray{Float32,2}`: The XY projection of the reconstructed volume.
"""
function cu_get_reconst_vol_and_xyprojection_lpf(wavefront::CuWavefront{ComplexF32}, transfer_front::CuTransfer{ComplexF32}, transfer_dz::CuTransfer{ComplexF32}, slices::Int, lpf::CuLowPassFilter{Float32})
    @assert size(wavefront.data) == size(transfer_front.data) == size(transfer_dz.data) "All arrays must have the same size. Got $(size(wavefront.data)), $(size(transfer_front.data)), $(size(transfer_dz.data))."

    vol = CuArray{Float32}(undef, size(wavefront.data)..., slices)
    
    fftholo = CUFFT.fftshift(CUFFT.fft(wavefront.data))
    fftholo .= fftholo.*transfer_front.data

    vol[:,:,1] .= fftholo |> (x -> x .* lpf.data) |> CUFFT.ifftshift |> CUFFT.ifft |> (x -> x .* conj.(x)) .|> abs .|> Float32

    for i in 2:slices
        fftholo .= fftholo.*transfer_dz.data
        vol[:,:,i] .= fftholo |> (x -> x .* lpf.data) |> CUFFT.ifftshift |> CUFFT.ifft |> (x -> x .* conj.(x)) .|> abs .|> Float32
    end

    xyprojection = CuArray{Float32}(undef, size(wavefront.data)...)
    threads = (32, 32)
    blocks = cld.((size(wavefront.data)[1], size(wavefront.data)[2]), threads)
    @cuda threads=threads blocks=blocks _cu_get_xy_projection_from_vol!(xyprojection, vol, size(wavefront.data, 1), slices)
    
    return vol, xyprojection
end