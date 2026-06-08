using CUDA
using cuFFT
using FixedPointNumbers
using LinearAlgebra

export cu_transfer_sqrt_arr, cu_transfer, cu_gabor_wavefront, cu_phase_retrieval_holo, cu_get_reconst_vol, cu_get_reconst_xyprojection, cu_get_reconst_vol_and_xyprojection, cu_get_reconst_complex_vol
export cu_asm_prop!, cu_2d_pad, cu_get_reconst_vol_and_xyprojection_padded

function _cu_transfer_sqrt_arr!(Plane, datLen, wavLen, dx)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if x <= datLen && y <= datLen
        @inbounds Plane[x, y] = 1.0 - ((x - datLen / 2 - 1.0) * wavLen / datLen / dx)^2 - ((y - datLen / 2 - 1.0) * wavLen / datLen / dx)^2
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
    @cuda threads = threads blocks = blocks _cu_transfer_sqrt_arr!(Plane, datlen, wavlen, dx)
    return CuTransferSqrtPart(Plane)
end

function _cu_transfer!(Plane, z0, datLen, wavLen, d_sqr)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if x <= datLen && y <= datLen
        @inbounds Plane[x, y] = exp(2im * pi * z0 / wavLen * sqrt(d_sqr[x, y]))
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
    @cuda threads = threads blocks = blocks _cu_transfer!(Plane, z0, datLen, wavLen, d_sqr.data)
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
    return CuWavefront(ComplexF32.(sqrt.(holo) .+ 0.0im))
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
    light1_fft = similar(light1)
    light2_fft = similar(light2)
    phi1 = CuArray{Float32}(undef, datlen, datlen)
    phi2 = CuArray{Float32}(undef, datlen, datlen)
    sqrtI1 = sqrt.(holo1)
    sqrtI2 = sqrt.(holo2)
    transfer_fft_order = _transfer_fft_order(transfer)
    invtransfer_fft_order = _transfer_fft_order(invtransfer)
    fft_plan = cuFFT.plan_fft(light1)
    ifft_plan = cuFFT.plan_ifft(light1)

    light1 .= sqrtI1 .+ 0.0im

    for _ in 1:priter
        # STEP1
        LinearAlgebra.mul!(light1_fft, fft_plan, light1)
        light1_fft .= light1_fft .* transfer_fft_order
        LinearAlgebra.mul!(light2, ifft_plan, light1_fft)
        phi2 .= angle.(light2)

        # STEP2
        light2 .= sqrtI2 .* exp.(1.0im .* phi2)

        # STEP3
        LinearAlgebra.mul!(light2_fft, fft_plan, light2)
        light2_fft .= light2_fft .* invtransfer_fft_order
        LinearAlgebra.mul!(light1, ifft_plan, light2_fft)
        phi1 .= angle.(light1)

        # STEP4
        light1 .= sqrtI1 .* exp.(1.0im .* phi1)
    end

    return CuWavefront(light1)
end

"""
    cu_2d_pad(inarr)

Pad the 2D CuArray `inarr` with its mean value to double its size in both dimensions.

# Arguments
- `inarr::CuArray{ComplexF32,2}`: The input 2D CuArray to be padded.
"""
function cu_2d_pad(inarr)
    inlen = size(inarr, 1)
    outarr = CUDA.zeros(ComplexF32, 2*inlen, 2*inlen)
    outlen = 2*inlen
    meaninarr = mean(inarr)

    outarr .= ComplexF32(meaninarr)
    outarr[(outlen÷4+1):(outlen÷4+inlen), (outlen÷4+1):(outlen÷4+inlen)] .= ComplexF32.(inarr)

    return outarr
end


########################  Recontruction functions  ########################

function _normedfloat_to_N0f8(val::AbstractFloat)
    return reinterpret(N0f8, round(UInt8, clamp(val, 0.0, 1.0) * 255))
end

function _transfer_fft_order(transfer::CuTransfer{T}) where {T<:Complex}
    return cuFFT.ifftshift(transfer.data)
end

function _ifft_from_fft_order!(out::AbstractArray{T,2}, fftholo::CuArray{T,2}, ifft_plan) where {T<:Complex}
    LinearAlgebra.mul!(out, ifft_plan, fftholo)
    return nothing
end

function _ifft_and_abs!(out::AbstractArray, fftholo::CuArray{T,2}, return_type::Type, ifft_plan, ifft_output::CuArray{T,2}) where {T<:Complex}
    _ifft_from_fft_order!(ifft_output, fftholo, ifft_plan)
    if return_type in [Float32, Float64, Float16]
        out .= return_type.(clamp.(abs2.(ifft_output), 0.0, 1.0))
    elseif return_type == N0f8
        out .= _normedfloat_to_N0f8.(clamp.(abs2.(ifft_output), 0.0, 1.0))
    else
        throw(ArgumentError("return_type must be Subtype of AbstractFloat or N0f8. Got $return_type"))
    end
    return nothing
end

function _ifft_abs2_float32!(out::AbstractArray{Float32,2}, fftholo::CuArray{T,2}, ifft_plan, ifft_output::CuArray{T,2}) where {T<:Complex}
    _ifft_from_fft_order!(ifft_output, fftholo, ifft_plan)
    out .= Float32.(abs2.(ifft_output))
    return nothing
end


"""
    cu_get_reconst_vol(holo, transfer_front, transfer_dz, slices, return_type)

Reconstruct the observation volume from the `wavefront` using the transfer functions `transfer_front` and `transfer_dz`. `transfer_front` propagates the wavefront to the front of the volume, and `transfer_dz` propagates the wavefront between the slices. `slices` is the number of slices in the volume.

# Arguments
- `wavefront::CuArray{ComplexF32,2}`: The wavefront to reconstruct. In Gabor's holography, this is the square root of the hologram.
- `transfer_front::CuTransfer{ComplexF32}`: The transfer function to propagate the wavefront to the front of the volume. See [`CuTransfer`](@ref).
- `transfer_dz::CuTransfer{ComplexF32}`: The transfer function to propagate the wavefront between the slices.
- `slices::Int`: The number of slices in the volume.
- `return_type::Type`: The return type of the reconstructed volume. Default is `N0f8`.

# Returns
- `CuArray{return_type,3}`: The reconstructed intensity volume.
"""
function cu_get_reconst_vol(wavefront::CuWavefront{ComplexF32}, transfer_front::CuTransfer{ComplexF32}, transfer_dz::CuTransfer{ComplexF32}, slices::Int, return_type::Type=N0f8)
    @assert size(wavefront.data) == size(transfer_front.data) == size(transfer_dz.data) "All arrays must have the same size. Got $(size(wavefront.data)), $(size(transfer_front.data)), $(size(transfer_dz.data))."

    vol = CuArray{return_type}(undef, size(wavefront.data)..., slices)
    fftholo_fft = similar(wavefront.data)
    ifft_output = similar(wavefront.data)
    transfer_front_fft_order = _transfer_fft_order(transfer_front)
    transfer_dz_fft_order = _transfer_fft_order(transfer_dz)
    fft_plan = cuFFT.plan_fft(wavefront.data)
    ifft_plan = cuFFT.plan_ifft(wavefront.data)

    LinearAlgebra.mul!(fftholo_fft, fft_plan, wavefront.data)
    fftholo_fft .= fftholo_fft .* transfer_front_fft_order

    _ifft_and_abs!(view(vol, :, :, 1), fftholo_fft, return_type, ifft_plan, ifft_output)

    for i in 2:slices
        fftholo_fft .= fftholo_fft .* transfer_dz_fft_order
        _ifft_and_abs!(view(vol, :, :, i), fftholo_fft, return_type, ifft_plan, ifft_output)
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
    fftholo_fft = similar(wavefront.data)
    transfer_front_fft_order = _transfer_fft_order(transfer_front)
    transfer_dz_fft_order = _transfer_fft_order(transfer_dz)
    fft_plan = cuFFT.plan_fft(wavefront.data)
    ifft_plan = cuFFT.plan_ifft(wavefront.data)

    LinearAlgebra.mul!(fftholo_fft, fft_plan, wavefront.data)
    fftholo_fft .= fftholo_fft .* transfer_front_fft_order

    _ifft_from_fft_order!(view(vol, :, :, 1), fftholo_fft, ifft_plan)

    for i in 2:slices
        fftholo_fft .= fftholo_fft .* transfer_dz_fft_order
        _ifft_from_fft_order!(view(vol, :, :, i), fftholo_fft, ifft_plan)
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
    fftholo_fft = similar(wavefront.data)
    ifft_output = similar(wavefront.data)
    transfer_front_fft_order = _transfer_fft_order(transfer_front)
    transfer_dz_fft_order = _transfer_fft_order(transfer_dz)
    fft_plan = cuFFT.plan_fft(wavefront.data)
    ifft_plan = cuFFT.plan_ifft(wavefront.data)

    LinearAlgebra.mul!(fftholo_fft, fft_plan, wavefront.data)
    fftholo_fft .= fftholo_fft .* transfer_front_fft_order

    _ifft_abs2_float32!(proj, fftholo_fft, ifft_plan, ifft_output)

    for i in 2:slices
        fftholo_fft .= fftholo_fft .* transfer_dz_fft_order
        _ifft_abs2_float32!(projtmp, fftholo_fft, ifft_plan, ifft_output)
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
    cu_get_reconst_vol_and_xyprojection(wavefront, transfer_front, transfer_dz, slices, return_type)

Reconstruct the observation volume from the `wavefront` and get the XY projection of the volume using the transfer functions `transfer_front` and `transfer_dz`. `transfer_front` propagates the wavefront to the front of the volume, and `transfer_dz` propagates the wavefront between the slices. `slices` is the number of slices in the volume.

# Arguments
- `wavefront::CuWavefront{ComplexF32}`: The wavefront to reconstruct. In Gabor's holography, this is the square root of the hologram.
- `transfer_front::CuTransfer{ComplexF32}`: The transfer function to propagate the wavefront to the front of the volume. See [`CuTransfer`](@ref).
- `transfer_dz::CuTransfer{ComplexF32}`: The transfer function to propagate the wavefront between the slices.
- `slices::Int`: The number of slices in the volume.
- `return_type::Type`: The return type of the reconstructed volume. Default is `N0f8`.

# Returns
- `CuArray{return_type,3}`: The reconstructed volume.
- `CuArray{Float32,2}`: The XY projection of the reconstructed volume.
"""
function cu_get_reconst_vol_and_xyprojection(wavefront::CuWavefront{ComplexF32}, transfer_front::CuTransfer{ComplexF32}, transfer_dz::CuTransfer{ComplexF32}, slices::Int, return_type::Type=N0f8)
    @assert size(wavefront.data) == size(transfer_front.data) == size(transfer_dz.data) "All arrays must have the same size. Got $(size(wavefront.data)), $(size(transfer_front.data)), $(size(transfer_dz.data))."

    vol = CuArray{return_type}(undef, size(wavefront.data)..., slices)
    fftholo_fft = similar(wavefront.data)
    ifft_output = similar(wavefront.data)
    transfer_front_fft_order = _transfer_fft_order(transfer_front)
    transfer_dz_fft_order = _transfer_fft_order(transfer_dz)
    fft_plan = cuFFT.plan_fft(wavefront.data)
    ifft_plan = cuFFT.plan_ifft(wavefront.data)

    LinearAlgebra.mul!(fftholo_fft, fft_plan, wavefront.data)
    fftholo_fft .= fftholo_fft .* transfer_front_fft_order

    _ifft_and_abs!(view(vol, :, :, 1), fftholo_fft, return_type, ifft_plan, ifft_output)

    for i in 2:slices
        fftholo_fft .= fftholo_fft .* transfer_dz_fft_order
        _ifft_and_abs!(view(vol, :, :, i), fftholo_fft, return_type, ifft_plan, ifft_output)
    end

    xyprojection = CuArray{Float32}(undef, size(wavefront.data)...)
    threads = (32, 32)
    blocks = cld.((size(wavefront.data)[1], size(wavefront.data)[2]), threads)
    @cuda threads = threads blocks = blocks _cu_get_xy_projection_from_vol!(xyprojection, vol, size(wavefront.data, 1), slices)

    return vol, xyprojection
end

"""
    cu_get_reconst_vol_and_xyprojection_padded(wavefront, transfer_front, transfer_dz, slices, return_type)

Reconstruct the observation volume from the padded `wavefront` and get the XY projection of the volume using the transfer functions `transfer_front` and `transfer_dz`. `transfer_front` propagates the wavefront to the front of the volume, and `transfer_dz` propagates the wavefront between the slices. `slices` is the number of slices in the volume.

# Arguments
- `wavefront::CuWavefront{ComplexF32}`: The wavefront to reconstruct. In Gabor's holography, this is the square root of the hologram.
- `transfer_front::CuTransfer{ComplexF32}`: The transfer function to propagate the wavefront to the front of the volume. See [`CuTransfer`](@ref).
- `transfer_dz::CuTransfer{ComplexF32}`: The transfer function to propagate the wavefront between the slices.
- `slices::Int`: The number of slices in the volume.
- `return_type::Type`: The return type of the reconstructed volume. Default is `N0f8`.

# Returns
- `CuArray{return_type,3}`: The reconstructed volume.
- `CuArray{Float32,2}`: The XY projection of the reconstructed volume.
"""
function cu_get_reconst_vol_and_xyprojection_padded(wavefront::CuWavefront{ComplexF32}, transfer_front::CuTransfer{ComplexF32}, transfer_dz::CuTransfer{ComplexF32}, slices::Int, return_type::Type=N0f8)
    expected_size = map(x -> 2 * x, size(wavefront.data))
    @assert expected_size == size(transfer_front.data) == size(transfer_dz.data) "size(transfer_front.data) and size(transfer_dz.data) must be equal to 2*size(wavefront.data). Got $(size(wavefront.data)), $(size(transfer_front.data)), $(size(transfer_dz.data))."

    datlen = size(wavefront.data, 1)
    vol = CuArray{return_type}(undef, datlen, datlen, slices)
    padded_wavefront = cu_2d_pad(wavefront.data)
    fftholo_fft = similar(padded_wavefront)
    ifft_output = similar(padded_wavefront)
    transfer_front_fft_order = _transfer_fft_order(transfer_front)
    transfer_dz_fft_order = _transfer_fft_order(transfer_dz)
    fft_plan = cuFFT.plan_fft(padded_wavefront)
    ifft_plan = cuFFT.plan_ifft(padded_wavefront)

    LinearAlgebra.mul!(fftholo_fft, fft_plan, padded_wavefront)
    fftholo_fft .= fftholo_fft .* transfer_front_fft_order

    tmparr = CuArray{return_type}(undef, size(padded_wavefront)...)
    _ifft_and_abs!(tmparr, fftholo_fft, return_type, ifft_plan, ifft_output)
    vol[:, :, 1] .= tmparr[div(datlen,2)+1:3*div(datlen,2), div(datlen,2)+1:3*div(datlen,2)]

    for i in 2:slices
        fftholo_fft .= fftholo_fft .* transfer_dz_fft_order
        _ifft_and_abs!(tmparr, fftholo_fft, return_type, ifft_plan, ifft_output)
        vol[:, :, i] .= tmparr[div(datlen,2)+1:3*div(datlen,2), div(datlen,2)+1:3*div(datlen,2)]
    end

    xyprojection = CuArray{Float32}(undef, size(wavefront.data)...)
    threads = (32, 32)
    blocks = cld.((size(wavefront.data)[1], size(wavefront.data)[2]), threads)
    @cuda threads = threads blocks = blocks ParticleHolography._cu_get_xy_projection_from_vol!(xyprojection, vol, size(wavefront.data, 1), slices)

    return vol, xyprojection
end


"""
    cu_asm_prop!(outholo, inholo, d_sqr, zprop, datlen, λ)

Perform angular spectrum method-based propagation of the wavefront `inholo` by distance `zprop` and store the result in `outholo`. `d_sqr` is the square-root part of the transfer function obtained with `cutransfersqrtarr(datlen, λ, dx)`. `datlen` is the size of the holograms, and `λ` is the wavelength of the light.

# Arguments
- `outholo::CuWavefront{ComplexF32}`: The output wavefront after propagation. See [`CuWavefront`](@ref).
- `inholo::CuWavefront{ComplexF32}`: The input wavefront to propagate. See [`CuWavefront`](@ref).
- `d_sqr::CuTransferSqrtPart{Float32}`: The square-root part of the transfer function obtained with `cutransfersqrtarr(datlen, λ, dx). See [`CuTransferSqrtPart`](@ref).
- `zprop::Float64`: The distance to propagate the wavefront.
- `datlen::Int`: The size of the holograms.
- `λ::Float64`: The wavelength of the light.

# Returns
- `Nothing`: The result is stored in `outholo`.
"""
function cu_asm_prop!(outholo::CuWavefront, inholo::CuWavefront, d_sqr::CuTransferSqrtPart,
                    zprop::Float64, datlen::Int, λ::Float64)
    tf = cu_transfer(zprop, datlen, λ, d_sqr)
    tf_fft_order = _transfer_fft_order(tf)
    fft_arr = similar(inholo.data)
    fft_plan = cuFFT.plan_fft(inholo.data)
    ifft_plan = cuFFT.plan_ifft(inholo.data)

    LinearAlgebra.mul!(fft_arr, fft_plan, inholo.data)
    fft_arr .= fft_arr .* tf_fft_order
    LinearAlgebra.mul!(outholo.data, ifft_plan, fft_arr)
    return nothing
end
