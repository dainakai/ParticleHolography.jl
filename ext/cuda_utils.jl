using ProgressMeter

# COV_EXCL_START
function _cu_kernel_vote!(votevol, nimg, H, W)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= H && j <= W
        @inbounds val = Int(nimg[i, j]) + 1
        @inbounds votevol[val, i, j] += Int32(1)
    end
    return
end

function _cu_kernel_argmax!(out_u8, votevol, H, W, B)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= H && j <= W
        maxv = Int32(-1)
        arg = Int32(1)
        @inbounds for b in 1:B
            v = votevol[b, i, j]
            if v > maxv
                maxv = v
                arg = Int32(b)
            end
        end
        @inbounds out_u8[i, j] = UInt8(arg - 1)
    end
    return
end
# COV_EXCL_STOP

"""
    cu_make_background_mode(grayimglist)

Compute the per-pixel intensity mode from the list returned by load_grayimg() and return it as a background image. The computation uses CUDA.

# Arguments
- `grayimglist::Vector{Array{N0f8, 2}}`: A list of grayscale images as Array{N0f8, 2}.

# Returns
- `Array{Float64, 2}`: The background image.
"""
function cu_make_background_mode(grayimglist)
    _assert_cuda_functional(:cu_make_background_mode)
    H, W = size(grayimglist[1])
    B = 256

    votevol = CUDA.zeros(Int32, B, H, W)

    threads = (32, 32)
    blocks = (cld(H, threads[1]), cld(W, threads[2]))

    nimg_gpu = CuArray{UInt8}(undef, H, W)

    @showprogress for img in grayimglist
        if eltype(img) === UInt8
            copyto!(nimg_gpu, img)
        else
            nimg_u8 = round.(UInt8, clamp.(Float32.(img) .* 255, 0, 255))
            copyto!(nimg_gpu, nimg_u8)
        end
        @cuda threads=threads blocks=blocks _cu_kernel_vote!(votevol, nimg_gpu, H, W)
    end

    CUDA.synchronize()

    mode_u8 = CuArray{UInt8}(undef, H, W)
    @cuda threads=threads blocks=blocks _cu_kernel_argmax!(mode_u8, votevol, H, W, B)
    CUDA.synchronize()

    mode_host = Array(mode_u8)
    background = Array{Float64}(undef, H, W)
    @. background = mode_host / 255.0
    return background
end
