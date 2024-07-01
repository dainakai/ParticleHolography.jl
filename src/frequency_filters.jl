using CUDA

export cu_rectangle_filter, cu_super_gaussian_filter

function _rect_filter!(arr, maxi, datlen)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if x>0 && x<=datlen && y>0 && y<=datlen
        if (x-datlen/2 +1)^2 + (y-datlen/2)^2 >= maxi^2
            arr[y,x] = 0.0
        end
    end
    return nothing
end

"""
    cu_rectangle_filter(prop_dist::AbstractFloat, wavlen::AbstractFloat, imglen::Int, pixel_picth::AbstractFloat)

Creates a low pass filter with a rectangular window. This can be multiplied with the Fourier transform of wavefront to get the low pass filtered wavefront after propagation. See Eq. 13 and 14 in (Fugal, 2009, https://doi.org/10.1088/0957-0233/20/7/075501)

# Arguments
- `prop_dist::AbstractFloat`: The maximum propagation distance of recorded objects.
- `wavlen::AbstractFloat`: The wavelength of the light.
- `imglen::Int`: The length of the image.
- `pixel_picth::AbstractFloat`: The pixel pitch of the image.

# Returns
- `CuLowPassFilter`: The low pass filter as a CuLowPassFilter object.
"""
function cu_rectangle_filter(prop_dist::AbstractFloat, wavlen::AbstractFloat, imglen::Int, pixel_picth::AbstractFloat)
    arr = CUDA.ones(Float32, (imglen, imglen))
    maxi = 1/wavlen * imglen^2 * pixel_picth^2 / sqrt(4.0*prop_dist^2 + imglen^2 * pixel_picth^2)
    threads = (32,32)
    blocks = cld.((imglen, imglen), threads)
    @cuda threads=threads blocks=blocks _rect_filter!(arr, maxi, imglen)
    return CuLowPassFilter(arr)
end

function _super_gaussian_filter!(out, σ_x, datlen, dx)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if x>0 && x<=datlen && y>0 && y<=datlen
        # if (x-datlen/2 +1)^2 + (y-datlen/2)^2 >= maxi^2
            out[y,x] = exp(-1/2 * (((x-datlen/2 +1)/datlen/dx/σ_x)^2 + ((y-datlen/2 +1)/datlen/dx/σ_x)^2 )^3)
        # end
    end
    return nothing
end

"""
    cu_super_gaussian_filter(prop_dist::AbstractFloat, wavlen::AbstractFloat, imglen::Int, pixel_picth::AbstractFloat)

Creates a low pass filter with a super Gaussian window. This can be multiplied with the Fourier transform of wavefront to get the low pass filtered wavefront after propagation. See Eq. 15 in (Fugal, 2009, https://doi.org/10.1088/0957-0233/20/7/075501)

# Arguments
- `prop_dist::AbstractFloat`: The maximum propagation distance of recorded objects.
- `wavlen::AbstractFloat`: The wavelength of the light.
- `imglen::Int`: The length of the image.
- `pixel_picth::AbstractFloat`: The pixel pitch of the image.

# Returns
- `CuLowPassFilter`: The low pass filter as a CuLowPassFilter object.
"""
function cu_super_gaussian_filter(prop_dist::AbstractFloat, wavlen::AbstractFloat, imglen::Int, pixel_picth::AbstractFloat)
    arr = CUDA.zeros(Float32, (imglen, imglen))
    maxi = 1/wavlen * imglen^2 * pixel_picth^2 / sqrt(4.0*prop_dist^2 + imglen^2 * pixel_picth^2)
    σ_x = maxi/(imglen*pixel_picth)/(2.0*log(2.0))^(1/6)
    threads = (32,32)
    blocks = cld.((imglen, imglen), threads)
    @cuda threads=threads blocks=blocks _super_gaussian_filter!(arr, σ_x, imglen, pixel_picth)
    return CuLowPassFilter(arr)
end