# Low-pass filtering
## Introduction

In [fugal](@cite) and other works, applying a low-pass filter to holograms is used to reduce phase distribution errors and artifacts, as well as to homogenize the reconstructed image. Furthermore, by limiting the bandwidth more strictly than the necessary and sufficient range to suppress phase undersampling, it is possible to reduce position detection errors caused by particle image elongation. Conversely, the increase in brightness of the bright-field reconstructed image due to the low-pass filter can make it difficult to determine the threshold for binarization and complicate particle size evaluation. While the effects of low-pass filtering have both advantages and disadvantages, it is often beneficial for improving certain metrics.

## Implementation

A low-pass filter can be applied to the Fourier domain of a hologram (treated as a [`CuWavefront`](@ref ParticleHolography.CuWavefront) structure) by taking the Hadamard product with a `NormedFloat` array of the same shape (see [`cu_apply_low_pass_filter!`](@ref cu_apply_low_pass_filter!) and [`cu_apply_low_pass_filter`](@ref cu_apply_low_pass_filter)). In ParticleHolography.jl, low-pass filters are defined by the [`CuLowPassFilter`](@ref ParticleHolography.CuLowPassFilter) structure, which is defined on the GPU device. By default, a rectangular window low-pass filter [`cu_rectangle_filter`](@ref cu_rectangle_filter) or a super-Gaussian function (n=3) low-pass filter [`cu_super_gaussian_filter`](@ref cu_super_gaussian_filter) can be used. For determining the cutoff frequency, please refer to the function documentation or the original research papers.

## Example

```julia
using ParticleHolography
using CUDA
using Images

# Load hologram
img = load_gray2float("./test/holo1.png")

# Parameters
λ = 0.6328 # Wavelength [μm] 
Δx = 10.0 # Pixel size [μm]
z0 = 80000.0 # Optical distance between the hologram and the front surface of the reconstruction volume [μm]
Δz = 100.0 # Optical distance between the reconstructed slices [μm]
datlen = 1024 # Data length
slices = 1000 # Number of slices

# Prepare the transfer functions
d_sqr = cu_transfer_sqrt_arr(datlen, λ, Δx)
d_tf = cu_transfer(-z0, datlen, λ, d_sqr)
d_slice = cu_transfer(-Δz, datlen, λ, d_sqr)

# Make a wavefront
d_wavefront = cu_gabor_wavefront(img)

# Apply low-pass filter
d_lpf = cu_super_gaussian_filter(z0 + Δz * slices, λ, datlen, Δx)
cu_apply_low_pass_filter!(d_wavefront, d_lpf)

# Reconstruction
d_xyproj = cu_get_reconst_xyprojection(d_wavefront, d_tf, d_slice, slices)
```

## References

```@bibliography
Pages = ["lowpassfilter.md"]
```