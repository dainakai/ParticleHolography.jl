# Gabor holography
## [GPU-accelerated Gabor reconstruction](@id gabor_reconst)

Please refer to [Gabor holography](@ref gabor_explain) for the principles of this method. The code below is an example of performing inline holographic reconstruction using an NVIDIA GPU (CUDA.jl). Your computer needs to be ready to use NVIDIA GPUs with CUDA.jl. It reconstructs a volume of size `datlenΔx` x `datlenΔx` x `slicesΔz` when the camera plane is considered as the ``xy`` plane and the direction perpendicular to the camera plane, which is the optical axis, is the ``z`` axis. In this example case, it creates an ``xy`` projection image of the reconstructed volume by taking the minimum value of the ``z`` axis profile at each pixel in the ``xy`` plane of the reconstructed volume. The operation of extracting the xy projection image from the volume can be expressed by the following equation (details in [`cu_get_reconst_xyprojection`](@ref cu_get_reconst_xyprojection)):

```math
\mathrm{xyproj}(x, y) = \min_{z} \left\{ \mathrm{rcstvol}(x, y, z) \right\}
```

Specify the hologram you want to reconstruct and the parameters, and save the projection image as `xyprojection.png`. 

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

# Reconstruction
d_xyproj = cu_get_reconst_xyprojection(d_wavefront, d_tf, d_slice, slices)

# Save the result
save("xyprojection_gabor.png", Array(d_xyproj)) # Copy the d_xyproj to host memory with Array()
```

```@raw html
<div style="display:flex; align-items:flex-start;">
   <div style="flex:1; margin-right:10px;">
       <img src="../../assets/holo.png" alt="Input hologram image" style="max-width:100%; height:auto;">
       <p style="text-align:center;">Input hologram image</p>
   </div>
   <div style="flex:1;">
       <img src="../../assets/xyprojection.png" alt="Output xy projection image" style="max-width:100%; height:auto;">
       <p style="text-align:center;">Output xy projection image</p>
   </div>
</div>
```
