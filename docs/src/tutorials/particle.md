# Particle handling

This section explains the method of detecting particles from the hologram reconstruction volume, performing particle pairing on time-series data, and tracking particle trajectories.

## Particle detection

We start by detecting particles from the reconstructed image stack of the hologram. The process begins with binarizing the reconstruction stack using a global threshold and then performing connected component labeling to create bounding boxes for the particles. Various methods and values should be considered when determining the global threshold. It's advisable to use tools like ImageJ to adjust and decide on the threshold. The `cu_get_reconst_vol` function returns a reconstruction stack of type `N0f8` by default, which can be saved as an nrrd file using `NRRD.jl` and opened in Fiji as an 8-bit grayscale image stack (Tiff stack files are easier to open in ImageJ, but that process seems to be more complex). Among the binarization methods provided by these tools, the Minimum method often yields the best threshold. For time-series data, it's good to use a consistent threshold throughout the measurement series if lighting conditions remain unchanged.

Although not shown in the following example, when working with experimental data or reconstructed images where artifacts may occur and you want to set a strict threshold, it's beneficial to use the `cu_dilate` function to expand the binarized volume `d_bin_vol`. Ideally, the particle mask `d_bin_vol` should cover an area slightly larger than the true particle image. This is because depth position determination methods may examine the difference between the particle image and its surroundings or utilize the gradient intensity of the particle contour. This expansion process is a normal dilation operation and is executed on the GPU, similar to the reconstruction process.

The connected component labeling applied to the binarized volume is not a strict 3D labeling in the conventional sense. First, 2D connected component labeling [playne](@cite) ([GPU implementation](https://github.com/FolkeV/CUDA_CCL)) is performed on each slice, and then elements that overlap even partially in x-y coordinates across all slices are considered identical. In other words, this process does not distinguish between different particles that overlap in x-y coordinates. As long as particles do not overlap, this method provides rough Bounding Boxes for particles and suppresses ghost particles. After completing the labeling and Bounding Box creation for all slices, we exclude those that can be considered ghost particles or are too small to observe. For more details, refer to the [`finalize_particle_neighborhoods!`](@ref finalize_particle_neighborhoods!) function.

The particle Bounding Boxes are provided in dictionary format. The `key` is a UUID, and the `value` is in the format `[xmin, ymin, zmin, xmax, ymax, zmax]`. This is used to evaluate the coordinates and size of the particles. The UUID keys assigned during the creation of Bounding Boxes are carried over to create a new dictionary, which is then returned. If only coordinates are evaluated, the key is a 3-dimensional vector; if particle size is also evaluated, it's 4-dimensional. Evaluation functions can take depth position determination methods and particle size evaluation methods as arguments. For more details, refer to the respective functions: [`particle_coordinates`](@ref particle_coordinates), [`particle_coor_diams`](@ref particle_coor_diams)

```julia
using ParticleHolography
using CUDA

λ = 0.6328 # Wavelength [μm]
Δx = 10.0 # Pixel size [μm]
z0 = 80000.0 # Optical distance between the hologram and the front surface of the reconstruction volume [μm]
Δz = 100.0 # Optical distance between the reconstructed slices [μm]
datlen = 1024 # Data length
slices = 1000 # Number of slices
pr_dist = 80000.0 # Optical distance between the two holograms [μm]
pr_iter = 9
threshold = 30/255

img1 = load_gray2float("./holo1.png")
img2 = load_gray2float("./holo2.png")

d_sqr = cu_transfer_sqrt_arr(datlen, λ, Δx)
d_tf = cu_transfer(-z0, datlen, λ, d_sqr)
d_slice = cu_transfer(-Δz, datlen, λ, d_sqr)
d_pr = cu_transfer(pr_dist, datlen, λ, d_sqr)
d_pr_inv = cu_transfer(-pr_dist, datlen, λ, d_sqr)

# Phase retrieval using Gerchberg-Saxton algorithm
d_holo = cu_phase_retrieval_holo(cu(img1), cu(img2), d_pr, d_pr_inv, pr_iter, datlen)

# Reconstruction
d_vol = cu_get_reconst_vol(d_holo, d_tf, d_slice, slices)

# Binarization
d_bin_vol = d_vol .<= threshold

particle_bbs = particle_bounding_boxes(d_bin_vol)
particle_coords = particle_coordinates(particle_bbs, d_vol)
```

## Scatter plotting





## References

```@bibliography
Pages = ["particle.md"]
```