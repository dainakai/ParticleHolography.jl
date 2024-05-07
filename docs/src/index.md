```@meta
CurrentModule = ParticleHolography
```

# ParticleHolography

Documentation for [ParticleHolography](https://github.com/dainakai/ParticleHolography.jl).

A package for particle measurement using inline holography and phase retrieved holography.


!!! note "Note" 

    This package is under development, and none of the functions are guaranteed to work.


## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/dainakai/ParticleHolography.jl.git")
```

## Usage

### GPU-accelerated reconstruction

```julia
using ParticleHolography
using CUDA
using Images

# Load hologram
img = load_gray2float("holo.png")

# Parameters
λ = 0.6328 # Wavelength [μm] 
Δx = 10.0 # Pixel size [μm]
z0 = 220000.0 # Optical distance between the hologram and the front surface of the reconstruction volume [μm]
Δz = 100.0 # Optical distance between the reconstructed slices [μm]
datlen = 1024 # Data length

# Prepare the transfer functions
d_sqr = cu_transfer_sqrt_arr(datlen, λ, Δx)
d_tf = cu_transfer(z0, datlen, λ, d_sqr)
d_slice = cu_transfer(Δz, datlen, λ, d_sqr)

# Reconstruction
d_xyproj = cu_get_reconst_xyprojectin(cu(ComplexF32.(sqrt.(img))), d_tf, d_slice, 500)

# Save the result
save("xyprojection.png", Array(d_xyproj)) # Copy the d_xyproj to host memory with Array()
```

![holo.png](assets/holo.png)

![xyprojection.png](assets/xyprojection.png)

```@docs
load_gray2float
cu_transfer_sqrt_arr
cu_transfer
cu_get_reconst_xyprojectin
```

### CPU-based reconstruction

Preparing...

## Index

```@index
```

```@autodocs
Modules = [ParticleHolography]
```