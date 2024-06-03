```@meta
CurrentModule = ParticleHolography
```

# ParticleHolography

Documentation for [ParticleHolography](https://github.com/dainakai/ParticleHolography.jl).

A package for particle measurement using inline holography.


!!! note "Note" 

    This package is under development, and none of the functions are guaranteed to work.


## What you can do with ParticleHolography.jl

- Perform inline holographic reconstruction using NVIDIA GPUs (CUDA.jl)
- Detect particles in reconstructed volumes
- Visualize the particle trajectories

```@raw html
<div style="display:flex; align-items:flex-start;">
   <div style="flex:1; margin-right:10px;">
       <img src="assets/holo.gif" alt="4000 fps droplet holograms" style="max-width:100%; height:auto;">
       <p style="text-align:center;">Droplet holograms @4000 fps</p>
   </div>
   <div style="flex:1.23;">
       <img src="assets/traj.gif" alt="Particle trajectories" style="max-width:100%; height:auto;">
       <p style="text-align:center;">Particle trajectories</p>
   </div>
</div>
```

## Installation

```julia
using Pkg
Pkg.add(url="ParticleHolography")
```