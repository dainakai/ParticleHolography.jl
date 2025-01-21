```@meta
CurrentModule = ParticleHolography
```

# ParticleHolography

Documentation for [ParticleHolography](https://github.com/dainakai/ParticleHolography.jl).

A package for particle measurement using inline holography.


## What you can do with ParticleHolography.jl

- Volume reconstruction from inline holograms using NVIDIA GPUs (CUDA.jl)
- Detect particles in the reconstructed volume (image stack)
- Visualize the particle trajectories

```@raw html
<div style="display:flex; align-items:flex-start;">
   <div style="flex:1; margin-right:10px;">
       <img src="assets/outhologif.gif" alt="4000 fps droplet holograms" style="max-width:100%; height:auto;">
       <p style="text-align:center;">Droplet holograms @4000 fps</p>
   </div>
   <div style="flex:1.23;">
       <img src="assets/sub_animated_3d.gif" alt="Particle trajectories" style="max-width:100%; height:auto;">
       <p style="text-align:center;">Particle trajectories</p>
   </div>
</div>
```

## Installation

```julia
using Pkg
Pkg.add("ParticleHolography")
```

## Demo

For a quick demo, please see the [ParticleHolography.jl demo notebook](https://github.com/dainakai/phdemo)