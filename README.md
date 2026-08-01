# ParticleHolography.jl

[![Documentation](https://github.com/dainakai/ParticleHolography.jl/actions/workflows/Documentation.yml/badge.svg)](https://github.com/dainakai/ParticleHolography.jl/actions/workflows/Documentation.yml)
[![CPU](https://github.com/dainakai/ParticleHolography.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/dainakai/ParticleHolography.jl/actions/workflows/CI.yml)
[![Metal](https://github.com/dainakai/ParticleHolography.jl/actions/workflows/Metal.yml/badge.svg)](https://github.com/dainakai/ParticleHolography.jl/actions/workflows/Metal.yml)
[![CUDA](https://github.com/dainakai/ParticleHolography.jl/actions/workflows/CUDA.yml/badge.svg)](https://github.com/dainakai/ParticleHolography.jl/actions/workflows/CUDA.yml)
[![Coverage](https://codecov.io/gh/dainakai/ParticleHolography.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/dainakai/ParticleHolography.jl)

ParticleHolography.jl reconstructs inline holograms, detects particles in 3-D
intensity volumes, and links detections into trajectories. Version 1 uses one
high-level API on CPU, Apple Metal, and NVIDIA CUDA.

## Install

```julia
using Pkg
Pkg.add("ParticleHolography")
```

CPU works without a GPU or display package. Add only the accelerator used by
your application:

```julia
using ParticleHolography
b = backend(:cpu)

# Apple silicon / macOS 14+
# Pkg.add("Metal"); using Metal; b = backend(:metal)

# NVIDIA
# Pkg.add("CUDA"); using CUDA; b = backend(:cuda)
```

## Small reconstruction

```julia
n = size(hologram, 1)
wavelength = 0.6328       # μm
pixel_pitch = 10.0        # μm

sqrt_part = transfer_sqrt(b, n, wavelength, pixel_pitch)
front = transfer(b, -80_000.0, wavelength, sqrt_part)
step = transfer(b, -100.0, wavelength, sqrt_part)
wavefront = gabor_wavefront(b, hologram)

plan = ReconstructionPlan(b, front, step)
volume, projection = reconstruct_and_projection(plan, wavefront, 1_000)
projection_host = to_host(projection)
```

Plans reuse FFT plans and work buffers across frames. New reconstruction APIs
return unclipped `Float32` by default. Deprecated v0.2 `cu_*` names remain as
CUDA wrappers for incremental migration.

## Learn and run real data

- [Stable documentation](https://dainakai.github.io/ParticleHolography.jl/stable/)
- [Development documentation](https://dainakai.github.io/ParticleHolography.jl/dev/)
- [phdemo experimental-data tutorial](https://github.com/dainakai/phdemo)
- [v0.2 → v1 migration guide](https://dainakai.github.io/ParticleHolography.jl/dev/migration-v1/)

Start with the documentation's CPU quickstart. Then use phdemo's `doctor` and
`smoke` commands before processing the full droplet dataset.
