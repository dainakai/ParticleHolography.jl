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
backend(:cpu)

# Apple silicon / macOS 14+
# Pkg.add("Metal"); using Metal; backend(:metal)

# NVIDIA
# Pkg.add("CUDA"); using CUDA; backend(:cuda)
```

The selection is process-wide. After this one setup call, the same processing
code runs on the selected backend. Explicit backend arguments remain available
for side-by-side CPU/GPU comparisons.

## Small reconstruction

```julia
n = size(hologram, 1)
wavelength = 0.6328       # μm
pixel_pitch = 10.0        # μm

grid = propagation_grid(n, wavelength, pixel_pitch)
front = propagation_kernel(-80_000.0, wavelength, grid)
step = propagation_kernel(-100.0, wavelength, grid)
wavefront = gabor_wavefront(hologram)

request = ReconstructionRequest(1_000;
    volume=Float32,
    min_projection=N0f8,
)
plan = ReconstructionPlan(front, step; request)
@show memory_diagnostic(plan, request)

result = reconstruct(plan, wavefront)
volume = result.volume
projection = to_host(result.min_projection)
```

The volume and minimum-intensity projection are produced in one depth scan.
Plans reuse FFT plans and work buffers across frames and reject an allocation
that the conservative memory preflight considers unsafe. Deprecated v0.2
`cu_*`, `transfer_sqrt`, and `transfer` names remain as migration wrappers.

## Learn and run real data

- [Stable documentation](https://dainakai.github.io/ParticleHolography.jl/stable/)
- [Development documentation](https://dainakai.github.io/ParticleHolography.jl/dev/)
- [phdemo experimental-data tutorial](https://github.com/dainakai/phdemo)
- [v0.2 → v1 migration guide](https://dainakai.github.io/ParticleHolography.jl/dev/migration-v1/)

Start with the documentation's CPU quickstart. Then use phdemo's `doctor` and
`smoke` commands before processing the full droplet dataset.
