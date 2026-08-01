# Migrate from v0.2

v1 is a breaking release. CUDA-specific names remain as deprecated wrappers so
an existing NVIDIA workflow can be migrated incrementally.

| v0.2 | v1 |
|---|---|
| `cu_transfer_sqrt_arr(n, λ, dx)` | `transfer_sqrt(b, n, λ, dx)` |
| `cu_transfer(z, n, λ, s)` | `transfer(b, z, λ, s)` |
| `cu_gabor_wavefront(cu(image))` | `gabor_wavefront(b, image)` |
| `cu_phase_retrieval_holo(...)` | `PhaseRetrievalPlan` + `phase_retrieval` |
| `cu_get_reconst_vol(...)` | `ReconstructionPlan` + `reconstruct` |
| `cu_get_reconst_xyprojection(...)` | `xyprojection` |
| `cu_get_reconst_vol_and_xyprojection(...)` | `reconstruct_and_projection` |
| `cu_dilate(binary)` | `dilate(binary)` |
| `CuWavefront`, `CuTransfer`, … | `Wavefront`, `Transfer`, … |

## Minimal rewrite

```julia
# v0.2
using ParticleHolography, CUDA
s = cu_transfer_sqrt_arr(n, λ, dx)
front = cu_transfer(-z0, n, λ, s)
step = cu_transfer(-dz, n, λ, s)
wave = cu_gabor_wavefront(cu(image))
volume = cu_get_reconst_vol(wave, front, step, slices)

# v1
using ParticleHolography
b = backend(:cpu) # or load CUDA/Metal and choose that backend
s = transfer_sqrt(b, n, λ, dx)
front = transfer(b, -z0, λ, s)
step = transfer(b, -dz, λ, s)
wave = gabor_wavefront(b, image)
plan = ReconstructionPlan(b, front, step)
volume = reconstruct(plan, wave, slices)
```

## Behaviour changes

- CPU-only import no longer loads CUDA, a GPU driver, Makie, or Plots.
- New reconstruction functions default to unclipped `Float32`. Deprecated
  `cu_get_*` wrappers retain clipped `N0f8` defaults where v0.2 did.
- Transfer and filter arrays use FFT-native order. Manually constructed centred
  arrays must be reordered before wrapping.
- Combined projection uses unquantised `Float32` intensity even if the stored
  volume is `N0f8`.
- Bounding-box overlap is inclusive, UUID generation no longer restarts a fixed
  random generator, and dictionary deletion is performed safely.
- Tracking preserves graph branches and rejects duplicate UUIDs across frames.
- Tests no longer write images or JSON into tracked fixture paths.
- `CUDABackend` and `MetalBackend` implementation types are not exported because
  CUDA.jl/Metal.jl export names with the same spelling. Use `backend(:cuda)` and
  `backend(:metal)`.

## Package environments

Move CUDA, Metal, and Plots to the application environment that needs them.
ParticleHolography lists them as weak dependencies, so a CPU service or docs
build need not download accelerator/display stacks.
