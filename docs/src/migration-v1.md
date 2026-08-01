# Migrate from v0.2

v1 is a breaking release. CUDA-specific names remain as deprecated wrappers so
an existing NVIDIA workflow can be migrated incrementally.

| v0.2 | v1 |
|---|---|
| `cu_transfer_sqrt_arr(n, λ, dx)` | `propagation_grid(n, λ, dx)` |
| `cu_transfer(z, n, λ, grid)` | `propagation_kernel(z, λ, grid)` |
| `cu_gabor_wavefront(cu(image))` | `gabor_wavefront(image)` |
| `cu_phase_retrieval_holo(...)` | `PhaseRetrievalPlan` + `phase_retrieval` |
| `cu_get_reconst_vol(...)` | `ReconstructionPlan` + `reconstruct` |
| `cu_get_reconst_xyprojection(...)` | `xyprojection` |
| `cu_get_reconst_vol_and_xyprojection(...)` | `reconstruct_and_projection` |
| `cu_dilate(binary)` | `dilate(binary)` |
| `CuTransferSqrtPart` | `PropagationGrid` |
| `CuTransfer` | `PropagationKernel` |
| `CuWavefront` | `Wavefront` |

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
backend(:cpu) # or load CUDA/Metal and select it once
grid = propagation_grid(n, λ, dx)
front = propagation_kernel(-z0, λ, grid)
step = propagation_kernel(-dz, λ, grid)
wave = gabor_wavefront(image)
request = ReconstructionRequest(slices; volume=Float32)
plan = ReconstructionPlan(front, step; request)
volume = reconstruct(plan, wave).volume
```

## Behaviour changes

- CPU-only import no longer loads CUDA, a GPU driver, Makie, or Plots.
- `backend(:cpu)`, `backend(:metal)`, or `backend(:cuda)` sets the process-wide
  default used by later backend-less calls.
- New reconstruction functions default to unclipped `Float32`. Deprecated
  `cu_get_*` wrappers retain clipped `N0f8` defaults where v0.2 did.
- `ReconstructionRequest` produces a requested volume and MinIP together in one
  depth scan and performs a conservative memory preflight.
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
- `transfer_sqrt`/`TransferSqrtPart` and `transfer`/`Transfer` remain deprecated
  aliases for `propagation_grid`/`PropagationGrid` and
  `propagation_kernel`/`PropagationKernel`.

## Package environments

Move CUDA, Metal, and Plots to the application environment that needs them.
ParticleHolography lists them as weak dependencies, so a CPU service or docs
build need not download accelerator/display stacks.
