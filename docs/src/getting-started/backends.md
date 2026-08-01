# Choose CPU, Metal, or CUDA

The backend is an explicit object stored in reusable plans. ParticleHolography
does not change a process-global GPU preference. This makes CPU/GPU comparisons
possible in one Julia process and prevents the package from changing another
library's device choice.

## CPU

CPU needs no optional package and works on Linux, macOS, and Windows.

```julia
using ParticleHolography
b = backend(:cpu)
```

Use CPU first when checking units, array shapes, and thresholds. It can be slow
for a full 1024×1024×1000 volume, but it is the numerical reference used by the
shared accelerator tests.

## NVIDIA CUDA

```julia
using Pkg
Pkg.add("CUDA")

using ParticleHolography
using CUDA                 # activates ParticleHolographyCUDAExt
CUDA.functional() || error("CUDA is not functional")
b = backend(:cuda)
```

`backend(:cuda; device=0)` selects a zero-based CUDA device before allocation.
Use `CUDA.versioninfo()` when reporting setup problems.

## Apple Metal

Metal requires an Apple-silicon Mac, macOS 14 or later, Julia 1.10–1.13, and
Metal.jl 1.10 or later.

```julia
using Pkg
Pkg.add("Metal")

using ParticleHolography
using Metal                # activates ParticleHolographyMetalExt
Metal.functional() || error("Metal is not functional")
b = backend(:metal)
```

Metal support is tested on GitHub's arm64 `macos-15` runner with the same small
contract as CUDA. Metal.jl describes the runner GPU as paravirtualized and
best-effort; a CI failure is reported rather than silently skipping the test.
See the [Metal.jl 1.10 release notes](https://juliagpu.org/post/2026-07-01-metal-1.10/index.html)
and [GitHub-hosted runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
for the maintained platform requirements and runner architecture.

## One processing function

The processing function does not mention a GPU package:

```julia
function reconstruct_frame(hologram, b; wavelength, pixel_pitch,
                           front_distance, slice_spacing, slices)
    n = size(hologram, 1)
    sqrt_part = transfer_sqrt(b, n, wavelength, pixel_pitch)
    front = transfer(b, -front_distance, wavelength, sqrt_part)
    step = transfer(b, -slice_spacing, wavelength, sqrt_part)
    wavefront = gabor_wavefront(b, hologram)
    plan = ReconstructionPlan(b, front, step)
    return reconstruct_and_projection(plan, wavefront, slices)
end
```

Only `b` and the optional package loaded by the environment change.

## What stays on the device

| Stage | CPU | Metal | CUDA |
|---|---:|---:|---:|
| transfer/wavefront arrays | host | device | device |
| FFT phase retrieval and reconstruction | host | device | device |
| XY dilation | host | device | device |
| per-slice connected components | host | host fallback | host fallback |
| coordinate/diameter metrics | host | host fallback | host fallback |
| calibration PIV | host | host fallback | device kernel |
| plotting and file IO | host | host | host |

The host stages are deliberate compatibility boundaries, not silent backend
changes. `to_host(x)` performs the explicit final copy when saving or comparing
results. `synchronize_backend(b)` is available for timing asynchronous GPU work.

## Discovery and errors

`available_backends()` reports extensions loaded in the current process. An
installed package does not activate its extension until `using CUDA` or
`using Metal` has run. `backend(:auto)` prefers CUDA, then Metal, then CPU, but
explicit selection is recommended for reproducible analysis.
