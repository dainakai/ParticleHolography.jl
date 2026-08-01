# Choose CPU, Metal, or CUDA

Call `backend(:cpu)`, `backend(:metal)`, or `backend(:cuda)` once during program
setup.
The selected backend becomes the process-wide default for later calls that omit
a backend argument, and each reusable plan records that selection.

```julia
using ParticleHolography

backend(:cpu)
grid = propagation_grid(1024, 0.6328, 10.0)
wavefront = gabor_wavefront(hologram)
```

`backend()` returns the current selection.
Changing it while concurrent tasks are using ParticleHolography is unsupported.
For simultaneous CPU/GPU work or numerical comparisons, retain the returned
object and pass it explicitly, such as `propagation_grid(cpu, ...)`.

## CPU

CPU needs no optional package and works on Linux, macOS, and Windows.

```julia
using ParticleHolography
backend(:cpu)
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
backend(:cuda)
```

`backend(:cuda; device=0)` selects a zero-based CUDA device before allocation.
Use `CUDA.versioninfo()` when reporting setup problems.
See the official [CUDA.jl documentation](https://cuda.juliagpu.org/stable/),
[CuArray guide](https://cuda.juliagpu.org/stable/usage/array/), and
[GPU memory guide](https://cuda.juliagpu.org/stable/usage/memory/) for runtime,
array, and memory-pool details.

## Apple Metal

Metal requires an Apple-silicon Mac, macOS 14 or later, Julia 1.10–1.13, and
Metal.jl 1.10 or later.

```julia
using Pkg
Pkg.add("Metal")

using ParticleHolography
using Metal                # activates ParticleHolographyMetalExt
Metal.functional() || error("Metal is not functional")
backend(:metal)
```

Metal support is tested on GitHub's arm64 `macos-15` runner with the same small
contract as CUDA. Metal.jl describes the runner GPU as paravirtualized and
best-effort; a CI failure is reported rather than silently skipping the test.
See the [Metal.jl 1.10 release notes](https://juliagpu.org/post/2026-07-01-metal-1.10/index.html)
and [GitHub-hosted runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
for the maintained platform requirements and runner architecture.
The official [Metal.jl documentation](https://metal.juliagpu.org/stable/) and
[MtlArray guide](https://metal.juliagpu.org/stable/usage/array/) explain device
arrays and supported element types.

## One processing function

The processing function does not mention a GPU package:

```julia
function reconstruct_frame(hologram; wavelength, pixel_pitch,
                           front_distance, slice_spacing, slices)
    n = size(hologram, 1)
    grid = propagation_grid(n, wavelength, pixel_pitch)
    front = propagation_kernel(-front_distance, wavelength, grid)
    step = propagation_kernel(-slice_spacing, wavelength, grid)
    wavefront = gabor_wavefront(hologram)
    request = ReconstructionRequest(slices;
        volume=Float32,
        min_projection=N0f8,
    )
    plan = ReconstructionPlan(front, step; request)
    return reconstruct(plan, wavefront)
end
```

Only the setup call and optional package loaded by the environment change.
The processing function is identical on all three backends.

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
results. `synchronize_backend()` is available for timing asynchronous GPU work.

CPU FFT execution is supplied by [FFTW.jl](https://github.com/JuliaMath/FFTW.jl).
The shared planning interface comes from
[AbstractFFTs.jl](https://juliamath.github.io/AbstractFFTs.jl/stable/api/),
which CUDA.jl and Metal.jl implement for their device arrays.

## Discovery and errors

`available_backends()` reports extensions loaded in the current process. An
installed package does not activate its extension until `using CUDA` or
`using Metal` has run. `backend(:auto)` prefers CUDA, then Metal, then CPU, but
explicit selection is recommended for reproducible analysis.

Both Plots.jl and ParticleHolography export a function named `backend`.
If an interactive session runs `using Plots`, select the execution backend as
`ParticleHolography.backend(:cpu)` or import only the plotting names you need.
