# Troubleshooting

## `backend(:cuda)` says CUDA is unavailable

`Pkg.add("CUDA")` is not enough; run `using CUDA` in the current process so the
extension activates. Then run `CUDA.functional()` and `CUDA.versioninfo()`.
Check `nvidia-smi` outside Julia. The CUDA CI treats a missing GPU as failure,
not a skipped test.

## `backend(:metal)` says Metal is unavailable

Use Apple silicon, macOS 14+, Julia 1.10–1.13, and Metal.jl 1.10+. Run
`using Metal; Metal.functional(); Metal.versioninfo()`. Intel Macs do not meet
the supported v1 contract.

## `available_backends()` reports only CPU

It reports loaded extensions. Run `using CUDA` or `using Metal` before calling
it. If an optional package is intentionally absent, CPU-only import is working
as designed.

## Out of memory

Inspect `memory_diagnostic(backend(), working_shape, request)` before building
the plan.
If only a projection is needed, use
`ReconstructionRequest(slices; volume=nothing, min_projection=Float32)`.
Otherwise reduce the slice count or crop size, process one frame at a time, and
reuse plans.
Setting an output to `N0f8` saves memory but loses measurement precision.
`check_memory=false` bypasses the conservative check and should be used only
after independently confirming that the allocation is safe.

## Reconstruction is finite but physically wrong

Check, in order: image shape, wavelength/pixel-pitch units, distance units,
propagation signs, front and final depth, camera order, phase-plane separation,
and whether the second camera image was corrected. The package cannot identify
a unit mix from numeric values alone.

## No particles are returned

Inspect the raw `Float32` projection and several volume slices. Verify that dark
particles satisfy `volume .<= threshold`, the boxes cover more than one slice,
and their inclusive XY area is at least 10 pixels. Try a coarse threshold scan;
do not tune only one atypical frame.

## Plotting function says Plots.jl is required

Plotting is optional. Install Plots.jl and run `using Plots` before calling
`particleplot` or `trajectoryplot`. On a headless Linux runner set
`GKSwstype=100`.

## Old `cu_*` code warns

The compatibility wrappers remain for v1 but require `using CUDA`.
Follow [Migrate from v0.2](@ref) to select a default backend once and gain plan
reuse.
