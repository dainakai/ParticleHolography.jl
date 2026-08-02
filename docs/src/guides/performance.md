# Performance and memory

## Reuse plans first

For a time series, create propagation kernels and both FFT plans outside the
frame loop.
Reusing `PhaseRetrievalPlan` and `ReconstructionPlan` removes repeated FFT
planning and major work-buffer allocation.
This is the principal v1 hot-path optimisation.

## Request outputs together

`ReconstructionRequest` lets one depth scan fill any supported volume and MinIP
combination.
Intensity is calculated once per plane when either a real volume or MinIP needs
it.
No full volume is allocated when `volume=nothing`, and no intensity plane is
calculated for a complex-only volume.

```julia
request = ReconstructionRequest(1_000;
    volume=Float32,
    min_projection=N0f8,
)
plan = ReconstructionPlan(front, step; request)
result = reconstruct(plan, wavefront)
```

## Estimate before allocating

A dense volume uses approximately

```text
height × width × slices × sizeof(output element)
```

A 1024×1024×1000 `Float32` volume alone is about 3.91 GiB.
`N0f8` is four times smaller but clips to 0–1 and quantises to 256 levels.
Complex volumes require 8 bytes per voxel for `ComplexF32` and 16 for
`ComplexF64`.
The current propagation workspace remains `ComplexF32`; `ComplexF64` selects
output storage rather than higher-precision propagation.

The package also accounts conservatively for propagation planes, FFT
workspaces, MinIP accumulation, padding, and Metal conversion staging:

```julia
request = ReconstructionRequest(1_000;
    volume=Float32,
    min_projection=N0f8,
)
diagnostic = memory_diagnostic(backend(), (1024, 1024), request)
display(diagnostic)
```

`diagnostic.safe` compares the estimate multiplied by the safety factor with
currently available host, CUDA device, or Metal unified memory.
Plan constructors and reconstruction calls perform the same check and stop
before an unsafe large allocation.
The default safety factor is 1.2 and can be changed with
`memory_safety_factor`.
Use `check_memory=false` only after independently establishing that the
allocation is safe.

## Padding without a padded volume

`reconstruct_padded(plan, wavefront; mode=:mean)` propagates on the larger plan
shape but stores only the central original field of view.
This avoids allocating a padded three-dimensional result.
The remaining cost is the larger FFT workspace and one padded input plane,
which the memory diagnostic includes.

## Avoid hidden transfers

- Load and correct camera images on the host.
- Call `gabor_wavefront(image)` or a plan function once to move each frame.
- Keep thresholding and `dilate` on the selected backend.
- Expect per-slice host transfer for connected components and per-box transfer
  for focus and diameter metrics.
- Call `to_host` once at an output boundary.

For GPU timing, call `synchronize_backend()` after the measured operation;
otherwise asynchronous work may be charged to the next statement.

The official [CUDA.jl memory guide](https://cuda.juliagpu.org/stable/usage/memory/)
explains its allocator and reclamation controls.
The [Metal.jl array API](https://metal.juliagpu.org/stable/api/array/) documents
its GPU array element-type conversions.

## FFT-native propagation order

v1 propagation kernels and low-pass filters are generated in the order consumed
by FFT plans.
Phase retrieval and reconstruction do not call `fftshift` or `ifftshift` in
each invocation.
Do not construct `PropagationKernel` manually from a centred v0.2 array;
generate it with `propagation_kernel` or convert its order first.

## CPU expectations

CPU exists for portability and reference accuracy, not to promise interactive
performance at the largest volume.
Start with fewer slices or a crop.
The same processing script remains valid when the initial backend selection is
changed to Metal or CUDA.
