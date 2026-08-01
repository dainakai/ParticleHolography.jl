# Gabor and phase retrieval

## Gabor reconstruction from one image

Select the backend once, describe the outputs once, and reuse the plan for every
frame with the same geometry.

```julia
using ParticleHolography

backend(:cpu) # change only this setup for Metal or CUDA
image = load_gray2float("hologram.png")
n = size(image, 1)

wavelength = 0.6328
pixel_pitch = 10.0
front_distance = 80_000.0
slice_spacing = 100.0
slices = 1_000

grid = propagation_grid(n, wavelength, pixel_pitch)
front = propagation_kernel(-front_distance, wavelength, grid)
step = propagation_kernel(-slice_spacing, wavelength, grid)
wavefront = gabor_wavefront(image)

request = ReconstructionRequest(slices;
    volume=Float32,
    min_projection=N0f8,
)
plan = ReconstructionPlan(front, step; request)
@show memory_diagnostic(plan, request)

result = reconstruct(plan, wavefront)
volume = result.volume
projection_host = to_host(result.min_projection)
```

The volume and MinIP are filled during one traversal of the reconstruction
depths.
Requesting both does not repeat propagation.

## Choose only the outputs you need

`ReconstructionRequest` separates what is calculated from how each result is
stored.

| Requested result | Allowed storage | Meaning |
|---|---|---|
| intensity volume | `Float32` | unclipped squared wavefront magnitude for measurement |
| intensity volume | `N0f8` | clipped 0–1, 8-bit normalized storage |
| wavefront volume | `ComplexF32`, `ComplexF64` | complex propagated wavefront |
| MinIP | `Float32`, `N0f8` | minimum intensity across depth |
| omitted output | `nothing` | no allocation for that output |

Every valid volume/MinIP combination uses the same engine.
For example, a `Float32` volume with an `N0f8` MinIP is:

```julia
request = ReconstructionRequest(slices;
    volume=Float32,
    min_projection=N0f8,
)
result = reconstruct(plan, wavefront, request)
```

For MinIP alone, set `volume=nothing`.
For a complex wavefront volume alone, set `volume=ComplexF64` and
`min_projection=nothing`; intensity is then not calculated.
Propagation workspaces currently use `ComplexF32` on every backend.
A requested `ComplexF64` volume preserves the unsquared complex wavefront in
`ComplexF64` storage, but does not change propagation arithmetic to Float64.
Metal.jl does not provide native `ComplexF64` GPU arithmetic, so Metal converts
each slice and stores this particular output on the host.

## Mean-value padding

Padding reduces wraparound from FFT-periodic boundaries.
Mean padding usually creates a less abrupt border than zero padding for an
illuminated hologram.

```julia
padded_shape = (2 * size(image, 1), 2 * size(image, 2))
padded_grid = propagation_grid(padded_shape, wavelength, pixel_pitch)
padded_front = propagation_kernel(-front_distance, wavelength, padded_grid)
padded_step = propagation_kernel(-slice_spacing, wavelength, padded_grid)
padded_plan = ReconstructionPlan(
    padded_front,
    padded_step;
    request,
    output_shape=size(image),
)

result = reconstruct_padded(padded_plan, wavefront; mode=:mean)
```

Propagation uses the larger plane, but only the original central field of view
is stored for each depth.
Use `mode=:zero` when zero padding is scientifically appropriate.

## Memory preflight

`PhaseRetrievalPlan` and `ReconstructionPlan` estimate their FFT workspaces and
requested output allocations before constructing large arrays.
CPU uses currently free host memory, CUDA uses currently free device memory,
and Metal uses currently free unified memory.

```julia
diagnostic = memory_diagnostic(
    backend(),
    (2048, 2048),
    ReconstructionRequest(1_000; volume=Float32, min_projection=N0f8),
)
@show diagnostic.required_bytes diagnostic.available_bytes diagnostic.safe
```

The default safety factor is 1.2.
An unsafe estimate raises an error before the large allocation.
Reduce the image, slice count, or outputs first.
If the conservative estimate is inappropriate for a controlled environment,
pass `check_memory=false` to the plan constructor or `reconstruct` call.

## Phase retrieval from two images

The image closer to the object is `image1`; `image2` is separated from it by
`phase_distance` in the positive propagation direction.

```julia
image2_corrected = quadratic_distortion_correction(image2, coefficients)
forward = propagation_kernel(phase_distance, wavelength, grid)
backward = propagation_kernel(-phase_distance, wavelength, grid)
phase_plan = PhaseRetrievalPlan(forward, backward)
retrieved = phase_retrieval(phase_plan, image1, image2_corrected; iterations=10)

reconstruction_plan = ReconstructionPlan(front, step; request)
result = reconstruct(reconstruction_plan, retrieved)
```

`phase_retrieval!` returns a `Wavefront` that aliases its plan workspace.
Use it inside a frame loop when the next call may overwrite the prior result.
The non-bang function returns an owning copy.

## Calibration

Use a feature-rich calibration target and reconstructed images from the two
cameras.
Calibration uses the same brute-force normalized cross-correlation PIV method
as v0.2.

```julia
coefficients = get_distortion_coefficients(calibration1, calibration2;
    grid_size=128,
    interrogation_size=128,
    search_size=256,
)
```

CPU and Metal run calibration PIV on the CPU.
CUDA uses its device kernel.
`verbose=true` requires `using Plots` and writes before/after diagnostics to
`save_dir`.

## Low-pass filtering

Filters are stored in FFT-native order, so hot loops do not shift Fourier
arrays.

```julia
filter = super_gaussian_filter(front_distance + slices * slice_spacing,
                               wavelength, n, pixel_pitch)
filtered_wavefront = apply_low_pass_filter(retrieved, filter)
filtered_result = reconstruct(reconstruction_plan, filtered_wavefront)
```

Filtering may stabilise focus metrics but changes intensity and apparent size.
Keep unfiltered and filtered roles explicit in the analysis record.
