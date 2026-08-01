# Gabor and phase retrieval

## Gabor reconstruction from one image

```julia
using ParticleHolography

b = backend(:cpu) # change only backend setup for Metal or CUDA
image = load_gray2float("hologram.png")
n = size(image, 1)

wavelength = 0.6328
pixel_pitch = 10.0
front_distance = 80_000.0
slice_spacing = 100.0
slices = 1_000

sqrt_part = transfer_sqrt(b, n, wavelength, pixel_pitch)
front = transfer(b, -front_distance, wavelength, sqrt_part)
step = transfer(b, -slice_spacing, wavelength, sqrt_part)
wavefront = gabor_wavefront(b, image)
plan = ReconstructionPlan(b, front, step)
volume, projection = reconstruct_and_projection(plan, wavefront, slices)

projection_host = to_host(projection)
```

The convenience call allocates `volume`. If only the projection is required,
use `xyprojection(plan, wavefront, slices)` to avoid retaining the full stack.

## Phase retrieval from two images

The image closer to the object is `image1`; `image2` is separated from it by
`phase_distance` in the positive propagation direction.

```julia
image2_corrected = quadratic_distortion_correction(image2, coefficients)
forward = transfer(b, phase_distance, wavelength, sqrt_part)
backward = transfer(b, -phase_distance, wavelength, sqrt_part)
phase_plan = PhaseRetrievalPlan(b, forward, backward)
retrieved = phase_retrieval(phase_plan, image1, image2_corrected; iterations=10)

reconstruction_plan = ReconstructionPlan(b, front, step)
volume = reconstruct(reconstruction_plan, retrieved, slices)
```

`phase_retrieval!` returns a `Wavefront` that aliases its plan workspace. Use it
inside a frame loop when the next call may overwrite the prior result. The
non-bang function returns an owning copy.

## Calibration

Use a feature-rich calibration target and reconstructed images from the two
cameras. Calibration uses the same brute-force normalized cross-correlation PIV
method as v0.2.

```julia
coefficients = get_distortion_coefficients(calibration1, calibration2;
    backend=b,
    grid_size=128,
    interrogation_size=128,
    search_size=256,
)
```

CPU and Metal run calibration PIV on the CPU. CUDA uses its device kernel.
`verbose=true` requires `using Plots` and writes before/after diagnostics to
`save_dir`.

## Low-pass filtering

Filters are stored in FFT-native order, so hot loops do not shift Fourier
arrays.

```julia
filter = super_gaussian_filter(b, front_distance + slices * slice_spacing,
                               wavelength, n, pixel_pitch)
filtered_wavefront = apply_low_pass_filter(retrieved, filter)
filtered_volume = reconstruct(reconstruction_plan, filtered_wavefront, slices)
```

Filtering may stabilise focus metrics but changes intensity and apparent size.
Keep unfiltered and filtered roles explicit in the analysis record.
