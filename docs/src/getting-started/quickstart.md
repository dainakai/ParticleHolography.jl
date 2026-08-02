# 10-minute CPU quickstart

This example is intentionally small. It teaches the objects and array shapes
without requiring a camera image, GPU, or output files. The synthetic image is
only an API demonstration, not a physical hologram generator.

## 1. Select a backend and make an image

An inline hologram is a 2-D intensity image. ParticleHolography represents it as
a real-valued array, normally scaled near the interval 0–1.

```@example quickstart
using ParticleHolography

backend(:cpu)
n = 32
rows = reshape(Float32.(1:n), n, 1)
cols = reshape(Float32.(1:n), 1, n)
hologram = @. 0.8f0 - 0.15f0 * exp(-((rows - 16.5f0)^2 +
                                     (cols - 16.5f0)^2) / 24f0)
size(hologram)
```

## 2. State the optical geometry

All lengths below use micrometres. Any consistent unit works, but mixing metres
and micrometres produces a numerically valid and scientifically wrong result.

```@example quickstart
wavelength = 0.6328      # μm
pixel_pitch = 10.0       # μm / pixel
front_distance = 800.0   # μm, camera to first reconstructed plane
slice_spacing = 20.0     # μm
slices = 8

grid = propagation_grid(n, wavelength, pixel_pitch)
front = propagation_kernel(-front_distance, wavelength, grid)
step = propagation_kernel(-slice_spacing, wavelength, grid)
nothing
```

The negative distances back-propagate from the camera toward the object. If
your coordinate convention is reversed, reverse the signs consistently.

## 3. Reconstruct

Gabor reconstruction assumes zero phase at the camera and uses the square root
of measured intensity as the wavefront amplitude.

```@example quickstart
wavefront = gabor_wavefront(hologram)
request = ReconstructionRequest(slices;
    volume=Float32,
    min_projection=N0f8,
)
plan = ReconstructionPlan(front, step; request)
diagnostic = memory_diagnostic(plan, request)
result = reconstruct(plan, wavefront)
(size(result.volume), size(result.min_projection),
 eltype(result.volume), eltype(result.min_projection), diagnostic.safe)
```

`result.volume[:, :, z]` is reconstructed intensity at one depth.
`result.min_projection` is the minimum intensity over depth, useful because
opaque particles reconstruct as dark regions.
The `Float32` volume preserves measurement values, while the compact `N0f8`
projection is convenient for display and storage.
Both outputs came from one propagation loop.

## 4. Reuse the plan for a sequence

FFT plans and the largest work arrays are expensive to create. Build a
`ReconstructionPlan` once for a fixed image shape and geometry, then reuse it.

```julia
request = ReconstructionRequest(slices; volume=Float32, min_projection=N0f8)
plan = ReconstructionPlan(front, step; request)
for hologram in frames
    wavefront = gabor_wavefront(hologram)
    result = reconstruct(plan, wavefront)
    # threshold, detect, and save this frame
end
```

The allocating functions are convenient for notebooks. For tighter control,
preallocate arrays and use `reconstruct!`, `xyprojection!`, or
`reconstruct_and_projection!`.

Use `reconstruct_padded(plan, wavefront; mode=:mean)` when the plan was built
for a larger working plane and the border should use the input wavefront mean
rather than zero.

## Next

- Use real camera parameters: [Parameters and units](@ref).
- Move this exact workflow to a GPU: [Choose CPU, Metal, or CUDA](@ref).
- Use two cameras to retrieve phase: [Gabor and phase retrieval](@ref).
- Threshold and detect particles: [Detection and tracking](@ref).
