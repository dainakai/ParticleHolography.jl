# 10-minute CPU quickstart

This example is intentionally small. It teaches the objects and array shapes
without requiring a camera image, GPU, or output files. The synthetic image is
only an API demonstration, not a physical hologram generator.

## 1. Select a backend and make an image

An inline hologram is a 2-D intensity image. ParticleHolography represents it as
a real-valued array, normally scaled near the interval 0–1.

```@example quickstart
using ParticleHolography

b = backend(:cpu)
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

sqrt_part = transfer_sqrt(b, n, wavelength, pixel_pitch)
front = transfer(b, -front_distance, wavelength, sqrt_part)
step = transfer(b, -slice_spacing, wavelength, sqrt_part)
nothing
```

The negative distances back-propagate from the camera toward the object. If
your coordinate convention is reversed, reverse the signs consistently.

## 3. Reconstruct

Gabor reconstruction assumes zero phase at the camera and uses the square root
of measured intensity as the wavefront amplitude.

```@example quickstart
wavefront = gabor_wavefront(b, hologram)
plan = ReconstructionPlan(b, front, step)
volume, xy_projection = reconstruct_and_projection(plan, wavefront, slices)
(size(volume), size(xy_projection), eltype(volume), extrema(volume))
```

`volume[:, :, z]` is reconstructed intensity at one depth. `xy_projection` is
the minimum intensity over depth, useful because opaque particles reconstruct
as dark regions. The default v1 output is `Float32`; this avoids the clipping
and quantisation that occurred when v0.2 returned `N0f8` by default.

## 4. Reuse the plan for a sequence

FFT plans and the largest work arrays are expensive to create. Build a
`ReconstructionPlan` once for a fixed image shape and geometry, then reuse it.

```julia
plan = ReconstructionPlan(b, front, step)
for hologram in frames
    wavefront = gabor_wavefront(b, hologram)
    volume = reconstruct(plan, wavefront, slices)
    # threshold, detect, and save this frame
end
```

The allocating functions are convenient for notebooks. For tighter control,
preallocate arrays and use `reconstruct!`, `xyprojection!`, or
`reconstruct_and_projection!`.

## Next

- Use real camera parameters: [Parameters and units](@ref).
- Move this exact workflow to a GPU: [Choose CPU, Metal, or CUDA](@ref).
- Use two cameras to retrieve phase: [Gabor and phase retrieval](@ref).
- Threshold and detect particles: [Detection and tracking](@ref).
