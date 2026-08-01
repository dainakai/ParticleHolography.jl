# Parameters and units

Most incorrect reconstructions come from geometry, sign, or unit mistakes.
Record these values with the experiment rather than embedding them in a script.

| Parameter | Meaning | Common unit | Effect of an error |
|---|---|---|---|
| `wavelength` | illumination wavelength | μm | changes every propagation phase |
| `pixel_pitch` | physical camera pixel spacing | μm/pixel | changes field size and spatial frequencies |
| `front_distance` | camera to near edge of reconstructed volume | μm | shifts the whole depth range |
| `slice_spacing` | distance between reconstructed planes | μm | changes depth sampling |
| `slices` | number of reconstructed planes | count | changes range and memory |
| `phase_distance` | separation of the two hologram planes | μm | controls phase-retrieval propagation |
| `iterations` | Gerchberg–Saxton iterations | count | trades runtime for convergence |
| `threshold` | intensity classified as particle | normalized intensity | changes detected extent/count |

## Use one length unit

The transfer equation contains ratios, so the API cannot detect a mix of metres,
millimetres, and micrometres. This is valid:

```julia
wavelength = 0.6328       # μm
pixel_pitch = 10.0        # μm
front_distance = 80_000.0 # μm
```

This is not valid even though it runs: wavelength in metres with the other two
values in micrometres.

## Axis order and coordinates

Julia arrays use `(row, column, slice)`, corresponding to `(y, x, z)`. Particle
coordinate vectors use `[x, y, z]`. Bounding boxes use
`[xmin, ymin, zmin, xmax, ymax, zmax]`, with inclusive one-based indices.

`transfer(b, distance, ...)` uses positive distance in the mathematical
propagation direction. Camera-to-object reconstruction normally uses negative
distances. Particle plotting defaults to a negative z scale to match the
historical optical-axis display convention; set `scaling` explicitly for your
laboratory frame.

## Choose a depth range

The reconstructed planes are approximately

```text
front_distance,
front_distance + slice_spacing,
…,
front_distance + (slices - 1) * slice_spacing
```

Start with a coarse, shallow CPU reconstruction and verify that focus crosses
the expected range. Increase slice count only after the sign and endpoints are
correct.

## Threshold and focus

Opaque particles appear dark in the current workflow, so binarisation is
normally `volume .<= threshold`. Inspect several frames and keep a consistent
threshold when illumination is stable. `particle_coordinates` selects depth
using the Tamura focus profile by default; a low-pass-filtered volume may be
passed to `particle_coor_diams` for focus while diameter is measured from the
unfiltered volume.
