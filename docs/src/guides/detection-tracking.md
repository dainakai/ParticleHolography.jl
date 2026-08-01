# Detection and tracking

Assume `volume` is a reconstructed `Float32` array on any backend.

## Detect one frame

```julia
threshold = 30 / 255
binary = volume .<= threshold
binary = dilate(dilate(binary))

boxes = particle_bounding_boxes(binary)
particles = particle_coor_diams(boxes, volume)
dictsave("particles.json", particles)
```

`dilate` remains on CPU, Metal, or CUDA. Connected components stage one slice
at a time to the host, and coordinate metrics stage one bounding subvolume at a
time. This bounds memory and keeps the public workflow identical.

Choose `particle_bounding_boxes_3d` if a component must touch the immediately
previous z slice. Use the default non-strict function for the historical
holographic ghost-suppression behaviour.

## Output schema

```text
Dict(
  UUID(...) => Float32[x, y, z],
  # or Float32[x, y, z, equivalent_diameter]
)
```

Coordinates are one-based pixel/slice coordinates. Multiply x and y by pixel
pitch and z by slice spacing, then add your reconstructed-volume origin to
obtain physical coordinates.

## Link successive frames

```julia
frames = dictload.(["000001.json", "000002.json", "000003.json"])
graphs = [labonte(a, b) for (a, b) in zip(frames[1:end-1], frames[2:end])]
paths = enum_edge(first(graphs))
for graph in Iterators.drop(graphs, 1)
    append_path!(paths, graph)
end
full = gen_fulldict(frames)
```

Tune `max_distance` in `labonte` to the maximum plausible frame-to-frame
motion in coordinate units. `dim3weight` scales the less precise depth axis.
The function does not mutate the input dictionaries and rejects UUID reuse
across frames.

## Plotting is optional

```julia
using Plots # activates ParticleHolographyPlotsExt
particleplot(frames[1]; scaling=(10.0, 10.0, -100.0))
trajectoryplot(paths, full; scaling=(10.0, 10.0, -100.0))
```

The core package does not load a display stack. On a compute server, omit
Plots.jl and save particle dictionaries for later visualisation.
