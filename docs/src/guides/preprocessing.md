# Preprocessing and files

## Read an image

`load_gray2float(path)` uses FileIO/ImageIO and returns a two-dimensional
`Array{Float32}`. Verify its shape and range before sending it to a backend.

```julia
image = load_gray2float("hologram.png")
@assert ndims(image) == 2
@assert all(isfinite, image)
```

## Background estimate

```julia
paths = sort(readdir("holograms"; join=true))
background = make_background(paths; mode=:mode) # or :mean
image = clamp.(load_gray2float(first(paths)) .- background .+ mean(background),
               0, 1)
```

The mode implementation no longer allocates a `256 × height × width` vote
volume. It uses bounded host memory and parallelises image rows. Background
estimation is preprocessing and intentionally uses the host on every backend.

## Save particle dictionaries

`dictsave` writes UUID keys and Float32 coordinate vectors to JSON;
`dictload` restores that schema. Keep optical parameters and coordinate units in
a separate versioned config file next to the JSON sequence. The package does
not infer missing metadata from a filename.

For a complete directory convention and YAML configuration, see
[phdemo real-data tutorial](@ref).
