# phdemo real-data tutorial

[`dainakai/phdemo`](https://github.com/dainakai/phdemo) is the official
companion workflow for the supplied water-droplet experiment. It is separate
because the raw images and analysis-specific postprocessing do not belong in a
general Julia package.

## Relationship to this package

- ParticleHolography.jl owns propagation, phase retrieval, reconstruction,
  dilation, detection, calibration, JSON IO, correspondence, and plotting APIs.
- phdemo owns the experiment configuration, camera/scene directory mapping,
  raw data, output naming, and droplet-specific trajectory postprocessing.
- phdemo selects one backend once. Its processing functions receive that object
  and call only backend-neutral v1 API names.

## Recommended order

```bash
git clone https://github.com/dainakai/ParticleHolography.jl.git
git clone https://github.com/dainakai/phdemo.git
cd phdemo
julia --project=. -e 'using Pkg; Pkg.develop(path="../ParticleHolography.jl"); Pkg.instantiate()'
julia --project=. bin/phdemo.jl doctor --config configs/sample.yaml
julia --project=. bin/phdemo.jl smoke --config configs/sample.yaml --backend cpu
```

The local `Pkg.develop` step is needed while testing an unreleased v1 checkout.
After ParticleHolography.jl v1 is registered, ordinary `Pkg.instantiate()` uses
the compatible release declared by phdemo.

After `doctor` validates paths, paired frame counts, image shape, backend, and
calibration availability, run the stages required by your data:

```bash
julia --project=. bin/phdemo.jl background --config configs/sample.yaml
julia --project=. bin/phdemo.jl calibrate  --config configs/sample.yaml --backend cuda
julia --project=. bin/phdemo.jl process    --config configs/sample.yaml --backend cuda
julia --project=. bin/phdemo.jl track      --config configs/sample.yaml
```

The `smoke` command uses a small crop, a few slices, and one frame pair. It is a
pipeline check, not a scientific result. Run it before committing hours of CPU
work or allocating a full GPU volume.

See the phdemo README for its current data layout and exact config keys. Use a
phdemo release compatible with ParticleHolography.jl v1; do not mix its old
CUDA-only `include("src/proc.jl")` workflow with this API.
