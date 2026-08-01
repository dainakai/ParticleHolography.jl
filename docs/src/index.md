```@meta
CurrentModule = ParticleHolography
```

# ParticleHolography.jl

ParticleHolography.jl turns one or two inline hologram images into a numerical
3-D light-intensity volume, detected particle positions, and particle tracks.
Version 1 uses the same processing code on a CPU, Apple Metal GPU, or NVIDIA
CUDA GPU; the selected backend controls where arrays and FFTs run.

You do not need to understand holography before starting. Read the pages in
this order:

| Your goal | Start with |
|---|---|
| See a complete reconstruction without GPU setup | [10-minute CPU quickstart](@ref) |
| Choose hardware or fix backend setup | [Choose CPU, Metal, or CUDA](@ref) |
| Understand wavelength, distance, slices, and signs | [Parameters and units](@ref) |
| Understand what each processing stage means | [From hologram to particles](@ref) |
| Process the supplied experimental data | [phdemo real-data tutorial](@ref) |
| Update old `cu_*` code | [Migrate from v0.2](@ref) |

## What the package does

```text
camera image(s)
    → optional background and camera correction
    → complex wavefront (Gabor or phase retrieval)
    → 3-D reconstruction / minimum-intensity projection
    → threshold and connected components
    → particle coordinates and diameters
    → frame-to-frame correspondences and trajectories
```

The package implements these existing workflows; v1 does not choose scientific
parameters automatically. Keep a record of camera pixel pitch, wavelength,
camera spacing, reconstruction range, threshold, and coordinate units.

## Install the CPU package

```julia
using Pkg
Pkg.add("ParticleHolography")
```

CPU is always available and is the reference implementation. GPU packages are
optional, so a machine without a GPU driver can still install, import, test,
and use ParticleHolography.jl. See [Choose CPU, Metal, or CUDA](@ref) before
installing Metal.jl or CUDA.jl.

## Backends at a glance

| Backend | Extra package | Typical use |
|---|---|---|
| `backend(:cpu)` | none | learning, CI, reference results, any machine |
| `backend(:metal)` | Metal.jl | Apple-silicon Mac, macOS 14+ |
| `backend(:cuda)` | CUDA.jl | NVIDIA GPU workstation or server |

Reconstruction and portable dilation stay on the selected device. Connected
component labeling and particle metrics use a documented host stage so they
remain available on every backend. CUDA has an accelerated calibration PIV
path; CPU and Metal use the same CPU reference for calibration.

## Real data

[`dainakai/phdemo`](https://github.com/dainakai/phdemo) is the companion
repository. It contains experimental droplet holograms, a configuration file,
and commands for background removal, calibration, reconstruction, detection,
and tracking. First complete the synthetic CPU quickstart here, then follow
[phdemo real-data tutorial](@ref).
