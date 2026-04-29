# ParticleHolography

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://dainakai.github.io/ParticleHolography.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://dainakai.github.io/ParticleHolography.jl/dev/)
[![Build Status](https://github.com/dainakai/ParticleHolography.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/dainakai/ParticleHolography.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/dainakai/ParticleHolography.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/dainakai/ParticleHolography.jl)

Documentation for [ParticleHolography](https://github.com/dainakai/ParticleHolography.jl).

A package for particle measurement using inline holography. Please refer to the [documentation](https://dainakai.github.io/ParticleHolography.jl/stable/) for more information.

## Installation

```julia
using Pkg
Pkg.add("ParticleHolography")
```

## CPU-only and CUDA use

The core package is importable without CUDA:

```julia
using ParticleHolography
```

CPU-side utilities such as image loading, contour helpers, particle dictionary IO, and tracking helpers are available from the core package. GPU APIs whose names start with `cu_` require CUDA.jl and a functional CUDA runtime:

```julia
using CUDA
using ParticleHolography
```

If CUDA is not loaded or `CUDA.functional()` is false, GPU APIs throw an `ArgumentError` explaining the missing CUDA requirement instead of failing during package import.
