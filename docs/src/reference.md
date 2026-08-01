# API reference

## Backends and transfers

```@docs
AbstractBackend
CPUBackend
backend
available_backends
to_backend
to_host
synchronize_backend
TransferSqrtPart
Transfer
Wavefront
LowPassFilter
transfer_sqrt
transfer
gabor_wavefront
```

## Plans and reconstruction

```@docs
PhaseRetrievalPlan
phase_retrieval!
phase_retrieval
ReconstructionPlan
reconstruct!
reconstruct
reconstruct_complex!
reconstruct_complex
xyprojection!
xyprojection
reconstruct_and_projection!
reconstruct_and_projection
asm_propagate!
asm_propagate
```

## Filters, detection, calibration, and tracking

```@autodocs
Modules = [ParticleHolography]
Order = [:function]
```

## Index

```@index
```
