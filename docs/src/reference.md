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
PropagationGrid
PropagationKernel
Wavefront
LowPassFilter
propagation_grid
propagation_kernel
gabor_wavefront
pad2d
```

## Plans and reconstruction

```@docs
PhaseRetrievalPlan
phase_retrieval!
phase_retrieval
ReconstructionRequest
ReconstructionResult
MemoryDiagnostic
memory_diagnostic
ReconstructionPlan
reconstruct!
reconstruct
reconstruct_padded
reconstruct_complex!
reconstruct_complex
xyprojection!
xyprojection
reconstruct_and_projection!
reconstruct_and_projection
asm_propagate!
asm_propagate
```

## Deprecated migration aliases

`TransferSqrtPart`, `Transfer`, `transfer_sqrt`, and `transfer` are retained as
deprecated aliases or wrappers for source migration.
New code should use `PropagationGrid`, `PropagationKernel`,
`propagation_grid`, and `propagation_kernel`.

## Filters, detection, calibration, and tracking

```@autodocs
Modules = [ParticleHolography]
Order = [:function]
```

## Index

```@index
```
