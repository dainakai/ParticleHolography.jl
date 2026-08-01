# Changelog

This file records user-visible changes. The repository history remains the
source of truth for individual patches.

## 1.0.0 — release candidate

### Breaking changes

- Julia 1.10 or later is required.
- The primary API is backend-neutral; CUDA-specific `cu_*` names are deprecated
  compatibility wrappers.
- New reconstruction functions return unclipped `Float32` instead of `N0f8` by
  default.
- Transfer and low-pass-filter arrays are stored in FFT-native order.
- Combined XY projection is calculated from unquantised intensity.
- Bounding-box overlap is inclusive and particle UUIDs are no longer generated
  by repeatedly resetting a fixed random generator.

### Added

- Explicit `CPUBackend`, `backend(:metal)`, and `backend(:cuda)` selection.
- Metal.jl and CUDA.jl package extensions with a shared numerical contract.
- `PhaseRetrievalPlan` and `ReconstructionPlan` reusable FFT plans/workspaces.
- Allocating and in-place APIs for phase retrieval, reconstruction, complex
  reconstruction, and minimum-intensity projection.
- CPU reference paths for all workflows, bounded-memory background mode, and
  documented host fallbacks for CCL/particle metrics/Metal calibration.
- Separate CPU, Metal, CUDA, plotting, and documentation CI jobs.
- Beginner quickstart, backend/parameter guides, troubleshooting, performance,
  phdemo, and v0.2 migration documentation.

### Fixed

- Dictionary mutation during iteration in particle box cleanup.
- UUID collisions between components created in different slices.
- Lost branches in `append_path!`.
- CCL/bounding-box edge-contact cases and calibration peak boundary handling.
- Tests overwriting tracked PNG and JSON fixtures.

## 0.2.4 — 2026-06-09

- Last CUDA-only release before the v1 backend and package-structure rewrite.
