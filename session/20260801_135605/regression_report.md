# v0.2.4 / v1.0.0 numerical regression

- Date: 2026-08-01
- Baseline: ParticleHolography.jl v0.2.4, commit `c23ea07`
- Candidate: local `codex/v1.0.0`
- Julia: 1.12.6
- GPU: NVIDIA GeForce RTX 4080 SUPER, CUDA.jl 6.2.1
- Input: deterministic 32 x 32 hologram pair
- Parameters: wavelength 0.6328, pixel pitch 10.0, phase distance 120.0,
  front distance -800.0, slice spacing -20.0, 3 phase iterations, 5 slices

The v0.2.4 implementation was run from a detached temporary worktree. Its
arrays were serialized, then compared against both v1 CPU and v1 CUDA. The v1
transfer arrays were shifted only for direct comparison with v0.2.4's old
centred storage order; v1 computation uses FFT-native order without a hot-loop
shift.

| Product | v0.2.4 vs v1 CPU max abs | v0.2.4 vs v1 CUDA max abs | v1 CPU vs CUDA max abs |
| --- | ---: | ---: | ---: |
| transfer square-root term | 0 | 0 | 0 |
| forward transfer | 3.5528e-5 | 3.5528e-5 | 0 |
| phase-retrieved wavefront | 8.5743e-6 | 8.5171e-6 | 6.3702e-7 |
| intensity volume | 5.9605e-6 | 6.0201e-6 | 8.9407e-7 |
| minimum-intensity projection | 5.9605e-6 | 6.0201e-6 | 8.9407e-7 |
| complex-amplitude volume | 1.4091e-5 | 1.4365e-5 | 6.8885e-7 |

The transfer difference is expected from v0.2.4 evaluating the phase in a
Float32 GPU kernel while v1 evaluates it on the host in Float64 before storing
ComplexF32. The downstream products remain within 1.5e-5 maximum absolute
error, and the v1 CPU/CUDA contract is within 1e-6 for this case. No algorithmic
change was detected.

Reproduction driver: `session/20260801_135605/regression_v1.jl`. It expects the
temporary v0.2.4 serialization at `/tmp/particleholo_v024_regression.bin`.
