# Performance and memory

## Reuse plans first

For a time series, create transfer arrays and both FFT plans outside the frame
loop. Reusing `PhaseRetrievalPlan` and `ReconstructionPlan` removes repeated FFT
planning and major work-buffer allocation. This is the principal v1 hot-path
optimisation.

## Estimate volume memory

A dense volume uses approximately

```text
height × width × slices × sizeof(output element)
```

A 1024×1024×1000 `Float32` volume alone is about 3.91 GiB. A reconstruction
also needs complex FFT buffers and transfer arrays. `N0f8` is four times smaller
but clips to 0–1 and quantises to 256 levels. Prefer `Float32` for measurement;
use `xyprojection` when the full volume is not needed.

## Avoid hidden transfers

- Load and correct camera images on the host.
- Call `gabor_wavefront(b, image)` or a plan function once to move each frame.
- Keep thresholding and `dilate` on the device.
- Expect per-slice host transfer for connected components and per-box transfer
  for focus/diameter metrics.
- Call `to_host` once at an output boundary.

For GPU timing, call `synchronize_backend(b)` after the measured operation;
otherwise asynchronous work may be charged to the next statement.

## FFT-native transfer order

v1 transfer and low-pass arrays are generated in the order consumed by FFT
plans. Phase retrieval and reconstruction no longer call `fftshift` or
`ifftshift` in each invocation. Do not construct `Transfer` manually from a
centred v0.2 array; generate it with `transfer` or convert its order first.

## CPU expectations

CPU exists for portability and reference accuracy, not to promise interactive
performance at the largest volume. Start with fewer slices or a crop. The same
script remains valid when a Metal/CUDA backend is introduced later.
