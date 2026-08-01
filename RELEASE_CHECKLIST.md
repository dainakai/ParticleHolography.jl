# v1.0.0 release checklist

## Package and API

- [x] `Project.toml` version is 1.0.0 and Julia compat starts at 1.10.
- [x] CPU import does not require CUDA, Metal, Makie, or Plots.
- [x] CPU/CUDA/Metal use one backend-neutral optical API.
- [x] v0.2 compatibility wrappers and migration table exist.
- [x] FFT plans and major work buffers are reusable.
- [x] tests write temporary files only under `mktempdir`.

## Local verification

- [x] CPU core: 99/99 tests pass on Julia 1.12.6.
- [x] CUDA shared contract: 10/10 tests pass on RTX 4080 SUPER.
- [x] CUDA integration/legacy: 7/7 tests pass on RTX 4080 SUPER.
- [ ] CPU suite passes on the minimum Julia 1.10 runtime.
- [x] docs build completes without unresolved references.
- [x] plotting extension test passes headlessly, including diagnostic files (5/5).
- [x] representative v0.2/v1 reconstruction regression is recorded.

## GitHub Actions

- [ ] Linux/Windows/macOS CPU matrix is green.
- [ ] `macos-15` arm64 Metal shared contract is green and not skipped.
- [ ] self-hosted Linux/X64 CUDA job reports `CUDA.functional() == true` and is green.
- [ ] documentation and optional plotting jobs are green.
- [x] fork PRs cannot execute code on the persistent self-hosted CUDA runner.

## Release operations (require explicit maintainer approval)

- [ ] Review final diff and unresolved limitations.
- [ ] Commit and push the release branch.
- [ ] Open/merge the v1 pull request.
- [ ] Confirm phdemo points to the released v1 compatibility range.
- [ ] Create signed/annotated `v1.0.0` tag.
- [ ] Publish GitHub Release with migration and backend notes.
- [ ] Verify stable docs and General registry compatibility.
