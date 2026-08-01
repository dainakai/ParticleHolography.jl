# Handoff

## Resume From Here
ローカル release candidate は完成した。CPU package test 99/99、CUDA shared contract 10/10、CUDA integration/legacy 7/7、Plots 5/5、phdemo 9/9、docs build、v0.2.4 数値回帰、workflow/TOML parse、ambiguity 0 件、両リポジトリの `git diff --check` が合格した。次は保守者の許可後に branch を push し、GitHub Actions と macOS Metal を確認する。

## Next Actions
- 両リポジトリの差分を保守者が確認する。
- 明示的な許可後に `codex/v1.0.0` を commit して push する。
- ParticleHolography.jl の Julia 1.10、hosted CPU 行列、macOS Metal、self-hosted CUDA、docs、Plots workflow を確認する。
- ParticleHolography.jl v1 を先に公開し、その revision に対する phdemo CI を確認する。
- 許可後に tag、GitHub Release、General registry、stable docs の作業を行う。

## Watch Outs
- 破壊的変更は許容されるが、新規アルゴリズム追加は不要。
- session 記録以外に開始時点のユーザー変更はない。
- push、PR、タグ、GitHub Release はまだ許可されていない。
- Julia 1.10 と macOS Metal はローカル未確認であり、workflow green を推測してはならない。
- phdemo remote CI は ParticleHolography.jl の公開 revision を必要とするため、ParticleHolography.jl を先に merge または tag する。
- test/environmentsのManifestはignore対象で、CIは毎回`Pkg.develop`/`Pkg.instantiate`する。
- self-hosted runner は GPU 専用 label がないため、CUDA.functional() を job 冒頭で必須確認する。
- existing test は tracked fixture を書き換えるため、v1 test は `mktempdir` を使う。
