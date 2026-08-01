# Handoff

## Resume From Here
初回 release candidate は ParticleHolography.jl `b050823` と phdemo `1b18b56` で commit 済みである。

文書確認後の追加変更は `feat!: add default backend and unified reconstruction outputs` として commit 済みである。

phdemo の対応は `948e1eb` として commit 済みである。

CPU package test 180/180、CUDA shared/physical contract 19/19、CUDA integration/legacy 7/7、Plots 5/5、phdemo 14/14、docs build、CPU coverage 90.1%、v0.2.4 数値回帰が合格した。

次は保守者の許可後に branch を push して GitHub Actions と macOS Metal を確認する。

## Next Actions
- 両リポジトリの差分を保守者が確認する。
- 明示的な許可後に `codex/v1.0.0` を push する。
- ParticleHolography.jl の Julia 1.10、hosted CPU 行列、macOS Metal、self-hosted CUDA、docs、Plots workflow を確認する。
- ParticleHolography.jl v1 を先に公開し、その revision に対する phdemo CI を確認する。
- 許可後に tag、GitHub Release、General registry、stable docs の作業を行う。

## Watch Outs
- 破壊的変更は許容されるが、新規アルゴリズム追加は不要。
- session 記録以外に開始時点のユーザー変更はない。
- push、PR、タグ、GitHub Release はまだ許可されていない。
- Julia 1.10 と macOS Metal はローカル未確認であり、workflow green を推測してはならない。
- phdemo の通常 CI は `Project.toml` の compat から登録済み v1 を解決するため、ParticleHolography.jl を先に登録する。
- 未公開 ParticleHolography revision に対する phdemo CI は workflow dispatch の任意 revision override を使う。
- test/environmentsのManifestはignore対象で、CIは毎回`Pkg.develop`/`Pkg.instantiate`する。
- self-hosted runner は GPU 専用 label がないため、CUDA.functional() を job 冒頭で必須確認する。
- existing test は tracked fixture を書き換えるため、v1 test は `mktempdir` を使う。
