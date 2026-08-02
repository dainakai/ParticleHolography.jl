# Handoff

## Resume From Here
初回 release candidate は ParticleHolography.jl `b050823` と phdemo `1b18b56` で commit 済みである。

文書確認後の追加変更は `feat!: add default backend and unified reconstruction outputs` として commit 済みである。

phdemo の対応は `948e1eb` として commit 済みである。

CPU package test 197/197、CUDA shared/physical contract 19/19、CUDA integration/legacy 7/7、Metal shared contract 18/18、Plots 7/7、phdemo 14/14、docs build、統合 coverage project 82.56% / patch 82.2%、v0.2.4 数値回帰が合格した。

ParticleHolography.jl Draft PR #90 は作成済みで、GitHub Actions と macOS Metal を確認済みである。

## Next Actions
- ParticleHolography.jl Draft PR #90 の差分を保守者が確認する。
- phdemo の `codex/v1.0.0` branch を公開し、別PRで本体PRと相互参照する。
- ParticleHolography.jl v1 を先に公開し、その revision に対する phdemo CI を確認する。
- 許可後に tag、GitHub Release、General registry、stable docs の作業を行う。

## Watch Outs
- 破壊的変更は許容されるが、新規アルゴリズム追加は不要。
- session 記録以外に開始時点のユーザー変更はない。
- tag、GitHub Release、merge はまだ許可されていない。
- Metalはローカル未確認だが、GitHub-hosted Apple silicon上でMetal.jl 1.10.0と共有契約18/18が合格した。
- phdemo の通常 CI は `Project.toml` の compat から登録済み v1 を解決するため、ParticleHolography.jl を先に登録する。
- 未公開 ParticleHolography revision に対する phdemo CI は workflow dispatch の任意 revision override を使う。
- test/environmentsのManifestはignore対象で、CIは毎回`Pkg.develop`/`Pkg.instantiate`する。
- self-hosted runner は GPU 専用 label がないため、CUDA.functional() を job 冒頭で必須確認する。
- existing test は tracked fixture を書き換えるため、v1 test は `mktempdir` を使う。
