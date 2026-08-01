# ParticleHolography.jl v1.0.0 リリース候補報告書

- 作成日: 2026-08-01
- 対象ブランチ: `ParticleHolography.jl` と `phdemo` の `codex/v1.0.0`
- 基準版: ParticleHolography.jl v0.2.4 (`c23ea07`)

## 判定

CPU と NVIDIA CUDA のローカル検証、文書生成、任意描画機能、phdemo 実データ smoke、v0.2.4 との数値回帰まで合格したため、ローカルのリリース候補として引き渡せる状態である。

macOS Metal、Julia 1.10、GitHub-hosted の OS 行列、self-hosted CUDA の Actions 結果は、ブランチを push して workflow を実行するまで未確認である。

commit、push、pull request、tag、GitHub Release は、保守者の明示的な許可がないため実施していない。

## 実行バックエンド

利用者は同じ高水準 API に対し、`backend(:cpu)`、`backend(:cuda)`、`backend(:metal)`、`backend(:auto)` のいずれかを選択できる。

CPU は FFTW、CUDA は CUDA.jl、Metal は Metal.jl を使い、配列転送、FFT、同期、再構成を package extension で接続する。

CUDA.jl と Metal.jl が公開する名前との衝突を避けるため、accelerator の実装型は公開せず、利用者向けの選択 API を `backend(...)` に統一した。

同期 API は `synchronize_backend` とした。

CPU package の import は CUDA、Metal、Plots、display server を必要としない。

Metal 上の connected-component labeling、粒子指標計算、brute-force PIV は、既存の処理方式を保つため host 参照経路を使う。

## 構造と API

`Project.toml` を 1.0.0 とし、Julia の最低互換版を 1.10 に設定した。

CUDA、Metal、Plots を weak dependency と package extension に分離した。

光学処理を `AbstractArray` と `AbstractFFTs` に基づく実装へ変更し、CPU、CUDA、Metal で共通の API を使えるようにした。

`PhaseRetrievalPlan` と `ReconstructionPlan` を追加し、FFT plan と主要な作業配列を複数フレーム間で再利用できるようにした。

伝達関数を FFT-native order で生成し、反復中の不要な shift と中間確保を減らした。

既存の `cu_*` API は移行用 wrapper として残し、旧既定値と clamp 動作を維持した。

背景推定は処理方式を変えず、入力全体の巨大な中間配列を避ける bounded-memory の並列実装へ変更した。

追跡処理では UUID 衝突、辞書走査中の削除、軌跡分岐の欠落、境界条件の不具合を修正した。

## 文書と phdemo

文書は、ホログラフィの概要、CPU quickstart、バックエンド選択、パラメータ、再構成、粒子検出と追跡、前処理、性能、トラブル対応、v1 移行の順で読める構成に変更した。

単位、z 座標の向き、入力型、保存先、メモリ見積り、host fallback の範囲を明記した。

phdemo は hard-coded な CUDA script 群から `PhDemo` v1 application へ変更し、一つの YAML 設定と CLI から `doctor`、`smoke`、`background`、`calibrate`、`process`、`track` を実行できるようにした。

phdemo の CI は ParticleHolography.jl の remote revision を取得するため、ParticleHolography.jl v1 の merge または tag を先に行う必要がある。

## 検証結果

| 対象 | 環境 | 結果 |
| --- | --- | --- |
| CPU package test | Julia 1.12.6 | 99/99 合格 |
| CUDA shared contract | RTX 4080 SUPER、CUDA.jl 6.2.1、runtime 13.3 | 10/10 合格 |
| CUDA integration と legacy wrapper | 同上 | 7/7 合格 |
| Plots extension | headless GR | 5/5 合格 |
| phdemo | 100 組の 1024×1024 BMP と 64×64×4 smoke | 9/9 合格 |
| 文書 | doctest、cross-reference、render | 合格 |
| Julia source | 全ファイルの parse | 合格 |
| method ambiguity | `Test.detect_ambiguities` | 0 件 |
| workflow と package metadata | YAML と TOML の parse | 合格 |
| 差分衛生 | `git diff --check` | 合格 |

phdemo smoke の出力範囲は `(0.090627916f0, 0.33805847f0)` で、foreground voxel は 0 個だった。

この smoke は入出力と再構成 pipeline の接続確認であり、粒子検出精度の評価には用いていない。

v0.2.4 と v1 の再構成および投影の最大絶対差は約 `6.1e-6` だった。

v0.2.4 と v1 の複素体積の最大絶対差は `1.5e-5` 未満だった。

v1 の CPU と CUDA の差は各比較で `1e-6` 未満だった。

## CI の安全性

CPU、Metal、CUDA、文書、Plots の workflow を分離した。

Metal workflow は `macos-15` の arm64 runner を使い、Metal が functional でない場合は skip せず失敗する。

CUDA workflow は `nvidia-smi` と `CUDA.functional()` を必須確認し、shared contract と integration test を実行する。

永続的な self-hosted runner で外部 pull request のコードを実行しないよう、CUDA workflow から `pull_request` trigger を除外した。

## 未確認事項と制約

Julia 1.10 での最低版確認は、ローカルに該当 runtime がないため GitHub Actions 待ちである。

Metal 実機の数値契約は、macOS arm64 workflow の実行待ちである。

Linux、Windows、macOS の CPU 行列、文書、Plots、self-hosted CUDA の Actions は、push 後に結果を確認する必要がある。

Metal の host fallback は正しさを優先した経路であり、大量フレーム処理では転送時間が支配的になる場合がある。

1024×1024×1024 の `Float32` volume 本体だけで約 4 GiB を必要とし、FFT と作業 buffer を含む実際の必要量はさらに大きい。

## 推奨するリリース順序

1. 両リポジトリの最終差分を保守者が確認する。
2. 許可後に `codex/v1.0.0` を commit して push する。
3. ParticleHolography.jl の CPU、Metal、CUDA、文書、Plots workflow をすべて確認する。
4. Julia 1.10 と macOS Metal の失敗があれば、release branch 上で修正して再検証する。
5. ParticleHolography.jl v1 を merge し、tag と GitHub Release を作成する。
6. phdemo を公開済み v1 に向けて確認し、phdemo の CI を通してから merge する。
7. stable 文書と General registry への登録可能性を確認する。
