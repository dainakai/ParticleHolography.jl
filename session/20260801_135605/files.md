# Files

## Inspected
- `Project.toml` - v0.2.4。CUDA/cuFFT と CairoMakie/Makie/Plots が必須、Julia 1.8 compat。
- `src/` - 全体構成と CUDA/Array 固定箇所を監査。再構成、CCL、dilation、背景 mode、PIV が CUDA 固定。
- `test/runtests.jl` - 621 行の単一 CUDA 前提 suite。大規模 1024x1024x1000 再構成と生成物書き換えを含む。
- `.github/workflows/CI.yml` - 全テストを label `self-hosted` の単一 job で実行。CPU/macOS/Metal 分離なし。
- `docs/` - 理論解説はあるが、backend/環境/parameter/エラー導線がなく、一部例が実行不能。
- `../phdemo/` - top-level 実行、hard-coded date/scenes、CUDA 固定、重複 dilation/background kernel、CI/test なし。
- GitHub Issues #58-#84 - v1.0.0 に関係する既知の構造/API/テスト/文書課題。
- GitHub PR #85 - weakdeps/extensions による CPU-only import の未マージ案。

## Changed
- `Project.toml` - v1.0.0、Julia 1.10+、AbstractFFTs/FFTW core、CUDA/Metal/Plots weakdeps/extensions。
- `src/ParticleHolography.jl` - backend/core/plot wrapper の include 構成へ変更。
- `src/types.jl` - CuArray 固定型を generic optical wrapper と互換 alias へ変更。
- `src/holofunc.jl` - backend-neutral optical API、FFT-native transfer、reusable plans、legacy wrappers へ全面更新。
- `src/frequency_filters.jl` - backend-neutral filter と shift-free FFT path へ更新。
- `src/utils.jl` - CUDA kernel を除去し、bounded-memory background mode と validation を追加。
- `src/ccl.jl` - host reference CCL と安全な bounding-box merge へ更新。
- `src/particle_detection.jl` - generic arrays、portable dilation、single-transfer host metrics へ更新。
- `src/bundleadjustment.jl` - backend-dispatched PIV、CPU reference、plot extension hook へ更新。
- `src/particle_tracking.jl` - Labonté 処理を整理し mutation/branch/validation bugs を修正。
- `src/plot_recipes.jl` - core から削除し optional Plots extension へ移動。
- `src/backends.jl` - CPU/CUDA/Metal backend registry と transfer/sync contract を追加。
- `src/plotting.jl` - optional plotting dispatch wrapper を追加。
- `ext/ParticleHolographyCUDAExt.jl` - CUDA arrays/device/sync と PIV kernels を追加。
- `ext/ParticleHolographyMetalExt.jl` - Metal arrays/sync を追加。
- `ext/ParticleHolographyPlotsExt.jl` - particles/trajectory/calibration diagnostics を追加。
- `.github/workflows/` - CPU/Metal/CUDA/docs/Plotsを独立jobへ分離。
- `test/core/` - 99件のCPU-only core testsへ置換。
- `test/shared/backend_contract.jl` - CPUとacceleratorの共有数値契約を追加。
- `test/environments/{cuda,metal,plotting}/` - optional integration環境を追加。
- `docs/src/getting-started/`, `concepts/`, `guides/` - beginner-first v1文書を追加。
- `docs/src/troubleshooting.md`, `migration-v1.md` - setup/移行案内を追加。
- `README.md`, `CHANGELOG.md`, `RELEASE_CHECKLIST.md` - v1 surface/release状態へ更新。
- `../phdemo/Project.toml` - PhDemo v1 application metadata/depsへ更新。
- `../phdemo/src/PhDemo.jl` - config-driven pipeline moduleを追加。
- `../phdemo/bin/phdemo.jl`, `configs/sample.yaml` - unified CLI/configを追加。
- `../phdemo/src/{backrem,bundle_adjustment,proc,particle_analysis,plots}.jl` - old hard-coded top-level scriptsを削除。
- `../phdemo/README.md`, `test/runtests.jl`, `.github/workflows/CI.yml` - v1 tutorial/CPU smoke contractへ更新。
- `session/20260801_135605/session_state.md` - baseline と確定した設計判断を更新。
- `session/20260801_135605/timeline.md` - CUDA baseline と Metal FFT/CI 調査を追記。
- `session/20260801_135605/files.md` - session artifact 一覧を更新。
- `session/20260801_135605/handoff.md` - 次の実装開始点を更新。
- `.codecov.yml` - 4環境の統合実測値に基づき project と patch coverage の status target を80%に設定。
- `docs/generate_assets.jl` - hologram から粒子検出までの四 panel 図を public API から再生成する script を追加。
- `docs/src/assets/hologram-to-particles.png` - 初学者向け workflow 図を追加。
- `src/backends.jl` - process-wide 既定 backend、引数なし取得、memory probe を追加。
- `src/types.jl` - `PropagationGrid` と `PropagationKernel` を主型名に変更。
- `src/holofunc.jl` - 複合出力の一回走査、平均値 padding、memory 診断、物理 API test 対応を追加。
- `test/core/optics.jl` - 全出力組合せ、memory、padding、解析解、往復伝搬、既知深度再集束を追加。
- `test/shared/backend_contract.jl` - CUDA と Metal の既知深度再集束契約を追加。
- `../phdemo/README.md` - 登録済み v1 を `Pkg.instantiate()` で利用する通常導入と local development を分離。
- `../phdemo/src/PhDemo.jl` - process-wide backend と複合再構成出力、memory doctor を利用するように変更。
- `../phdemo/.github/workflows/CI.yml` - 登録済み v1 を通常利用し、未公開 revision override を任意入力に変更。

## Generated
- `session/20260801_135605/` - 長期作業の再開用セッション記録。
- `session/20260801_135605/v1_plan.md` - v1.0.0 の API、package、test、CI、docs、phdemo、受け入れ基準を定めた実装計画。
- `session/20260801_135605/regression_v1.jl` - v0.2.4基準とv1 CPU/CUDAを同一入力で比較する再現driver。
- `session/20260801_135605/regression_report.md` - v0.2.4/v1の実GPU数値回帰結果。
- `session/20260801_135605/release_candidate_report.md` - 実装内容、検証結果、未確認事項、公開順序をまとめた最終報告書。
- `session/20260801_135605/feedback_plan.md` - 文書確認後に追加された既定 backend、型名、再構成出力、メモリ診断、妥当性テスト、図、phdemo の設計。
