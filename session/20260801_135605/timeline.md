# Timeline

## 2026-08-01 13:56:05 JST
- User asked: ParticleHolography.jl v1.0.0 に向けて CPU / Metal / CUDA の共通実行基盤、初心者向け文書、phdemo 連携、macOS と NVIDIA の CI を整え、コードベース全体を最適化する。
- Context gathered: 作業ディレクトリが空であることを確認し、ParticleHolography.jl と phdemo をクローンした。ParticleHolography.jl の最新 Release は v0.2.4、default branch は main。
- Decision: NeMo-RL 専用の auto-research ブランチ・学習ジョブは本件に適用しない。セッション記録と、再現可能なバックエンド比較の検証規律を採用する。
- Result: セッション 20260801_135605 を開始した。

## 2026-08-01 14:02:32 JST
- Context gathered: 全 src、test 構成、docs、Actions、phdemo、公開 Issue/PR を監査した。open Issue #58-#84 が今回の主要課題を具体化しており、PR #85 は CUDA/plotting の weak dependency 化を試している。
- Decision: PR #85 の optional dependency 分離を参考にするが、CPU import stub に留めず、backend-neutral API と reusable FFT plan/workspace を v1.0.0 の中心にする。JACC の global preference はライブラリ API には採らず、AbstractArray/AbstractFFTs と extensions を第一候補にする。
- Context gathered: Metal.jl 1.10 は AbstractFFTs の FFT/plan API を提供する。GitHub-hosted `macos-15` は arm64 M1。self-hosted NVIDIA runner は online だが GPU 固有 label はない。
- Result: 現行 CUDA テストを基準実行する前の checkpoint を保存した。

## 2026-08-01 14:18:42 JST
- Context gathered: Julia 1.12.6 / CUDA.jl 6.2.1 / RTX 4080 SUPER で v0.2.4 の全 test を実行した。108 / 108 合格、suite 約 89.9 秒、隔離環境準備を含む `Pkg.test` 全体約 271.7 秒。
- Observation: test は Qt/GR display warning を出し、追跡対象の画像・JSON fixture を上書きする。基準確認後、その fixture だけを HEAD へ復元した。
- Context gathered: Metal.jl 1.10 の FFT 実装と test を確認し、`plan_fft`, `plan_ifft`, `LinearAlgebra.mul!` が `MtlArray` で利用可能と確認した。公式 CI は `macos-15` / aarch64 で paravirtualized GPU の core test を行う。
- Decision: process-global backend は使わず、明示的 backend object、AbstractArray/AbstractFFTs、package extensions、reusable plan/workspace を採る。CPU を数値参照とする。
- Result: `v1_plan.md` に実装順、API、CI、文書、phdemo、受け入れ基準を固定した。

## 2026-08-01 14:24:30 JST
- Decision: local branch `codex/v1.0.0` を `main` から作成した。commit/push/PR は行わない。
- Current work: GPU/plotting weak dependencies、backend registry、generic wrappers、AbstractFFTs plan を使う optical core の実装を開始する。

## 2026-08-01 14:53:00 JST
- Changed: `Project.toml` を 1.0.0 / Julia 1.10+ とし、CUDA、Metal、Plots を weak dependency + extension に分離した。
- Changed: CPU/CUDA/Metal backend objects、transfer/wavefront/filter の generic wrapper、FFT-native transfer、PhaseRetrievalPlan、ReconstructionPlan、backend-neutral optical API と v0.2 `cu_*` wrappers を追加した。
- Changed: CCL、dilation、particle coordinate/diameter、background mode、bundle adjustment/PIV、tracking を CUDA 型制約から分離した。CCL/Metal calibration の host fallback 境界を明示した。
- Fixed: deterministic UUID collision、Dict iteration 中の deletion、inclusive bounding-box edge、trajectory branch loss、zero denominator/boundary peak を処理した。
- Result: `src/` と `ext/` の全 Julia file が `Meta.parseall` に成功し、`git diff --check` も成功した。
- Next: dependency resolve、CPU import、small contract tests で実行時問題を修正する。

## 2026-08-01 15:35:00 JST
- Result: CUDA/Metal/Plotsを読み込まないCPU-only precompile/importに成功した。
- Result: CPU core 95/95、CUDA shared numerical contract 8/8、CUDA legacy/integration 6/6が合格した。CUDA側ではN0f8 legacy reconstruction、host-fallback CCL、device PIV kernelも確認した。
- Decision: CUDA.jl/Metal.jl自身がexportするbackend/synchronize名との衝突を避け、利用者には`backend(:cuda|:metal)`と`synchronize_backend`を公開する。
- Changed: GitHub ActionsをCPU matrix、macOS Metal、self-hosted CUDA、docs、Plots extensionへ分離した。Metal/CUDAはfunctionalでなければfailする。
- Changed: beginner-first documentation、README、CHANGELOG、migration guide、release checklistを追加した。
- Changed: phdemoを`PhDemo` v1 applicationへ作り直し、設定駆動CLIとdoctor/smoke/background/calibrate/process/trackを追加、重複CUDA kernelとtop-level実行scriptを削除した。
- Next: Particle依存軽量化後のresolve、docs build、phdemo CPU doctor/smoke、Plots extensionを実行する。

## 2026-08-01 15:38:18 JST
- Result: Documenter local buildはdoctest、cross-reference、renderまで成功した。deployを含まない`docs/build_local.jl`とCI専用`docs/make.jl`を分離した。
- Result: phdemoはsibling v1 packageをdevelopした環境で9/9合格し、既存100組の1024x1024 BMPに対するdoctorと64x64x4 CPU smoke CLIを確認した。
- Result: Plots extensionはparticle/trajectory plotに加え校正診断3画像の実保存まで5/5合格した。
- Fixed: CUDA driver未搭載時の`isfunctional`判定を例外安全にし、Metal contractの配列生成を明示的`MtlArray`へ変更した。phdemoの`calibrate --verbose`はPlotsを自動loadし、未導入時に導入方法を返す。
- Result: v0.2.4 `c23ea07`を一時worktreeからRTX 4080 SUPERで実行し、v1 CPU/CUDAと比較した。再構成・投影のv0.2差は最大6.1e-6、複素体積は1.5e-5未満、v1 CPU/CUDA差は1e-6未満だった。
- Next: 全suite最終再実行、workflow/diff/API衛生、外部CI以外のrelease checklistを締める。

## 2026-08-01 15:58:25 JST
- Result: isolated CPU package test 99/99、CUDA shared contract 10/10、CUDA integration/legacy 7/7、Plots 5/5、phdemo 9/9 が最終状態で合格した。
- Result: docs build、全 Julia source parse、ambiguity 0 件、workflow YAML と Project TOML parse、両リポジトリの `git diff --check` が合格した。
- Security: 永続 self-hosted CUDA runner で外部 pull request のコードを実行しないよう、CUDA workflow から `pull_request` trigger を除外した。
- Result: ローカルで検証済みの項目と、Julia 1.10、macOS Metal、GitHub Actions の未確認項目を `release_candidate_report.md` に分離して記録した。
- Pending: commit、push、PR、tag、GitHub Release は許可されていないため実施していない。

## 2026-08-01 16:05:00 JST
- Result: 初回 release candidate を ParticleHolography.jl `b050823` と phdemo `1b18b56` としてローカル commit した。
- User feedback: backend 引数の省略、光伝搬型名、処理の図解、登録済み v1 を使う phdemo 導入、依存 package の公式リンク、複合再構成出力、平均値 padding、memory 診断、物理妥当性 test、coverage 改善を追加要望として受けた。
- Decision: 既定 backend は process-wide とし、並行 task 中の切替えは対象外、明示 backend 引数は比較用途に残す。
- Result: 追加要望の詳細設計と完了条件を `feedback_plan.md` に記録した。

## 2026-08-01 17:15:00 JST
- Changed: `PropagationGrid`、`PropagationKernel`、`ReconstructionRequest`、`ReconstructionResult`、`MemoryDiagnostic`、`reconstruct_padded` を追加した。
- Changed: 実数 volume、複素 wavefront volume、MinIP の全有効組合せを一回の伝搬 loop で生成し、不要な三次元配列と不要な強度計算を省くようにした。
- Changed: 平均値または zero padding を追加し、padded plane で伝搬しながら中央の元画像領域だけを出力するようにした。
- Changed: CPU free memory、CUDA free VRAM、Metal unified memory を基準に、既定安全係数 1.2 の事前診断と `check_memory=false` override を追加した。
- Result: 128×128×96 の CPU 参考計測では、Float32 volume と N0f8 MinIP の同時生成は別々の二回処理より 1.76 倍高速だった。

## 2026-08-01 17:38:03 JST
- Result: CPU core 180/180、CUDA shared/physical contract 19/19、CUDA integration/legacy 7/7、Plots 5/5、phdemo 14/14 が合格した。
- Result: CPU line coverage は全体 90.1%、`src/holofunc.jl` は 96.7% だった。
- Result: 平面波解析解、往復伝搬、既知深度への再集束を CPU で検証し、同じ既知深度 test と CPU 数値比較を RTX 4080 SUPER の CUDA contract で検証した。
- Changed: 「From hologram to particles」に public API で再生成する四 panel 図を追加し、CUDA.jl、Metal.jl、AbstractFFTs.jl、FFTW.jl の公式文書へ接続した。
- Result: Documenter build と Chrome 描画が成功し、更新済み文書を `http://127.0.0.1:8766/` で提供している。
- Changed: phdemo の通常導入を phdemo の clone と `Pkg.instantiate()` だけにし、ParticleHolography の clone と `Pkg.develop` は未公開版を同時開発する節へ分離した。
- Result: phdemo `doctor` は full 設定に 4.06 GiB 必要で、111.13 GiB の host 空き memory に対して安全と診断した。
- Result: 文書確認後の追加実装を ParticleHolography.jl の `feat!: add default backend and unified reconstruction outputs` と phdemo `948e1eb` へローカル commit した。
- Pending: push、pull request、tag、GitHub Release、macOS Metal workflow は未実施である。
