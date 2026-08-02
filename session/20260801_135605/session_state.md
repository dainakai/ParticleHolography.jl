# Session State

- Session: 20260801_135605
- Repo: /media/dai-server/DATAM2/siderepos/260801/ParticleHolography.jl
- Branch: codex/v1.0.0
- Started: 2026-08-01 13:56:05 JST
- Updated: 2026-08-02 14:58:17 JST

## Goal
ParticleHolography.jl を v1.0.0 品質へ引き上げ、同じ利用コードから CPU、macOS Metal、NVIDIA CUDA を選べる実行基盤、初心者向け文書、phdemo との明確な連携、各環境の CI 検証を整える。

## Current Subtask
文書確認後の追加要望を実装した release candidate を総点検し、push 後の Julia 1.10、macOS Metal、GitHub Actions 確認と公開操作を保守者判断へ渡す。

## Loaded Skills
- `nemo-rl-session-memory` - 長期作業を切断後も再開できるよう、状態・時系列・変更ファイル・引き継ぎを記録する。
- `nemo-rl-auto-research` - 本リポジトリは NeMo-RL ではないため実験キャンペーン機能は適用外。再現可能な基準検証、明示的な評価軸、実行記録という規律のみをバックエンド検証へ適用する。
- `github:github` - リポジトリ、Actions、リリース状態を確認し、外部への公開操作を明示的な許可なしに行わない。

## Current Status
初回 release candidate は ParticleHolography.jl `b050823`、phdemo `1b18b56` としてローカル commit 済みである。

文書確認後の追加実装も両リポジトリへローカル commit 済みである。

文書確認後に、process-wide の既定 backend、`PropagationGrid` と `PropagationKernel`、全出力組合せを一回の深さ走査で生成する `ReconstructionRequest`、平均値 padding、host・CUDA VRAM・Metal unified memory の事前診断を追加した。

CPU core は 197/197、CUDA shared/physical contract は 19/19、CUDA integration/legacy は 7/7、Metal shared contract は 18/18、Plots extension は 7/7、phdemo は 14/14 合格した。

CPU、CUDA、Metal、Plots の4レポートを統合した Codecov 行 coverage は project 82.56%、patch 82.2% であり、両 target を 80% に設定した。以前の90.1%は、未ロードの拡張を分母に含めないローカル集計だったため、全体値としては使用しない。

平面波の解析解、正負距離の往復伝搬、既知深度の吸収粒子が正しい slice へ再集束することを CPU と CUDA で確認した。

Documenter は生成図、doctest、cross-reference、render まで成功し、ローカル URL の Chrome 描画も確認した。

phdemo の `doctor` は 1024×1024×1024 の Float32 volume と N0f8 MinIP に 4.06 GiB 必要と診断し、文書どおりの CPU `doctor` と `smoke` が成功した。

Draft PR #90 を作成し、Julia 1.10、hosted OS、Metal、CUDA、docs、Plots の GitHub Actions は合格した。残りは PR review、phdemo PR、merge、tag、Release、General 登録である。

## Plan
- [x] 現行 CUDA テストを基準実行し、挙動・時間・失敗を記録する。
- [x] リポジトリ全体と履歴、CI、公開 API、計算カーネルの監査結果を設計書へまとめる。
- [x] CPU / Metal / CUDA の共通バックエンド設計と互換性方針を確定する。
- [x] v1.0.0 の実装、テスト、ドキュメント、CI、移行案内を段階的に行う。
- [x] ローカル CPU と NVIDIA CUDA の検証、workflow の静的検証、差分衛生を確認する。
- [x] リリース候補を総点検し、未実施の外部 CI と公開操作を明示して引き渡す。
- [x] 文書確認後の既定 backend、型名、複合出力、padding、memory 診断、物理 test、coverage、図、phdemo 導入手順を実装する。
- [x] push 後に GitHub macOS Metal、Julia 1.10、hosted OS 行列、self-hosted NVIDIA CI の結果を確認する。

## Assumptions
- 破壊的変更は許容されるが、既存機能と同等の処理能力を保ち、不要な新規アルゴリズムは追加しない。
- macOS / Metal の実機確認は GitHub-hosted macOS runner、CUDA は既存 self-hosted runner を利用する。
- タグ、Release、push、PR は別途明示的な許可が必要であり、まずローカルで完成度を上げる。
- Metal の最低要件は Metal.jl 1.10 に合わせて macOS 14+、Julia 1.10+ とする。
- JACC.jl は v1.0 で CPU/CUDA/Metal を支援するが、ライブラリ全体の global backend preference より、配列型 dispatch + AbstractFFTs + package extensions の方が同一プロセス内の明示的 backend object に適するため採用しない。

## Blockers
- None known.
