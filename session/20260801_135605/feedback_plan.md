# v1.0.0 文書確認後の追加設計

- 作成日: 2026-08-01
- 基準コミット: ParticleHolography.jl `b050823`、phdemo `1b18b56`

## 既定バックエンド

`backend(:cpu)`、`backend(:cuda)`、`backend(:metal)`、`backend(:auto)` は、利用可否を確認したバックエンドを返すと同時に、現在の Julia process の既定バックエンドへ設定する。

引数なしの `backend()` は、現在の既定バックエンドを返す。

バックエンド引数を省略できる関数は、配列の保存場所ではなく `backend()` の値を使う。

既存の明示的な `function(b, ...)` と `backend=b` は、同一 process 内で CPU と GPU を比較する用途のために残す。

既定バックエンドは process 全体の状態なので、複数 task が実行中に設定を書き換える使い方は対象外とする。

## 光伝搬の型名

現在の `TransferSqrtPart` は、伝搬距離に依存しない空間周波数格子を保持しており、名前と内容が一致していない。

この型の新しい名前を `PropagationGrid` とする。

現在の `Transfer` は、FFT 後の波面に乗算する距離依存の複素配列なので、新しい名前を `PropagationKernel` とする。

伝達関数という光学用語自体は正しいが、`PropagationKernel` の方が配列の用途をコード上で判別しやすい。

新しい関数名を `propagation_grid` と `propagation_kernel` とし、旧型名と `transfer_sqrt`、`transfer` は移行用 alias または非推奨 wrapper として残す。

`Wavefront` は複素光波を表す名前として維持する。

## 再構成出力の組合せ

一回の深さ走査から必要な出力だけを生成するため、`ReconstructionRequest` と `ReconstructionResult` を追加する。

`ReconstructionRequest` は slice 数、体積の要否と型、MinIP の要否と型を保持する。

体積は `nothing`、`N0f8`、`Float32`、`ComplexF32`、`ComplexF64` を受け付ける。

実数型の体積は強度 `abs2(wavefront)` を保持し、複素型の体積は再構成波面を保持する。

MinIP は `nothing`、`N0f8`、`Float32` を受け付ける。

`N0f8` の MinIP は Float32 で深さ方向の最小値を求めた後に一度だけ量子化する。

体積と MinIP を同時に要求した場合も、FFT と深さ方向の伝搬は一回だけ実行する。

体積を要求しない場合は三次元配列を確保せず、MinIP を要求しない複素体積では強度計算を省略する。

Metal の計算経路は ComplexF32 を使うため、ComplexF64 の出力を要求した場合は、各 slice を host の ComplexF64 体積へ変換する。

Metal で ComplexF64 を指定しても計算精度が Float64 へ上がるわけではないことを文書化する。

既存の `reconstruct`、`reconstruct_complex`、`xyprojection`、`reconstruct_and_projection` は、新しい一回走査の内部実装へ接続して返り値の互換性を保つ。

## 平均値パディング

`pad2d` に target shape と `mode=:mean` を追加し、平均値パディングであることを呼出し側から判別できる API にする。

パディング付き再構成は、padded volume 全体を確保してから crop する方式をやめる。

新しい処理は padded plane 上で伝搬し、各 slice の中央領域だけを最終体積と MinIP へ書き込む。

この方式は二倍幅のパディングで三次元出力メモリが八倍になる問題を避ける。

## メモリ診断

`MemoryDiagnostic` と `memory_diagnostic` を追加し、plan workspace、FFT の保守的な予備量、体積、MinIP、変換用一時配列を byte 単位で見積もる。

CPU は `Sys.free_memory()`、CUDA は `CUDA.free_memory()`、Metal は unified memory として `Sys.free_memory()` を利用可能量の基準にする。

既定の安全係数は 1.2 とし、`required_bytes × 1.2 > available_bytes` の場合は plan 作成前または出力確保前に `OutOfMemoryError` ではなく説明付き `ArgumentError` を返す。

実行者は `check_memory=false` を指定して診断による中断を無効化できる。

診断値は FFT implementation と allocator の内部量を完全には取得できないため、保証値ではなく保守的な事前判定として表示する。

## 再構成の妥当性テスト

平面波を伝搬したときに解析解 `exp(im * 2πz/λ)` と一致することを確認する。

既知の複素物体面を camera 面へ伝搬し、複数の候補深さから正しい slice へ戻したときに誤差が最小になることを確認する。

正距離と負距離の往復伝搬で元の波面を復元することを確認する。

体積五種類と MinIP 三種類の全有効組合せについて、型、数値、未要求配列の非確保を確認する。

CPU の物理テストを数値基準とし、CUDA と Metal の共有 contract でも同じ観測量を比較する。

CI の coverage 出力は Codecov へ継続送信し、ローカル計測値を記録して未実行分岐を追加テストで減らす。

## 文書と phdemo

「From hologram to particles」には、同じ package API で生成した hologram、焦点面、MinIP、検出位置の図を掲載する。

図を再生成する Julia script を repository に置き、静的画像と生成条件を対応づける。

バックエンド文書には CUDA.jl、Metal.jl、AbstractFFTs.jl、FFTW.jl の公式文書へのリンクを置く。

phdemo の通常導入手順は、phdemo だけを clone して `Pkg.instantiate()` を実行する形に変更する。

phdemo の `Project.toml` は `ParticleHolography = "1"` を維持するため、v1.0.0 登録後は package 本体の clone を必要としない。

未公開 branch を同時開発する場合だけ、別節で `Pkg.develop(path=...)` を案内する。

文書に掲載する `doctor` と `smoke` の command は、fresh phdemo checkout と登録済み ParticleHolography.jl を想定した CI で実行する。

## 完了条件

- `backend(:cpu)` の後にバックエンド引数なしの quickstart が動く。
- 全再構成出力の組合せが一回の伝搬走査から得られる。
- 平均値パディング付き処理が cropped output だけを確保する。
- plan 作成時と再構成直前にメモリ診断が動き、無効化もできる。
- CPU と CUDA の物理的妥当性テストが合格する。
- Metal 用の同じ contract が GitHub Actions に含まれる。
- 図入り文書と phdemo の通常導入手順がローカル build で確認できる。

## 実装結果

上記の完了条件は、ローカルで確認可能な CPU、CUDA、文書、phdemo、coverage の範囲ですべて満たした。

CPU core は 180/180、CUDA shared/physical contract は 19/19、CUDA integration/legacy は 7/7 合格した。

CPU line coverage は全体 90.1%、光学 core 96.7% だった。

macOS Metal contract と Julia 1.10 を含む GitHub Actions は、branch push 後の確認項目として残っている。
