# ParticleHolography.jl v1.0.0 実装計画

- 作成日: 2026-08-01
- 対象: `dainakai/ParticleHolography.jl` と `dainakai/phdemo`
- 基準版: ParticleHolography.jl v0.2.4 (`c23ea07`)

## 1. 目標

同一の高水準 API から `backend(:cpu)`、`backend(:metal)`、`backend(:cuda)` を明示的に選び、Gabor wavefront、位相回復、三次元再構成、投影、粒子検出、軌跡作成、bundle adjustment という既存の処理系列を環境ごとに書き換えず実行できるようにする。

v1.0.0 では新しい再構成法や検出法を増やさず、現在の処理を次の観点で整理する。

1. CUDA 固定の型・関数・依存関係を backend-neutral にする。
2. FFT plan と作業配列を再利用し、フレームごとの確保を減らす。
3. CPU を数値上の参照実装とし、Metal/CUDA が同じ契約を満たすことを共有テストで確認する。
4. 重い可視化と GPU package を optional dependency にする。
5. 初心者が「画像を一枚再構成する」所から実データ処理まで迷わず進める文書と phdemo を用意する。

## 2. 現行基準

- Julia: 1.12.6
- GPU: NVIDIA GeForce RTX 4080 SUPER、CUDA.jl 6.2.1、CUDA runtime 13.3
- v0.2.4 test: 108 / 108 合格
- test suite 本体: 約 89.9 秒
- `Pkg.test` の隔離環境準備を含む全体: 約 271.7 秒
- 現行 import は CUDA/cuFFT、Makie/Plots を常に読み込み、CPU-only/Metal 環境で成立しない。
- 単一 test file が CUDA、巨大ボリューム、描画、追跡 fixture の上書きを混在させている。

## 3. 公開 API と backend 契約

### 3.1 backend object

コア package は次を公開する。

```julia
CPUBackend()
backend(:cpu | :cuda | :metal | :auto)
available_backends()
to_backend(backend, x)
to_host(x)
synchronize_backend(backend)
```

最終実装では CUDA.jl と Metal.jl が公開する名前との衝突を避けるため、`CUDABackend` と `MetalBackend` の実装型を export せず、利用者は `backend(:cuda)` と `backend(:metal)` を使う。

backend は呼出し時または reusable plan の生成時に渡す。process-global preference は採用しない。これにより同一 Julia process 内で CPU と GPU の参照結果を比較でき、library 利用者の global state も変更しない。

`backend(:auto)` は明示指定がない簡便用途だけに使い、優先順は CUDA、Metal、CPU とする。再現性が必要な例と phdemo は明示指定する。

### 3.2 package extensions

- core dependencies: `AbstractFFTs`, `FFTW` と処理に必要な軽量 package
- weak dependencies: `CUDA`, `Metal`, plotting stack
- `ParticleHolographyCUDAExt`: CUDA array、device、同期、CUDA 固有高速経路
- `ParticleHolographyMetalExt`: Metal array、device、同期
- plotting extension: plot recipes と診断描画

GPU extension が未ロードの場合は、該当 backend の constructor で導入・`using` 方法を示す明確な error を返す。CPU package import は GPU driver や display server を要求しない。

### 3.3 backend-neutral optical API

新しい主 API は接頭辞 `cu_` を持たない。

- `transfer_sqrt`, `transfer`
- `gabor_wavefront`
- `PhaseRetrievalPlan`, `phase_retrieval!`, `phase_retrieval`
- `ReconstructionPlan`, `reconstruct!`, `reconstruct`
- `reconstruct_complex`, `xyprojection!`, `xyprojection`
- `reconstruct_and_projection!`, `reconstruct_and_projection`
- `pad2d`, `asm_propagate!`, `asm_propagate`
- `lowpass_filter`, `highpass_filter`

array-returning API は原則として指定 backend 上の `Float32` / `ComplexF32` array を返す。保存・表作成など host 処理へ渡す境界だけ `to_host` を使う。

既存 `cu_*` API は v1 移行用 wrapper として残し、CUDA backend を選ぶ。非推奨警告と旧名→新名表を文書化する。旧 API の削除は v2 以降とする。

### 3.4 reusable plan/workspace

`PhaseRetrievalPlan` と `ReconstructionPlan` は次を所有する。

- backend と画像 shape
- 伝搬 transfer arrays
- `plan_fft` / `plan_ifft`
- Fourier domain と image domain の work buffers
- algorithm parameter と単位情報

allocating convenience API は一回用 plan を内部生成する。複数 frame の本番処理と phdemo は plan を一度作り、`!` API を反復使用する。

伝達関数は FFT-native frequency order で生成し、各反復の `fftshift` / `ifftshift` を避ける。数式、符号、座標、単位を API docstring と数値テストで固定する。

## 4. 既存処理の portability 方針

### 4.1 FFT 中心処理

Gabor wavefront、位相回復、再構成、ASM、周波数 filter は `AbstractArray` と `AbstractFFTs` plan の `mul!` を共通経路にする。CPU は FFTW、CUDA は CUDA.jl、Metal は Metal.jl の AbstractFFTs 実装を使う。

### 4.2 custom kernels

dilation、背景 mode、bundle adjustment の brute-force PIV は、既存の計算法と算術順序を保った portable kernel または明確な CPU fallback にする。性能最適化のために異なる解析法へ置き換えない。

粒子 CCL は host 参照経路をすべての backend で利用可能にし、CUDA の既存高速経路は extension へ隔離できる。Metal で host fallback が発生する境界と転送量は文書化する。

粒子座標・径・軌跡 API は `CuArray` 型制約を外す。空集合、境界接触、重なり、単一 slice、非連続 label、再現可能な UUID をテストする。

## 5. package 構造

目標構造は次の通り。

```text
src/
  ParticleHolography.jl
  backends.jl
  types.jl
  optics/
  detection/
  tracking/
  calibration/
  io/
ext/
  ParticleHolographyCUDAExt.jl
  ParticleHolographyMetalExt.jl
  ParticleHolographyPlottingExt.jl
test/
  runtests.jl
  core/
  shared/
  environments/{cuda,metal,plotting}/
docs/src/
  getting-started/
  concepts/
  guides/
  reference/
  troubleshooting.md
  migration-v1.md
```

ファイル分割は責務に基づき、単なる行数調整にはしない。validation、shape、units、error message を入口で統一する。

## 6. 検証計画

### 6.1 CPU core

- GPU package を install/import しない環境で `using ParticleHolography` が成功する。
- 小型 synthetic hologram で transfer、wavefront、phase retrieval、reconstruction、projection を検証する。
- reference 値、型、shape、finite、入力非破壊、`!` API の buffer 再利用を確認する。
- detection/tracking の edge case を fixture 非依存で確認する。
- test は追跡対象ファイルを書き換えず、必要な生成物は `mktempdir` に置く。

### 6.2 shared accelerator contract

同じ test source を CPU、Metal、CUDA に適用する。

- backend discovery と host/device round trip
- FFT plan `mul!`
- transfer と wavefront
- 位相回復 1–3 iteration
- 小型 3D reconstruction と XY projection
- CPU reference との `rtol` / `atol` 内一致
- repeated execution と同期
- invalid shape/parameter の同じ error contract

浮動小数点 reduction と FFT 実装差があるため bitwise equality は要求せず、物理量ごとに tolerance を定義する。

### 6.3 GitHub Actions

- `cpu.yml`: GitHub-hosted Ubuntu/macOS/Windows。Julia 1.10 LTS 相当の最低版と 1.12 current を含む。
- `metal.yml`: `macos-15`, `arch: aarch64`。Metal.jl 1.10+ を導入し、Metal が functional であることを確認して shared contract を実行する。
- `cuda.yml`: `[self-hosted, Linux, X64]`。job 冒頭で `nvidia-smi` と `CUDA.functional()` を hard-fail させ、shared contract と CUDA integration を実行する。
- `docs.yml`: CPU-only で doctest、link check、Documenter build。
- optional plotting tests を core から分離する。

self-hosted runner は現時点で GPU 固有 label がないため、誤配置を silent skip にしない。

### 6.4 性能・回帰

v0.2.4 の代表的 CUDA 入力と v1 の結果を比較し、再構成値、検出結果、軌跡 result の変化を記録する。benchmark は新しい処理方式の競争ではなく、plan reuse、allocation、host-device transfer の回帰検知に限定する。

## 7. 文書計画

初心者向け導線を次の順序にする。

1. ホログラムとは何か、再構成で何が得られるか
2. 10 分 CPU quickstart
3. backend 選択表と install 手順
4. 画像一枚 → wavefront → 3D volume → 粒子座標という end-to-end guide
5. wavelength、pixel pitch、distance、depth step、threshold の意味・単位・調整法
6. 位相回復、伝搬、検出、tracking の概念
7. memory 見積り、backend ごとの対応範囲、速度と精度
8. troubleshooting
9. phdemo の実データ tutorial
10. v0.2 から v1 の migration

すべての quickstart code は CPU CI で実行する。GPU-only の例には CPU equivalent を併記する。保存先、入力 dtype、座標順、z の符号、単位を省略しない。

## 8. phdemo 連携

phdemo は v1 package の公式な実データ tutorial として再構成する。

- 一つの config で data path、scene、光学 parameter、backend、output path を指定する。
- top-level `include` と global variable の連鎖を関数/command entrypoint に置き換える。
- package と重複する CUDA dilation/background code を削除し、v1 API を呼ぶ。
- `cpu`, `cuda`, `metal` の環境 profile と、小型 smoke/dry-run を用意する。
- README に段階別 command、想定時間、必要 memory、出力の読み方、ParticleHolography.jl 文書への versioned link を置く。
- package docs 側から phdemo の同じ revision/tag へ相互 link する。

raw data の再配布・削除は本件では行わず、既存 data layout を config で扱う。

## 9. 実装順序

1. v1 branch と session checkpoint を作る。
2. CPU-only import を成立させ、backend registry と generic array 型を追加する。
3. optical FFT core を generic 化し、CPU test を先に通す。
4. CUDA extension を接続し、v0.2.4 基準値と比較する。
5. Metal extension と shared contract を追加する。
6. detection/tracking/calibration を backend-neutral にし、edge-case test を追加する。
7. package 構造、validation、IO、plotting extension を整理する。
8. CI を CPU / Metal / CUDA / docs に分離する。
9. 初心者文書、API reference、migration guide を完成させる。
10. phdemo を config-driven workflow へ移行し、smoke test と相互 link を加える。
11. formatting、Aqua/ambiguity、CPU local、CUDA local を実行する。
12. GitHub-hosted Metal と self-hosted CUDA の CI 結果を確認し、release checklist を仕上げる。

## 10. v1.0.0 受け入れ基準

- [x] GPU/display library なしの clean CPU environment で import と core test が通る。
- [ ] 一つの tutorial code が backend object の一行だけを変えて CPU/Metal/CUDA で動く。
- [ ] CPU、Metal、CUDA が shared numerical contract を満たす。
- [x] existing CUDA feature set に相当する public workflow が維持されるか、migration guide に明示される。
- [x] reusable plan により multi-frame hot path で FFT plan と主要 buffer を再利用する。
- [x] tests が repository fixture を書き換えない。
- [x] CPU、Metal、CUDA、docs の CI が分離され、GPU がない runner で silent success しない。
- [x] 初心者 quickstart と end-to-end guide が CI 上で実行可能である。
- [x] phdemo に backend-neutral smoke workflow と package docs への相互 link がある。
- [x] `Project.toml`、docs、CHANGELOG、migration、release checklist が v1.0.0 を指す。
- [x] 未解決の性能差、Metal host fallback、必要 memory が文書化される。

Metal の実機 contract と三バックエンド共通 tutorial の実行確認は、push 後の GitHub Actions で完了させる。

## 11. 公開操作

local implementation と検証は進めるが、commit、push、PR、tag、GitHub Release はこの計画の範囲に自動では含めない。それぞれユーザーの明示許可後に行う。
