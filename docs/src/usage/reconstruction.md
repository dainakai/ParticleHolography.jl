# Reconstruction

必要な前処理が完了し、`ParticleHolography.CuWaveFront` 型の配列 `d_wavefront` が得られたとします。この配列を用いて、観測体積の３次元再構成を行います。[Gabor holography](@ref gabor_reconst)ではすでに観測体積の光軸方向の投影画像を取得する関数について説明しましたが、その他にも必要な情報に応じて以下の再構成用関数を選択することができます。

```@docs
cu_get_reconst_xyprojection
cu_get_reconst_vol
cu_get_reconst_complex_vol
cu_get_reconst_vol_and_xyprojection
```