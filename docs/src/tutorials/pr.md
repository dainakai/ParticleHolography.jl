# Phase retrieval holography

[Phase retrieval holography](@ref pr_explain)

## Bundle adjustment

ランダムドットを印刷したガラスプレートなどの撮影画像組を用意

```@raw html
<div style="display:flex; align-items:flex-start;">
   <div style="flex:1; margin-right:10px;">
       <img src="../../assets/impcam1_enhanced.png" alt="Camera 1 image" style="max-width:100%; height:auto;">
       <p style="text-align:center;">Camera 1 image</p>
   </div>
   <div style="flex:1;">
       <img src="../../assets/impcam2_enhanced.png" alt="Camera 2 image" style="max-width:100%; height:auto;">
       <p style="text-align:center;">Camera 2 image</p>
   </div>
</div>
```

これらに対してバンドルアジャストメントを行う。すなわち、二次の画像変換後の画像のズレが最小となるような12個の係数配列を求める。

```julia
using ParticleHolography

# Load images
img1 = load_gray2float("../assets/impcam1_enhanced.png")
img2 = load_gray2float("../assets/impcam2_enhanced.png")

# Bundle adjustment
coeffs = get_distortion_coefficients(img1, img2, verbose=true)
```

```
12-element Vector{Float64}:
1.1502560374425654
0.9971390059912079
0.005022879500243858
9.62795044814447e-7
-7.032017562352377e-7
1.3542774090666107e-7
5.521164421321545
-0.005516712482369036
1.0009145355970703
-3.4727403247879636e-8
7.851521359601221e-7
-1.749346158409748e-6
```

`verbose=true`を指定すると、バンドルアジャストメントの変換前後の画像とズレ量マップが保存されます。

![Before bundle adjustment](../assets/before_BA.jpg)
*Before bundle adjustment*

![After bundle adjustment](../assets/after_BA.jpg)
*After bundle adjustment*

こうして得た係数配列を用いて、撮影画像の歪みを補正します。

```julia
img2_corrected = quadratic_distortion_correction(img2, coeffs)
```