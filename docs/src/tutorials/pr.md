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

```@example
using ParticleHolography

# Load images
img1 = load_gray2float("../assets/impcam1_enhanced.png")
img2 = load_gray2float("../assets/impcam2_enhanced.png")

# Bundle adjustment
coeffs = get_distortion_coefficients(img1, img2, verbose=true)
```

``verbose=true``を指定すると、バンドルアジャストメントの変換前後の画像とズレ量マップが保存されます。

![Before bundle adjustment](../assets/before_BA.jpg)
*Before bundle adjustment*

![After bundle adjustment](../assets/after_BA.jpg)
*After bundle adjustment*

