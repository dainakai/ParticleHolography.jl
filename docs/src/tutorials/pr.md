# Phase retrieval holography

Please refer to [Phase retrieval holography](@ref pr_explain) for the principles of this method. Below, we show the necessary procedures and an implementation example for reconstructing using this method.

## Bundle adjustment

We perform bundle adjustment [okatani](@cite) to correct for rotational and aberrational misalignments between the two camera views in the ``xy ``plane. First, for a pair of images with densely distributed features throughout the field of view, such as a glass plate with printed random dots, we create a vector map of displacement amounts (right figure below) by calculating the cross-correlation coefficients between neighboring batches between the two images, similar to Particle Image Velocimetry (PIV) [willert](@cite). This map represents the displacement of `img2` relative to the reference image `img1`. By determining the image transformation coefficients ``\bm{a}`` that make this map nearly zero throughout, alignment is achieved.

```math
\begin{aligned}
x' &= a_1 + a_2 x + a_3 y + a_4 x^2 + a_5 xy + a_6 y^2 \\
y' &= a_7 + a_8 x + a_9 y + a_{10} x^2 + a_{11} xy + a_{12} y^2
\end{aligned}
```

Prepare a set of benchmark images, such as a glass plate with printed random dots. The following are Gabor reconstruction images of random dot holograms.

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

We perform bundle adjustment on these images. If `verbose=true` is specified, the images before and after the bundle adjustment transformation and the displacement map are saved. If not specified (default is verbose=false), only the transformation coefficients are returned.

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

![Before bundle adjustment](../assets/before_BA.jpg)
*Before bundle adjustment*

![After bundle adjustment](../assets/after_BA.jpg)
*After bundle adjustment*

Using the coefficient array obtained in this way, we correct the distortion of the captured images.

```julia
img2_corrected = quadratic_distortion_correction(img2, coeffs)
```

## 