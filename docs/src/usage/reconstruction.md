# Reconstruction

We assume that the necessary preprocessing has been completed and an array of type `ParticleHolography.CuWaveFront`, `d_wavefront`, has been obtained. Using this array, we perform a 3D reconstruction of the observed volume. In [Gabor holography](@ref gabor_reconst), we have already described the functions for obtaining the projection images in the optical axis direction of the observed volume. In addition to this, depending on the required information, you can select the following reconstruction functions.


```@docs
cu_get_reconst_xyprojection
cu_get_reconst_vol
cu_get_reconst_complex_vol
cu_get_reconst_vol_and_xyprojection
```