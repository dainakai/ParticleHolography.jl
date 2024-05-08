var documenterSearchIndex = {"docs":
[{"location":"whats_inline_holography/#Gabor's-inline-holography","page":"What's inline holography?","title":"Gabor's inline holography","text":"","category":"section"},{"location":"whats_inline_holography/#Introduction","page":"What's inline holography?","title":"Introduction","text":"","category":"section"},{"location":"whats_inline_holography/","page":"What's inline holography?","title":"What's inline holography?","text":"Holography is an imaging and measurement technique first proposed by D. Gabor in 1948 [1]. The interference pattern between the object light, which is diffracted by obstacles such as particles, and a reference light is recorded on a photosensitive material like a film, which is called a hologram. When the hologram is illuminated with a reconstruction light, the original light field is partially reproduced. In the case of small opaque objects like particles, the reconstructed light field is obstructed at the particle positions, appearing as dark images in the reconstruction. This allows the 3D position and shape of the objects to be observed. Currently, photosensitive materials have been replaced by digital cameras, and hologram reconstruction is commonly performed using digital image processing techniques such as numerical light propagation calculations. In an in-line holography setup, the coherent parallel light, the object observation volume, and the camera plane are all arranged along the same axis, eliminating the need to separate the object and reference light, and simplifying the reconstruction calculation. This article explains the light propagation calculation for parallel light, hologram recording and reconstruction, and the phase retrieval method, which is an advanced technique derived from Gabor holography.","category":"page"},{"location":"whats_inline_holography/#Collimated-light-propagation","page":"What's inline holography?","title":"Collimated light propagation","text":"","category":"section"},{"location":"whats_inline_holography/","page":"What's inline holography?","title":"What's inline holography?","text":"The propagation of parallel light follows the Helmholtz equation and can be calculated quickly and accurately using the angular spectrum method [2] [3]. Defining the optical axis of the parallel light as the z axis and the plane perpendicular to it as the xy plane, the wavefront of the parallel light at z=z_0 is denoted as psi(x y z_0). The light field propagated by Delta z in the positive z direction is given by the following equation:","category":"page"},{"location":"whats_inline_holography/","page":"What's inline holography?","title":"What's inline holography?","text":"psi(xyz_0 + Delta z) = mathcalF^-1left mathcalFpsi(xyz_0) cdot expleft( fracmathrmj2pi Delta zlambda sqrt1-left( lambda alpha right)^2 - left( lambda beta right)^2 right) right","category":"page"},{"location":"whats_inline_holography/","page":"What's inline holography?","title":"What's inline holography?","text":"Here, mathcalF denotes the two-dimensional Fourier transform, alpha beta are the Fourier domain variables corresponding to xy, and lambda is the wavelength. This equation shows that by Fourier transforming the wavefront of the parallel light, multiplying it by the propagation function H_Delta z=expleft( fracmathrmj2pi Delta zlambda sqrt1-left( lambda alpha right)^2 - left( lambda beta right)^2 right), and then performing an inverse Fourier transform, the light field propagated by Delta z in the positive z direction can be obtained.","category":"page"},{"location":"whats_inline_holography/#Hologram-recording-(Computer-generated-holography)","page":"What's inline holography?","title":"Hologram recording (Computer generated holography)","text":"","category":"section"},{"location":"whats_inline_holography/","page":"What's inline holography?","title":"What's inline holography?","text":"Holography is a technique for reconstructing 3D images from obtained holograms, but it is also possible to numerically generate holograms by modeling the objects and their arrangements. Generally, an opaque particle with radius r_0 located at (x_0y_0z_0) can be treated as a circular disk with zero thickness on the object plane x_0-y_0 [4]. The object plane A_0 is represented as follows:","category":"page"},{"location":"whats_inline_holography/","page":"What's inline holography?","title":"What's inline holography?","text":"A_0(xy) = begincases\n1  textif quad (x-x_0)^2 + (y-y_0)^2 leq r_0^2 \n0  textotherwise\nendcases","category":"page"},{"location":"whats_inline_holography/","page":"What's inline holography?","title":"What's inline holography?","text":"The diffraction pattern of the particle appears on the hologram as the parallel light is blocked at the points where the object plane has a value of 1. The hologram can be calculated using the following equation:","category":"page"},{"location":"whats_inline_holography/","page":"What's inline holography?","title":"What's inline holography?","text":"I(xy z_0+Delta z) = left mathcalF^-1left mathcalFpsi(xyz_0)cdot left(1-A_0right) cdot H_Delta z right right^2","category":"page"},{"location":"whats_inline_holography/","page":"What's inline holography?","title":"What's inline holography?","text":"\\psi(x,y;z_0) is the wavefront of the parallel light just before passing through the object plane. If no objects exist before this point, the phase at each point can be set to 0, and we can assume psi(xyz_0)=1. Even if objects exist before this point, we can calculate the light field propagated from the position of the farthest particle by setting the phase there to 0 and using the same method as in the above equation. Since the hologram is the intensity distribution of the light field, it is represented by the square of the amplitude of the wavefront. The light field is a complex number, so this calculation involves taking the product with its complex conjugate.","category":"page"},{"location":"whats_inline_holography/#Hologram-reconstruction","page":"What's inline holography?","title":"Hologram reconstruction","text":"","category":"section"},{"location":"whats_inline_holography/","page":"What's inline holography?","title":"What's inline holography?","text":"D. Gabor. A new microscopic principle. Nature 161, 777–778 (1948).\n\n\n\nJ. W. Goodman. Introduction to Fourier Optics –3rd ed. (Roberts and Company publishers, 2005); pp. 42–62.\n\n\n\nM. A. T. M. Kreis and W. P. Jüptner. Methods of digital holography: a comparison. In Optical Inspection and Micromeasurements II, SPIE 3098, 224–233 ((1997)).\n\n\n\nC. S. Vikram. Particle Field Holography (Cambridge University Press, (1992)); p. 34.\n\n\n\n","category":"page"},{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"This is the reference documentation for the ParticleHolography package.","category":"page"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/#Functions","page":"Reference","title":"Functions","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [ParticleHolography]","category":"page"},{"location":"reference/#ParticleHolography.cu_get_reconst_vol-Tuple{CUDA.CuArray{ComplexF32, 2}, CUDA.CuArray{ComplexF32, 2}, CUDA.CuArray{ComplexF32, 2}, Int64}","page":"Reference","title":"ParticleHolography.cu_get_reconst_vol","text":"cu_get_reconst_vol(holo, transfer_front, transfer_dz, slices)\n\nReconstruct the observation volume from the light field light_field using the transfer functions transfer_front and transfer_dz. transfer_front propagates the light field to the front of the volume, and transfer_dz propagates the light field between the slices. slices is the number of slices in the volume.\n\nArguments\n\nlight_field::CuArray{ComplexF32,2}: The light_field to reconstruct. In Gabor's holography, this is the square root of the hologram.\ntransfer_front::CuArray{ComplexF32,2}: The transfer function to propagate the light field to the front of the volume.\ntransfer_dz::CuArray{ComplexF32,2}: The transfer function to propagate the light field between the slices.\nslices::Int: The number of slices in the volume.\n\nReturns\n\nCuArray{Float32,3}: The reconstructed volume.\n\n\n\n\n\n","category":"method"},{"location":"reference/#ParticleHolography.cu_get_reconst_vol_and_xyprojection-Tuple{CUDA.CuArray{ComplexF32, 2}, CUDA.CuArray{ComplexF32, 2}, CUDA.CuArray{ComplexF32, 2}, Int64}","page":"Reference","title":"ParticleHolography.cu_get_reconst_vol_and_xyprojection","text":"cu_get_reconst_vol_and_xyprojection(light_field, transfer_front, transfer_dz, slices)\n\nReconstruct the observation volume from the light field light_field and get the XY projection of the volume using the transfer functions transfer_front and transfer_dz. transfer_front propagates the light field to the front of the volume, and transfer_dz propagates the light field between the slices. slices is the number of slices in the volume.\n\nArguments\n\nlight_field::CuArray{ComplexF32,2}: The light_field to reconstruct. In Gabor's holography, this is the square root of the hologram.\ntransfer_front::CuArray{ComplexF32,2}: The transfer function to propagate the light field to the front of the volume.\ntransfer_dz::CuArray{ComplexF32,2}: The transfer function to propagate the light field between the slices.\nslices::Int: The number of slices in the volume.\n\nReturns\n\nCuArray{Float32,3}: The reconstructed volume.\nCuArray{Float32,2}: The XY projection of the reconstructed volume.\n\n\n\n\n\n","category":"method"},{"location":"reference/#ParticleHolography.cu_get_reconst_xyprojection-Tuple{CUDA.CuArray{ComplexF32, 2}, CUDA.CuArray{ComplexF32, 2}, CUDA.CuArray{ComplexF32, 2}, Int64}","page":"Reference","title":"ParticleHolography.cu_get_reconst_xyprojection","text":"cu_get_reconst_xyprojectin(light_field, transfer_front, transfer_dz, slices)\n\nGet the XY projection of the reconstructed volume from the light field light_field using the transfer functions transfer_front and transfer_dz. transfer_front propagates the light field to the front of the volume, and transfer_dz propagates the light field between the slices. slices is the number of slices in the volume.\n\nArguments\n\nlight_field::CuArray{ComplexF32,2}: The light_field to reconstruct. In Gabor's holography, this is the square root of the hologram.\ntransfer_front::CuArray{ComplexF32,2}: The transfer function to propagate the light field to the front of the volume.\ntransfer_dz::CuArray{ComplexF32,2}: The transfer function to propagate the light field between the slices.\nslices::Int: The number of slices in the volume.\n\nReturns\n\nCuArray{Float32,2}: The XY projection of the reconstructed volume.\n\n\n\n\n\n","category":"method"},{"location":"reference/#ParticleHolography.cu_phase_retrieval_holo-Tuple{CUDA.CuArray{Float32, 2}, CUDA.CuArray{Float32, 2}, CUDA.CuArray{ComplexF32, 2}, CUDA.CuArray{ComplexF32, 2}, Int64, Int64}","page":"Reference","title":"ParticleHolography.cu_phase_retrieval_holo","text":"cu_phase_retrieval_holo(holo1, holo2, transfer, invtransfer, priter, datlen)\n\nPerform the Gerchberg-Saxton algorithm-based phase retrieving on two holograms and return the retrieved light field at the z-coordinate point of holo1. The algorithm is repeated priter times. holo1 and holo2 are the holograms (I = |phi|^2) of the object at two different z-coordinates. transfer and invtransfer are the transfer functions for the propagation from holo1 to holo2 and vice versa. datlen is the size of the holograms.\n\nArguments\n\nholo1::CuArray{Float32,2}: The hologram at the z-cordinate of closer to the object.\nholo2::CuArray{Float32,2}: The hologram at the z-coordinate of further from the object.\ntransfer::CuArray{ComplexF32,2}: The transfer function from holo1 to holo2.\ninvtransfer::CuArray{ComplexF32,2}: The transfer function from holo2 to holo1.\npriter::Int: The number of iterations to perform the algorithm.\ndatlen::Int: The size of the holograms.\n\nReturns\n\nCuArray{ComplexF32,2}: The retrieved light field at the z-coordinate of holo1.\n\n\n\n\n\n","category":"method"},{"location":"reference/#ParticleHolography.cu_transfer-Tuple{AbstractFloat, Int64, AbstractFloat, CUDA.CuArray{Float32, 2}}","page":"Reference","title":"ParticleHolography.cu_transfer","text":"cu_transfer(z0, datLen, wavLen, d_sqr)\n\nCreate a CuArray of size datLen x datLen with the values of the transfer function for a given propagated distance z0. d_sqr can be obtained with cutransfersqrtarr(datlen, wavlen, dx).\n\nArguments\n\nz0::AbstractFloat: The distance to propagate the wave.\ndatLen::Int: The size of the CuArray.\nwavLen::AbstractFloat: The wavelength of the light.\nd_sqr::CuArray{Float32,2}: The square of the distance from the center of the hologram, obtained with cutransfersqrtarr(datlen, wavlen, dx).\n\nReturns\n\nCuArray{ComplexF32,2}: The transfer function for the propagation.\n\n\n\n\n\n","category":"method"},{"location":"reference/#ParticleHolography.cu_transfer_sqrt_arr-Tuple{Int64, AbstractFloat, AbstractFloat}","page":"Reference","title":"ParticleHolography.cu_transfer_sqrt_arr","text":"cu_transfer_sqrt_arr(datlen, wavlen, dx)\n\nCreate a CuArray of size datlen x datlen with the values of the square-root part of the transfer function.\n\nArguments\n\ndatlen::Int: The size of the CuArray.\nwavlen::AbstractFloat: The wavelength of the light.\ndx::AbstractFloat: The pixel size of the hologram.\n\nReturns\n\nCuArray{Float32,2}: The square-root part of the transfer function.\n\n\n\n\n\n","category":"method"},{"location":"reference/#ParticleHolography.cvgray2floatimg-Tuple{Any}","page":"Reference","title":"ParticleHolography.cvgray2floatimg","text":"cvgray2floatimg(img)\n\nConvert a OpenCV capable image to a Array{Float32, 2} image.\n\nArguments\n\nimg::Array{N0f8, 3}: The image to convert.\n\nReturns\n\nArray{Float32, 2}: The image as a Float32 array.\n\n\n\n\n\n","category":"method"},{"location":"reference/#ParticleHolography.find_external_contours-Tuple{Any}","page":"Reference","title":"ParticleHolography.find_external_contours","text":"find_external_contours(image)\n\nFinds non-hole contours in binary images. Equivalent to CVRETREXTERNAL and CVCHAINAPPROX_NONE modes of the findContours() function provided in OpenCV.\n\nArguments\n\nimage: The binary image. \n\nReturns\n\nVector{Vector{CartesianIndex}}: A vector of contours. Each contour is a vector of CartesianIndex.\n\n\n\n\n\n","category":"method"},{"location":"reference/#ParticleHolography.floatimg2cvgray-Tuple{Matrix{Float32}}","page":"Reference","title":"ParticleHolography.floatimg2cvgray","text":"floatimg2cvgray(img)\n\nConvert a Array{Float32, 2} image to a OpenCV capable image.\n\nArguments\n\nimg::Array{Float32, 2}: The image to convert.\n\nReturns\n\nArray{N0f8, 3}: The image as a OpenCV capable image.\n\n\n\n\n\n","category":"method"},{"location":"reference/#ParticleHolography.load_gray2float-Tuple{String}","page":"Reference","title":"ParticleHolography.load_gray2float","text":"load_gray2float(path)\n\nLoad a grayscale image from a file and return it as a Array{Float32, 2} array.\n\nArguments\n\npath::String: The path to the image file.\n\nReturns\n\nArray{Float32, 2}: The image as a Float32 array.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = ParticleHolography","category":"page"},{"location":"#ParticleHolography","page":"Home","title":"ParticleHolography","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for ParticleHolography.","category":"page"},{"location":"","page":"Home","title":"Home","text":"A package for particle measurement using inline holography.","category":"page"},{"location":"","page":"Home","title":"Home","text":"note: Note\nThis package is under development, and none of the functions are guaranteed to work.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using Pkg\nPkg.add(url=\"https://github.com/dainakai/ParticleHolography.jl.git\")","category":"page"},{"location":"#Quick-Demonstration","page":"Home","title":"Quick Demonstration","text":"","category":"section"},{"location":"#GPU-accelerated-Gabor-reconstruction","page":"Home","title":"GPU-accelerated Gabor reconstruction","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The code below is an example of performing inline holographic reconstruction using an NVIDIA GPU (CUDA.jl). Your computer needs to be ready to use NVIDIA GPUs with CUDA.jl. It reconstructs a volume of size datlenΔx x datlenΔx x slicesΔz when the camera plane is considered as the xy plane and the direction perpendicular to the camera plane, which is the optical axis, is the z axis. Furthermore, it creates an xy projection image of the reconstructed volume by taking the minimum value of the z axis profile at each pixel in the xy plane of the reconstructed volume. The operation of extracting the xy projection image from the volume can be expressed by the following equation:","category":"page"},{"location":"","page":"Home","title":"Home","text":"mathrmxyproj(x y) = min_z left mathrmrcstvol(x y z) right","category":"page"},{"location":"","page":"Home","title":"Home","text":"Specify the hologram you want to reconstruct and the parameters, and save the projection image as xyprojection.png. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"using ParticleHolography\nusing CUDA\nusing Images\n\n# Load hologram\nimg = load_gray2float(\"holo.png\")\n\n# Parameters\nλ = 0.6328 # Wavelength [μm] \nΔx = 10.0 # Pixel size [μm]\nz0 = 220000.0 # Optical distance between the hologram and the front surface of the reconstruction volume [μm]\nΔz = 100.0 # Optical distance between the reconstructed slices [μm]\ndatlen = 1024 # Data length\nslices = 500 # Number of slices\n\n# Prepare the transfer functions\nd_sqr = cu_transfer_sqrt_arr(datlen, λ, Δx)\nd_tf = cu_transfer(z0, datlen, λ, d_sqr)\nd_slice = cu_transfer(Δz, datlen, λ, d_sqr)\n\n# Reconstruction\nd_xyproj = cu_get_reconst_xyprojection(cu(ComplexF32.(sqrt.(img))), d_tf, d_slice, slices)\n\n# Save the result\nsave(\"xyprojection.png\", Array(d_xyproj)) # Copy the d_xyproj to host memory with Array()","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: holo.png)","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: xyprojection.png)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Pages = [\"index.md\"]\nOrder = [:function]","category":"page"},{"location":"","page":"Home","title":"Home","text":"load_gray2float\ncu_transfer_sqrt_arr\ncu_transfer\ncu_get_reconst_xyprojection","category":"page"},{"location":"#ParticleHolography.load_gray2float","page":"Home","title":"ParticleHolography.load_gray2float","text":"load_gray2float(path)\n\nLoad a grayscale image from a file and return it as a Array{Float32, 2} array.\n\nArguments\n\npath::String: The path to the image file.\n\nReturns\n\nArray{Float32, 2}: The image as a Float32 array.\n\n\n\n\n\n","category":"function"},{"location":"#ParticleHolography.cu_transfer_sqrt_arr","page":"Home","title":"ParticleHolography.cu_transfer_sqrt_arr","text":"cu_transfer_sqrt_arr(datlen, wavlen, dx)\n\nCreate a CuArray of size datlen x datlen with the values of the square-root part of the transfer function.\n\nArguments\n\ndatlen::Int: The size of the CuArray.\nwavlen::AbstractFloat: The wavelength of the light.\ndx::AbstractFloat: The pixel size of the hologram.\n\nReturns\n\nCuArray{Float32,2}: The square-root part of the transfer function.\n\n\n\n\n\n","category":"function"},{"location":"#ParticleHolography.cu_transfer","page":"Home","title":"ParticleHolography.cu_transfer","text":"cu_transfer(z0, datLen, wavLen, d_sqr)\n\nCreate a CuArray of size datLen x datLen with the values of the transfer function for a given propagated distance z0. d_sqr can be obtained with cutransfersqrtarr(datlen, wavlen, dx).\n\nArguments\n\nz0::AbstractFloat: The distance to propagate the wave.\ndatLen::Int: The size of the CuArray.\nwavLen::AbstractFloat: The wavelength of the light.\nd_sqr::CuArray{Float32,2}: The square of the distance from the center of the hologram, obtained with cutransfersqrtarr(datlen, wavlen, dx).\n\nReturns\n\nCuArray{ComplexF32,2}: The transfer function for the propagation.\n\n\n\n\n\n","category":"function"},{"location":"#ParticleHolography.cu_get_reconst_xyprojection","page":"Home","title":"ParticleHolography.cu_get_reconst_xyprojection","text":"cu_get_reconst_xyprojectin(light_field, transfer_front, transfer_dz, slices)\n\nGet the XY projection of the reconstructed volume from the light field light_field using the transfer functions transfer_front and transfer_dz. transfer_front propagates the light field to the front of the volume, and transfer_dz propagates the light field between the slices. slices is the number of slices in the volume.\n\nArguments\n\nlight_field::CuArray{ComplexF32,2}: The light_field to reconstruct. In Gabor's holography, this is the square root of the hologram.\ntransfer_front::CuArray{ComplexF32,2}: The transfer function to propagate the light field to the front of the volume.\ntransfer_dz::CuArray{ComplexF32,2}: The transfer function to propagate the light field between the slices.\nslices::Int: The number of slices in the volume.\n\nReturns\n\nCuArray{Float32,2}: The XY projection of the reconstructed volume.\n\n\n\n\n\n","category":"function"},{"location":"#CPU-based-reconstruction","page":"Home","title":"CPU-based reconstruction","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Preparing...","category":"page"}]
}
