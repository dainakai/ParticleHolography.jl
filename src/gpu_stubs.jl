export cu_transfer_sqrt_arr, cu_transfer, cu_gabor_wavefront, cu_phase_retrieval_holo
export cu_get_reconst_vol, cu_get_reconst_xyprojection, cu_get_reconst_vol_and_xyprojection, cu_get_reconst_complex_vol
export cu_asm_prop!, cu_2d_pad, cu_get_reconst_vol_and_xyprojection_padded
export cu_rectangle_filter, cu_super_gaussian_filter, cu_apply_low_pass_filter, cu_apply_low_pass_filter!
export cu_connected_component_labeling, cu_find_valid_labels, cu_dilate, cu_make_background_mode
export particle_bounding_boxes, particle_bounding_boxes_3d

const _cuda_functional = Ref{Function}(() -> false)
const _cuda_status_message = Ref{Function}(() -> "CUDA has not been loaded in this Julia session.")

function _cuda_available()
    try
        return Bool(_cuda_functional[]())
    catch err
        _cuda_status_message[] = () -> "CUDA.functional() failed with $(typeof(err)): $(sprint(showerror, err))."
        return false
    end
end

function _cuda_required_message(func::Symbol)
    return "ParticleHolography.$func requires CUDA. Load CUDA with `using CUDA` and ensure `CUDA.functional()` returns true before calling this API. $(_cuda_status_message[]())"
end

function _assert_cuda_functional(func::Symbol)
    _cuda_available() || throw(ArgumentError(_cuda_required_message(func)))
    return nothing
end

function _cuda_required_error(func::Symbol)
    _assert_cuda_functional(func)
    throw(ArgumentError("ParticleHolography.$func requires a CUDA extension method for these argument types. Check that the inputs are CUDA.CuArray-backed values and that CUDA loaded successfully."))
end

function _makie_required_error()
    throw(ArgumentError("verbose bundle-adjustment diagnostics require Makie and CairoMakie. Load them with `using Makie, CairoMakie` before calling `get_distortion_coefficients(...; verbose=true)`."))
end

function _save_bundle_adjustment_diagnostics(args...; kwargs...)
    _makie_required_error()
end

"""
    cu_transfer_sqrt_arr(datlen, wavlen, dx)

Create the square-root part of the angular-spectrum transfer function on a CUDA device.
This API requires `using CUDA` and a functional CUDA runtime.
"""
cu_transfer_sqrt_arr(args...; kwargs...) = _cuda_required_error(:cu_transfer_sqrt_arr)

"""
    cu_transfer(z0, datlen, wavlen, d_sqr)

Create a CUDA transfer function for propagation distance `z0`.
This API requires `using CUDA` and a functional CUDA runtime.
"""
cu_transfer(args...; kwargs...) = _cuda_required_error(:cu_transfer)

"""
    cu_gabor_wavefront(holo)

Create a complex CUDA wavefront from a hologram intensity image.
This API requires `using CUDA` and a functional CUDA runtime.
"""
cu_gabor_wavefront(args...; kwargs...) = _cuda_required_error(:cu_gabor_wavefront)

"""
    cu_phase_retrieval_holo(holo1, holo2, transfer, invtransfer, priter, datlen)

Run Gerchberg-Saxton phase retrieval for two CUDA-resident holograms.
This API requires `using CUDA` and a functional CUDA runtime.
"""
cu_phase_retrieval_holo(args...; kwargs...) = _cuda_required_error(:cu_phase_retrieval_holo)

"""
    cu_get_reconst_vol(wavefront, transfer_front, transfer_dz, slices, [return_type])

Reconstruct a CUDA intensity volume from a wavefront and propagation transfer functions.
This API requires `using CUDA` and a functional CUDA runtime.
"""
cu_get_reconst_vol(args...; kwargs...) = _cuda_required_error(:cu_get_reconst_vol)

"""
    cu_get_reconst_complex_vol(wavefront, transfer_front, transfer_dz, slices)

Reconstruct a CUDA complex-amplitude volume from a wavefront and propagation transfer functions.
This API requires `using CUDA` and a functional CUDA runtime.
"""
cu_get_reconst_complex_vol(args...; kwargs...) = _cuda_required_error(:cu_get_reconst_complex_vol)

"""
    cu_get_reconst_xyprojection(wavefront, transfer_front, transfer_dz, slices)

Reconstruct the optical-axis minimum-intensity XY projection on a CUDA device.
This API requires `using CUDA` and a functional CUDA runtime.
"""
cu_get_reconst_xyprojection(args...; kwargs...) = _cuda_required_error(:cu_get_reconst_xyprojection)

"""
    cu_get_reconst_vol_and_xyprojection(wavefront, transfer_front, transfer_dz, slices, [return_type])

Reconstruct a CUDA intensity volume and its XY projection in one pass.
This API requires `using CUDA` and a functional CUDA runtime.
"""
cu_get_reconst_vol_and_xyprojection(args...; kwargs...) = _cuda_required_error(:cu_get_reconst_vol_and_xyprojection)

"""
    cu_get_reconst_vol_and_xyprojection_padded(wavefront, transfer_front, transfer_dz, slices, [return_type])

Reconstruct a padded CUDA intensity volume and its XY projection.
This API requires `using CUDA` and a functional CUDA runtime.
"""
cu_get_reconst_vol_and_xyprojection_padded(args...; kwargs...) = _cuda_required_error(:cu_get_reconst_vol_and_xyprojection_padded)

"""
    cu_asm_prop!(out, input, d_sqr, z0, datlen, wavlen)

Propagate a CUDA wavefront in-place using the angular spectrum method.
This API requires `using CUDA` and a functional CUDA runtime.
"""
cu_asm_prop!(args...; kwargs...) = _cuda_required_error(:cu_asm_prop!)

"""
    cu_2d_pad(inarr)

Pad a CUDA 2D complex array with its mean value.
This API requires `using CUDA` and a functional CUDA runtime.
"""
cu_2d_pad(args...; kwargs...) = _cuda_required_error(:cu_2d_pad)

"""
    cu_rectangle_filter(prop_dist, wavlen, imglen, pixel_pitch)

Create a rectangular CUDA low-pass filter.
This API requires `using CUDA` and a functional CUDA runtime.
"""
cu_rectangle_filter(args...; kwargs...) = _cuda_required_error(:cu_rectangle_filter)

"""
    cu_super_gaussian_filter(prop_dist, wavlen, imglen, pixel_pitch)

Create a super-Gaussian CUDA low-pass filter.
This API requires `using CUDA` and a functional CUDA runtime.
"""
cu_super_gaussian_filter(args...; kwargs...) = _cuda_required_error(:cu_super_gaussian_filter)

"""
    cu_apply_low_pass_filter(holo, lpf)

Return a CUDA wavefront after applying a low-pass filter in Fourier space.
This API requires `using CUDA` and a functional CUDA runtime.
"""
cu_apply_low_pass_filter(args...; kwargs...) = _cuda_required_error(:cu_apply_low_pass_filter)

"""
    cu_apply_low_pass_filter!(holo, lpf)

Apply a low-pass filter to a CUDA wavefront in-place.
This API requires `using CUDA` and a functional CUDA runtime.
"""
cu_apply_low_pass_filter!(args...; kwargs...) = _cuda_required_error(:cu_apply_low_pass_filter!)

"""
    cu_connected_component_labeling(binary_img)

Label connected components in a CUDA binary image.
This API requires `using CUDA` and a functional CUDA runtime.
"""
cu_connected_component_labeling(args...; kwargs...) = _cuda_required_error(:cu_connected_component_labeling)

"""
    cu_find_valid_labels(labeled_img)

Find non-background labels in a CUDA connected-component label image.
This API requires `using CUDA` and a functional CUDA runtime.
"""
cu_find_valid_labels(args...; kwargs...) = _cuda_required_error(:cu_find_valid_labels)

"""
    cu_dilate(vol; blocksize=32)

Dilate a CUDA 3D binary volume in the XY neighborhood of each z-slice.
This API requires `using CUDA` and a functional CUDA runtime.
"""
cu_dilate(args...; kwargs...) = _cuda_required_error(:cu_dilate)

"""
    cu_make_background_mode(grayimglist)

Compute a per-pixel mode background image with CUDA acceleration.
This API requires `using CUDA` and a functional CUDA runtime.
"""
cu_make_background_mode(args...; kwargs...) = _cuda_required_error(:cu_make_background_mode)

particle_bounding_boxes(args...; kwargs...) = _cuda_required_error(:particle_bounding_boxes)
particle_bounding_boxes_3d(args...; kwargs...) = _cuda_required_error(:particle_bounding_boxes_3d)

"""
    getPIVMap_GPU(image1, image2, imgLen=1024, gridSize=128, intrSize=128, srchSize=256)

Compute a PIV displacement map with CUDA acceleration for bundle adjustment.
This API requires `using CUDA` and a functional CUDA runtime.
"""
getPIVMap_GPU(args...; kwargs...) = _cuda_required_error(:getPIVMap_GPU)
