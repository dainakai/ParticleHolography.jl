module ParticleHolographyCUDAExt

using ParticleHolography
using CUDA
using Statistics

const CuLowPassFilter = ParticleHolography.CuLowPassFilter
const CuTransfer = ParticleHolography.CuTransfer
const CuTransferSqrtPart = ParticleHolography.CuTransferSqrtPart
const CuWavefront = ParticleHolography.CuWavefront
const _assert_cuda_functional = ParticleHolography._assert_cuda_functional
const _save_bundle_adjustment_diagnostics = ParticleHolography._save_bundle_adjustment_diagnostics
const cu_2d_pad = ParticleHolography.cu_2d_pad
const cu_apply_low_pass_filter = ParticleHolography.cu_apply_low_pass_filter
const cu_apply_low_pass_filter! = ParticleHolography.cu_apply_low_pass_filter!
const cu_asm_prop! = ParticleHolography.cu_asm_prop!
const cu_connected_component_labeling = ParticleHolography.cu_connected_component_labeling
const cu_dilate = ParticleHolography.cu_dilate
const cu_find_valid_labels = ParticleHolography.cu_find_valid_labels
const cu_gabor_wavefront = ParticleHolography.cu_gabor_wavefront
const cu_get_reconst_complex_vol = ParticleHolography.cu_get_reconst_complex_vol
const cu_get_reconst_vol = ParticleHolography.cu_get_reconst_vol
const cu_get_reconst_vol_and_xyprojection = ParticleHolography.cu_get_reconst_vol_and_xyprojection
const cu_get_reconst_vol_and_xyprojection_padded = ParticleHolography.cu_get_reconst_vol_and_xyprojection_padded
const cu_get_reconst_xyprojection = ParticleHolography.cu_get_reconst_xyprojection
const cu_make_background_mode = ParticleHolography.cu_make_background_mode
const cu_phase_retrieval_holo = ParticleHolography.cu_phase_retrieval_holo
const cu_rectangle_filter = ParticleHolography.cu_rectangle_filter
const cu_super_gaussian_filter = ParticleHolography.cu_super_gaussian_filter
const cu_transfer = ParticleHolography.cu_transfer
const cu_transfer_sqrt_arr = ParticleHolography.cu_transfer_sqrt_arr
const finalize_particle_neighborhoods! = ParticleHolography.finalize_particle_neighborhoods!
const gen_particle_neighborhoods = ParticleHolography.gen_particle_neighborhoods
const get_bounding_rectangles = ParticleHolography.get_bounding_rectangles
const get_distortion_coefficients = ParticleHolography.get_distortion_coefficients
const getErrorVec = ParticleHolography.getErrorVec
const getPIVMap_GPU = ParticleHolography.getPIVMap_GPU
const getYacobian = ParticleHolography.getYacobian
const modified_Cholesky_decomposition = ParticleHolography.modified_Cholesky_decomposition
const particle_bounding_boxes = ParticleHolography.particle_bounding_boxes
const particle_bounding_boxes_3d = ParticleHolography.particle_bounding_boxes_3d
const simultanious_equation_solver = ParticleHolography.simultanious_equation_solver
const update_particle_neighborhoods! = ParticleHolography.update_particle_neighborhoods!
const update_particle_neighborhoods3d! = ParticleHolography.update_particle_neighborhoods3d!

const _ROOT_DIR = dirname(@__DIR__)
const _SRC_DIR = joinpath(_ROOT_DIR, "src")
const _EXT_DIR = @__DIR__

function __init__()
    ParticleHolography._cuda_functional[] = () -> begin
        try
            CUDA.functional()
        catch
            false
        end
    end
    ParticleHolography._cuda_status_message[] = () -> begin
        try
            CUDA.functional() ? "CUDA.functional() returned true." : "CUDA.functional() returned false. Check the CUDA driver/runtime and GPU availability."
        catch err
            "CUDA.functional() failed with $(typeof(err)): $(sprint(showerror, err))."
        end
    end
    return nothing
end

include(joinpath(_SRC_DIR, "holofunc.jl"))
include(joinpath(_SRC_DIR, "frequency_filters.jl"))
include(joinpath(_EXT_DIR, "cuda_utils.jl"))
include(joinpath(_EXT_DIR, "cuda_ccl.jl"))
include(joinpath(_EXT_DIR, "cuda_particle_detection.jl"))
include(joinpath(_EXT_DIR, "cuda_bundleadjustment.jl"))

end
