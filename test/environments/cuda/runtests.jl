using ParticleHolography
using CUDA
using Test

CUDA.functional() || error("CUDA is not functional on the selected self-hosted runner.")
include(joinpath(@__DIR__, "..", "..", "shared", "backend_contract.jl"))

@testset "CUDA backend contract" begin
    @test :cuda in available_backends()
    array = CUDA.zeros(Float32, 2, 2)
    @test backendof(array) isa ParticleHolography.CUDABackend
    @test backendof(@view array[:, 1:1]) isa ParticleHolography.CUDABackend
    run_backend_contract(backend(:cuda))
end

@testset "CUDA integration and v0.2 wrappers" begin
    n = 16
    wavelength = 0.6328
    pixel_pitch = 10.0
    sqrt_part = cu_transfer_sqrt_arr(n, wavelength, pixel_pitch)
    front = cu_transfer(-800.0, n, wavelength, sqrt_part)
    step = cu_transfer(-20.0, n, wavelength, sqrt_part)
    hologram = reshape(Float32.(range(0.1, 0.9; length=n^2)), n, n)
    wavefront = cu_gabor_wavefront(CUDA.CuArray(hologram))
    volume = cu_get_reconst_vol(wavefront, front, step, 2)
    @test volume isa CUDA.CuArray
    @test size(volume) == (n, n, 2)
    @test all(isfinite, Array(volume))

    binary = CUDA.zeros(Bool, 8, 8)
    binary[2:5, 2:5] .= true
    labels = cu_connected_component_labeling(binary)
    @test labels isa CUDA.CuArray{UInt32,2}
    @test count_labels(labels) == 1

    image = rand(Float32, 16, 16)
    cpu_vectors = piv_map(CPUBackend(), image, image; grid_size=8,
                          interrogation_size=8, search_size=16)
    cuda_vectors = piv_map(backend(:cuda), image, image; grid_size=8,
                           interrogation_size=8, search_size=16)
    @test cuda_vectors ≈ cpu_vectors atol=2f-4 rtol=2f-4
    cuda_device_vectors = piv_map(backend(:cuda), CUDA.CuArray(image),
                                  CUDA.CuArray(image); grid_size=8,
                                  interrogation_size=8, search_size=16)
    @test cuda_device_vectors ≈ cpu_vectors atol=2f-4 rtol=2f-4
end
