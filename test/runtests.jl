using ParticleHolography
using CUDA
using Test

@testset "ParticleHolography.jl" begin
    # Test CuTransferSqrtPart struct
    @testset "CuTransferSqrtPart struct" begin
        # Create CuTransferSqrtPart struct
        data = CUDA.fill(1.0f0, (10, 10))
        transfer_sqrt_part = ParticleHolography.CuTransferSqrtPart(data)

        # Test fields
        @test transfer_sqrt_part.data == data
    end

    # Test CuTransfer struct
    @testset "CuTransfer struct" begin
        # Create CuTransfer struct
        data = CUDA.fill(1.0f0 + 1.0f0im, (10, 10))
        transfer = ParticleHolography.CuTransfer(data)

        # Test fields
        @test transfer.data == data
    end

    # Test CuWavefront struct
    @testset "CuWavefront struct" begin
        # Create CuWavefront struct
        data = CUDA.fill(1.0f0 + 1.0f0im, (10, 10))
        wavefront = ParticleHolography.CuWavefront(data)

        # Test fields
        @test wavefront.data == data
    end

    # Test load_gray2float function
    @testset "load_gray2float function" begin
        # Test load_gray2float function
        img = ParticleHolography.load_gray2float("./data/holo1.bmp")

        # Test return type
        @test typeof(img) == Array{Float32, 2}

        # Test return size
        @test size(img) == (1024,1024)
    end

    # Test find_external_contours function
    @testset "find_external_contours function" begin
        # Test find_external_contours function
        img = ParticleHolography.load_gray2float("./data/binaryparticles.png")
        contours = ParticleHolography.find_external_contours(img)

        # Test return type
        @test typeof(contours) == Vector{Vector{CartesianIndex}}

        # Test return length
        @test length(contours) == 87
    end

    # Test draw_contours! function
    @testset "draw_contours! function" begin
        # Test draw_contours! function
        img = ParticleHolography.load_gray2float("./data/binaryparticles.png")
        contours = ParticleHolography.find_external_contours(img)
        img = zeros(size(img))
        ParticleHolography.draw_contours!(img, 1.0, contours)

        # Test output
        @test length(findall(img .== 1.0)) == 6247
    end

    # Test make_background function
    @testset "make_background function" begin
        # Test make_background function
        pathlist = ["./data/holo1.bmp", "./data/holo2.bmp"]
        background = ParticleHolography.make_background(pathlist; mode=:mean)

        # Test return type
        @test typeof(background) == Array{Float64, 2}

        # Test return size
        @test size(background) == (1024,1024)
    end

    # Test cu_connected_component_labeling function
    @testset "cu_connected_component_labeling function" begin
        # Test cu_connected_component_labeling function
        img = ParticleHolography.load_gray2float("./data/binaryparticles.png")
        output_img = ParticleHolography.cu_connected_component_labeling(cu(1.0f0 .-img))

        # Test return type
        @test typeof(output_img) <: CuArray{UInt32, 2}

        # Test return size
        @test size(output_img) == (1024,1024)
    end

    # Test count_labels function
    @testset "count_labels function" begin
        # Test count_labels function
        img = ParticleHolography.load_gray2float("./data/binaryparticles.png")
        output_img = ParticleHolography.cu_connected_component_labeling(cu(1.0f0 .-img))

        # Test output
        @test ParticleHolography.count_labels(Array(output_img)) == 84
    end

    # Test cu_find_valid_labels function
    @testset "cu_find_valid_labels function" begin
        # Test cu_find_valid_labels function
        img = ParticleHolography.load_gray2float("./data/binaryparticles.png")
        output_img = ParticleHolography.cu_connected_component_labeling(cu(1.0f0 .-img))
        valid_labels = ParticleHolography.cu_find_valid_labels(output_img)

        # Test return type
        @test typeof(valid_labels) == Vector{Int64}

        # Test return length
        @test length(valid_labels) == 84
    end

    # Test get_bounding_rectangles function
    @testset "get_bounding_rectangles function" begin
        # Test get_bounding_rectangles function
        img = ParticleHolography.load_gray2float("./data/binaryparticles.png")
        output_img = ParticleHolography.cu_connected_component_labeling(cu(1.0f0 .-img))
        valid_labels = ParticleHolography.cu_find_valid_labels(output_img)
        bounding_rectangles = ParticleHolography.get_bounding_rectangles(Array(output_img), valid_labels)

        # Test return type
        @test typeof(bounding_rectangles) == Array{Any,1}

        # Test return length
        @test length(bounding_rectangles) == 84
    end

    # Test get_distortion_coefficients function
    @testset "get_distortion_coefficients function" begin
        # Test get_distortion_coefficients function
        img1 = ParticleHolography.load_gray2float("./data/impcam1_enhanced.png")
        img2 = ParticleHolography.load_gray2float("./data/impcam2_enhanced.png")
        coeffs = ParticleHolography.get_distortion_coefficients(img1, img2, verbose=true)

        # Test return type
        @test typeof(coeffs) == Array{Float64,1}

        # Test return length
        @test length(coeffs) == 12

        # Test output
        @test isapprox(coeffs[1], 1.23, atol=1e-2)
    end

    # Test quadratic_distortion_correction function
    @testset "quadratic_distortion_correction function" begin
        # Test quadratic_distortion_correction function
        img1 = ParticleHolography.load_gray2float("./data/impcam1_enhanced.png")
        img2 = ParticleHolography.load_gray2float("./data/impcam2_enhanced.png")
        coeffs = ParticleHolography.get_distortion_coefficients(img1, img2, verbose=false)
        corrected_img = ParticleHolography.quadratic_distortion_correction(img2, coeffs)

        # Test return type
        @test typeof(corrected_img) == Array{Float64,2}

        # Test return size
        @test size(corrected_img) == (1024,1024)
    end

    # Test cu_transfer_sqrt_arr function
    @testset "cu_transfer_sqrt_arr function" begin
        # Test cu_transfer_sqrt_arr function
        transfer_sqrt_arr = ParticleHolography.cu_transfer_sqrt_arr(1024, 0.6328, 10.0)

        # Test return type
        @test typeof(transfer_sqrt_arr) == ParticleHolography.CuTransferSqrtPart{Float32}

        # Test return size
        @test size(transfer_sqrt_arr.data) == (1024,1024)
    end

    # Test cu_transfer function
    @testset "cu_transfer function" begin
        # Test cu_transfer function
        transfer_sqrt_arr = ParticleHolography.cu_transfer_sqrt_arr(1024, 0.6328, 10.0)
        transfer = ParticleHolography.cu_transfer(80000.0, 1024, 0.6328, transfer_sqrt_arr)

        # Test return type
        @test typeof(transfer) == ParticleHolography.CuTransfer{ComplexF32}

        # Test return size
        @test size(transfer.data) == (1024,1024)
    end

    # Test cu_gabor_wavefront function
    @testset "cu_gabor_wavefront function" begin
        # Test cu_gabor_wavefront function
        holo = CUDA.rand(Float32, (1024,1024))
        wavefront = ParticleHolography.cu_gabor_wavefront(holo)

        # Test return type
        @test typeof(wavefront) == ParticleHolography.CuWavefront{ComplexF32}

        # Test return size
        @test size(wavefront.data) == (1024,1024)
    end

    # Test cu_phase_retrieval_holo function
    @testset "cu_phase_retrieval_holo function" begin
        # Test cu_phase_retrieval_holo function
        holo1 = ParticleHolography.load_gray2float("./data/holo1.bmp")
        holo2 = ParticleHolography.load_gray2float("./data/holo2_corrected.png")
        transsqr = ParticleHolography.cu_transfer_sqrt_arr(1024, 0.6328, 10.0)
        transfer = ParticleHolography.cu_transfer(80000.0, 1024, 0.6328, transsqr)
        transferinv = ParticleHolography.cu_transfer(-80000.0, 1024, 0.6328, transsqr)
        d_pr_wf = ParticleHolography.cu_phase_retrieval_holo(cu(holo1), cu(holo2), transfer, transferinv, 20, 1024)

        # Test return type
        @test typeof(d_pr_wf) == ParticleHolography.CuWavefront{ComplexF32}

        # Test return size
        @test size(d_pr_wf.data) == (1024,1024)
    end

    # Test reconstruction functions
    @testset "reconstruction functions" begin
        # Test reconstruction functions
        holo = ParticleHolography.load_gray2float("./data/holo1.bmp")
        wf = ParticleHolography.cu_gabor_wavefront(cu(holo))
        transsqr = ParticleHolography.cu_transfer_sqrt_arr(1024, 0.6328, 10.0)
        transfer = ParticleHolography.cu_transfer(-80000.0, 1024, 0.6328, transsqr)
        transferslice = ParticleHolography.cu_transfer(-100.0, 1024, 0.6328, transsqr)

        @test ParticleHolography.cu_get_reconst_vol(wf, transfer, transferslice, 10) !== nothing
        @test ParticleHolography.cu_get_reconst_xyprojection(wf, transfer, transferslice, 10) !== nothing
        @test ParticleHolography.cu_get_reconst_vol_and_xyprojection(wf, transfer, transferslice, 10) !== nothing
    end
end
