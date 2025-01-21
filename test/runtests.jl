using ParticleHolography
using CUDA
using Test
using Glob
using Plots

@testset "ParticleHolography.jl" begin
    # types.jl --------------------------------------------------------------
    @testset "CuTransferSqrtPart struct" begin
        data = CUDA.fill(1.0f0, (10, 10))
        transfer_sqrt_part = CuTransferSqrtPart(data)
    
        # Test fields
        @test transfer_sqrt_part.data == data
    
        # Test size, axes, ndims
        @test size(transfer_sqrt_part) == (10, 10)
        @test axes(transfer_sqrt_part) == axes(data)
        @test ndims(transfer_sqrt_part) == 2
    end
    
    @testset "CuTransfer struct" begin
        data = CUDA.fill(1.0f0 + 1.0f0im, (10, 10))
        transfer = CuTransfer(data)
    
        # Test fields
        @test transfer.data == data
    
        # Test size, axes, ndims
        @test size(transfer) == (10, 10)
        @test axes(transfer) == axes(data)
        @test ndims(transfer) == 2
    end
    
    @testset "CuWavefront struct" begin
        data = CUDA.fill(1.0f0 + 1.0f0im, (10, 10))
        wavefront = CuWavefront(data)
    
        # Test fields
        @test wavefront.data == data
    
        # Test size, axes, ndims
        @test size(wavefront) == (10, 10)
        @test axes(wavefront) == axes(data)
        @test ndims(wavefront) == 2
    end
    
    @testset "CuLowPassFilter struct" begin
        data = CUDA.fill(1.0f0, (10, 10))
        low_pass_filter = CuLowPassFilter(data)
    
        # Test fields
        @test low_pass_filter.data == data
    
        # Test size, axes, ndims
        @test size(low_pass_filter) == (10, 10)
        @test axes(low_pass_filter) == axes(data)
        @test ndims(low_pass_filter) == 2
    end

    # utils.jl --------------------------------------------------------------
    # Test load_gray2float function
    @testset "load_gray2float function" begin
        # Test load_gray2float function
        img = load_gray2float("./data/holo1.bmp")

        # Test return type
        @test typeof(img) == Array{Float32, 2}

        # Test return size
        @test size(img) == (1024,1024)
    end

    # Test find_external_contours function
    @testset "find_external_contours function" begin
        # Test find_external_contours function
        img = load_gray2float("./data/binaryparticles.png")
        contours = find_external_contours(img)

        # Test return type
        @test typeof(contours) == Vector{Vector{CartesianIndex}}

        # Test return length
        @test length(contours) == 87
    end

    # Test draw_contours! function
    @testset "draw_contours! function" begin
        # Test draw_contours! function
        img = load_gray2float("./data/binaryparticles.png")
        contours = find_external_contours(img)
        img = zeros(size(img))
        draw_contours!(img, 1.0, contours)

        # Test output
        @test length(findall(img .== 1.0)) == 6247
    end

    # Test make_background function
    @testset "make_background function" begin
        # Test make_background function
        pathlist = ["./data/holo1.bmp", "./data/holo2.bmp"]
        background = make_background(pathlist; mode=:mean)
        background2 = make_background(pathlist; mode=:mode)

        # Test return type
        @test typeof(background) == Array{Float64, 2}
        @test typeof(background2) == Array{Float64, 2}

        # Test return size
        @test size(background) == (1024,1024)
        @test size(background2) == (1024,1024)
    end

    # Test pad_with_mean function
    @testset "pad_with_mean function" begin
        # Test pad_with_mean function
        img = load_gray2float("./data/holo1.bmp")
        output_img = pad_with_mean(img, 2048)

        # Test return type
        @test typeof(output_img) == Array{Float32, 2}

        # Test return size
        @test size(output_img) == (2048,2048)
    end

    # Test dictsave and dictload function
    @testset "dictsave and dictload functions" begin
        λ = 0.6328 # Wavelength [μm]
        Δx = 10.0 # Pixel size [μm]
        z0 = 80000.0 # Optical distance between the hologram and the front surface of the reconstruction volume [μm]
        Δz = 100.0 # Optical distance between the reconstructed slices [μm]
        datlen = 1024 # Data length
        slices = 1000 # Number of slices
        pr_dist = 80000.0 # Optical distance between the two holograms [μm]
        pr_iter = 9
        threshold = 10/255

        img1 = load_gray2float("./data/holo1.bmp")
        img2 = load_gray2float("./data/holo2.bmp")

        d_sqr = cu_transfer_sqrt_arr(datlen, λ, Δx)
        d_tf = cu_transfer(-z0, datlen, λ, d_sqr)
        d_slice = cu_transfer(-Δz, datlen, λ, d_sqr)
        d_pr = cu_transfer(pr_dist, datlen, λ, d_sqr)
        d_pr_inv = cu_transfer(-pr_dist, datlen, λ, d_sqr)

        # Phase retrieval using Gerchberg-Saxton algorithm
        d_holo = cu_phase_retrieval_holo(cu(img1), cu(img2), d_pr, d_pr_inv, pr_iter, datlen)

        # Reconstruction
        d_vol = cu_get_reconst_vol(d_holo, d_tf, d_slice, slices)

        # Binarization
        d_bin_vol = d_vol .<= threshold

        particle_bbs = particle_bounding_boxes(d_bin_vol)
        particle_coords = particle_coordinates(particle_bbs, d_vol)

        # Save the particle coordinates
        dictsave("./data/particles.json", particle_coords)
        d_vol = nothing
        d_bin_vol = nothing

        loadeddict = dictload("./data/particles.json")

        @test particle_coords == loadeddict
    end

    # ccl.jl --------------------------------------------------------------
    # Test cu_connected_component_labeling function
    @testset "cu_connected_component_labeling function" begin
        # Test cu_connected_component_labeling function
        img = load_gray2float("./data/binaryparticles.png")
        output_img = cu_connected_component_labeling(cu(1.0f0 .-img))

        # Test return type
        @test typeof(output_img) <: CuArray{UInt32, 2}

        # Test return size
        @test size(output_img) == (1024,1024)
    end

    # Test count_labels function
    @testset "count_labels function" begin
        # Test count_labels function
        img = load_gray2float("./data/binaryparticles.png")
        output_img = cu_connected_component_labeling(cu(1.0f0 .-img))

        # Test output
        @test count_labels(Array(output_img)) == 84
    end

    # Test cu_find_valid_labels function
    @testset "cu_find_valid_labels function" begin
        # Test cu_find_valid_labels function
        img = load_gray2float("./data/binaryparticles.png")
        output_img = cu_connected_component_labeling(cu(1.0f0 .-img))
        valid_labels = cu_find_valid_labels(output_img)

        # Test return type
        @test typeof(valid_labels) == Vector{Int64}

        # Test return length
        @test length(valid_labels) == 84
    end

    # bundleadjustment.jl --------------------------------------------------------------
    # Test get_distortion_coefficients function
    @testset "get_distortion_coefficients function" begin
        # Test get_distortion_coefficients function
        img1 = load_gray2float("./data/impcam1_enhanced.png")
        img2 = load_gray2float("./data/impcam2_enhanced.png")
        coeffs = get_distortion_coefficients(img1, img2, verbose=true, save_dir="./data")

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
        img1 = load_gray2float("./data/impcam1_enhanced.png")
        img2 = load_gray2float("./data/impcam2_enhanced.png")
        coeffs = get_distortion_coefficients(img1, img2, verbose=false)
        corrected_img = quadratic_distortion_correction(img2, coeffs)

        # Test return type
        @test typeof(corrected_img) == Array{Float64,2}

        # Test return size
        @test size(corrected_img) == (1024,1024)
    end

    # holofunc.jl --------------------------------------------------------------
    # Test cu_transfer_sqrt_arr function
    @testset "cu_transfer_sqrt_arr function" begin
        # Test cu_transfer_sqrt_arr function
        transfer_sqrt_arr = cu_transfer_sqrt_arr(1024, 0.6328, 10.0)

        # Test return type
        @test typeof(transfer_sqrt_arr) == ParticleHolography.CuTransferSqrtPart{Float32}

        # Test return size
        @test size(transfer_sqrt_arr.data) == (1024,1024)
    end

    # Test cu_transfer function
    @testset "cu_transfer function" begin
        # Test cu_transfer function
        transfer_sqrt_arr = cu_transfer_sqrt_arr(1024, 0.6328, 10.0)
        transfer = cu_transfer(80000.0, 1024, 0.6328, transfer_sqrt_arr)

        # Test return type
        @test typeof(transfer) == ParticleHolography.CuTransfer{ComplexF32}

        # Test return size
        @test size(transfer.data) == (1024,1024)
    end

    # Test cu_gabor_wavefront function
    @testset "cu_gabor_wavefront function" begin
        # Test cu_gabor_wavefront function
        holo = CUDA.rand(Float32, (1024,1024))
        wavefront = cu_gabor_wavefront(holo)

        # Test return type
        @test typeof(wavefront) == ParticleHolography.CuWavefront{ComplexF32}

        # Test return size
        @test size(wavefront.data) == (1024,1024)
    end

    # Test cu_phase_retrieval_holo function
    @testset "cu_phase_retrieval_holo function" begin
        # Test cu_phase_retrieval_holo function
        holo1 = load_gray2float("./data/holo1.bmp")
        holo2 = load_gray2float("./data/holo2_corrected.png")
        transsqr = cu_transfer_sqrt_arr(1024, 0.6328, 10.0)
        transfer = cu_transfer(80000.0, 1024, 0.6328, transsqr)
        transferinv = cu_transfer(-80000.0, 1024, 0.6328, transsqr)
        d_pr_wf = cu_phase_retrieval_holo(cu(holo1), cu(holo2), transfer, transferinv, 20, 1024)

        # Test return type
        @test typeof(d_pr_wf) == ParticleHolography.CuWavefront{ComplexF32}

        # Test return size
        @test size(d_pr_wf.data) == (1024,1024)
    end

    # Test reconstruction functions
    @testset "reconstruction functions" begin
        # Test reconstruction functions
        holo = load_gray2float("./data/holo1.bmp")
        wf = cu_gabor_wavefront(cu(holo))
        transsqr = cu_transfer_sqrt_arr(1024, 0.6328, 10.0)
        transfer = cu_transfer(-80000.0, 1024, 0.6328, transsqr)
        transferslice = cu_transfer(-100.0, 1024, 0.6328, transsqr)

        @test cu_get_reconst_vol(wf, transfer, transferslice, 10) !== nothing
        @test cu_get_reconst_complex_vol(wf, transfer, transferslice, 10) !== nothing
        @test cu_get_reconst_xyprojection(wf, transfer, transferslice, 10) !== nothing
        @test cu_get_reconst_vol_and_xyprojection(wf, transfer, transferslice, 10) !== nothing
    end

    # particle_detection.jl --------------------------------------------------------------
    @testset "particle detection functions" begin
        λ = 0.6328 # Wavelength [μm]
        Δx = 10.0 # Pixel size [μm]
        z0 = 80000.0 # Optical distance between the hologram and the front surface of the reconstruction volume [μm]
        Δz = 100.0 # Optical distance between the reconstructed slices [μm]
        datlen = 1024 # Data length
        slices = 1000 # Number of slices
        pr_dist = 80000.0 # Optical distance between the two holograms [μm]
        pr_iter = 9
        threshold = 30/255

        img1 = load_gray2float("./data/holo1.bmp")
        img2 = load_gray2float("./data/holo2.bmp")

        d_sqr = cu_transfer_sqrt_arr(datlen, λ, Δx)
        d_tf = cu_transfer(-z0, datlen, λ, d_sqr)
        d_slice = cu_transfer(-Δz, datlen, λ, d_sqr)
        d_pr = cu_transfer(pr_dist, datlen, λ, d_sqr)
        d_pr_inv = cu_transfer(-pr_dist, datlen, λ, d_sqr)

        # Phase retrieval using Gerchberg-Saxton algorithm
        d_holo = cu_phase_retrieval_holo(cu(img1), cu(img2), d_pr, d_pr_inv, pr_iter, datlen)

        # Reconstruction
        d_vol = cu_get_reconst_vol(d_holo, d_tf, d_slice, slices)

        # Low pass filtering and Reconstruction
        d_lpf = cu_super_gaussian_filter(pr_dist, λ, datlen, Δx)
        cu_apply_low_pass_filter!(d_holo, d_lpf)

        d_lpf_vol = cu_get_reconst_vol(d_holo, d_tf, d_slice, slices)

        # Binarization
        d_bin_vol = d_vol .<= threshold

        particle_bbs = particle_bounding_boxes(d_bin_vol)
        @test particle_bbs !== nothing
        @test particle_coordinates(particle_bbs, d_vol) !== nothing
        @test particle_coor_diams(particle_bbs, d_vol, d_lpf_vol) !== nothing

        d_vol = nothing
        d_lpf_vol = nothing
        d_bin_vol = nothing
    end

    # particle_tracking.jl --------------------------------------------------------------
    @testset "particle tracking functions" begin
        # Load the particle coordinates
        files = glob("./data/dicts/*.json")

        # Convert to dictionary with UUID keys and Float64 values
        dicts = dictload.(files)

        graphs = [Labonte(dict1, dict2) for (dict1, dict2) in zip(dicts[1:end-1], dicts[2:end])]
        @test graphs[1] !== nothing

        paths = enum_edge(graphs[1])
        @test paths !== nothing

        for graph in graphs[2:end]
            append_path!(paths, graph)
        end

        @test paths !== nothing

        @test gen_fulldict(dicts) == gen_fulldict(files)
    end

    # plot_recipes.jl --------------------------------------------------------------
    @testset "ParticlePlot" begin
        # Load the particle coordinates
        files = glob("./data/dicts/*.json")

        colors = cgrad(:viridis)[LinRange(0, 1, length(files))]
        plot()
        for (idx, file) in enumerate(files)
            data = dictload(file)
            particleplot!(data, legend = false, scaling=(10.0, 10.0, -100.0), shift=(0.0, 0.0, 1e5), color=colors[idx], xlabel="x [µm]", ylabel="z [µm]", zlabel="y [µm]", xlim=(0,10240), ylim=(0,1e5), zlim=(0,10240))
        end
        
        @test savefig("./data/particle_trajectories.png") !== nothing
    end

    @testset "TrajectoryPlot" begin
        # Load the particle coordinates
        files = glob("./data/dicts/*.json")[1:5]

        # Convert to dictionary with UUID keys and Float64 values
        dicts = dictload.(files)

        graphs = [Labonte(dict1, dict2) for (dict1, dict2) in zip(dicts[1:end-1], dicts[2:end])]

        paths = enum_edge(graphs[1])

        for graph in graphs[2:end]
            append_path!(paths, graph)
        end

        fulldict = gen_fulldict(dicts)

        trajectoryplot(paths, fulldict)

        @test savefig("./data/trajectory_plot.png") !== nothing
    end

    @testset "Low Pass Filter Tests" begin
        @testset "cu_rectangle_filter" begin
            prop_dist = 0.01
            wavlen    = 5.3e-7
            imglen    = 128
            pixel_pitch = 6.5e-6
    
            lpf_rect = cu_rectangle_filter(prop_dist, wavlen, imglen, pixel_pitch)
            @test isa(lpf_rect, CuLowPassFilter)
            @test size(lpf_rect.data) == (imglen, imglen)
        end

        @testset "cu_super_gaussian_filter" begin
            prop_dist = 0.01
            wavlen    = 5.3e-7
            imglen    = 128
            pixel_pitch = 6.5e-6
    
            lpf_super = cu_super_gaussian_filter(prop_dist, wavlen, imglen, pixel_pitch)
            @test isa(lpf_super, CuLowPassFilter)
            @test size(lpf_super.data) == (imglen, imglen)
        end
    
        @testset "cu_apply_low_pass_filter!" begin
            holo_data = zeros(ComplexF32, 128, 128)
            for i in 1:128, j in 1:128
                holo_data[i, j] = ComplexF32(i + j, i - j)
            end
            holo = CuWavefront(cu(holo_data))

            lpf_data = CUDA.ones(Float32, 128, 128)
            lpf = CuLowPassFilter(lpf_data)

            original_data = copy(holo.data)
    
            cu_apply_low_pass_filter!(holo, lpf)

            @test size(holo.data) == (128, 128)
            @test eltype(holo.data) == ComplexF32

            @test sum(abs.(holo.data .- original_data)) > 1e-7
        end
    
        @testset "cu_apply_low_pass_filter" begin
            holo_data = zeros(ComplexF32, 128, 128)
            for i in 1:128, j in 1:128
                holo_data[i, j] = ComplexF32(i + j, i - j)
            end
            holo = CuWavefront(cu(holo_data))
    
            lpf_data = CUDA.ones(Float32, 128, 128)
            lpf = CuLowPassFilter(lpf_data)
    
            filtered_holo = cu_apply_low_pass_filter(holo, lpf)
    
            @test filtered_holo !== holo
            @test size(filtered_holo.data) == (128, 128)
            @test eltype(filtered_holo.data) == ComplexF32

            @test sum(abs.(filtered_holo.data .- holo.data)) > 1e-7
        end
    end

end