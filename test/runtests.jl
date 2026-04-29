using ParticleHolography
using Test
using UUIDs
using Statistics

const DATA_DIR = joinpath(@__DIR__, "data")

@testset "ParticleHolography CPU" begin
    @testset "image utilities" begin
        img = load_gray2float(joinpath(DATA_DIR, "holo1.bmp"))
        @test typeof(img) == Array{Float32,2}
        @test size(img) == (1024, 1024)

        gray = load_grayimg(joinpath(DATA_DIR, "holo1.bmp"))
        @test eltype(gray) == ParticleHolography.N0f8
        @test size(gray) == (1024, 1024)

        binary = load_gray2float(joinpath(DATA_DIR, "binaryparticles.png"))
        contours = find_external_contours(binary)
        @test typeof(contours) == Vector{Vector{CartesianIndex}}
        @test length(contours) == 87

        contour_img = zeros(size(binary))
        draw_contours!(contour_img, 1.0, contours)
        @test count(==(1.0), contour_img) == 6247

        pathlist = [joinpath(DATA_DIR, "holo1.bmp"), joinpath(DATA_DIR, "holo2.bmp")]
        background = make_background(pathlist; mode=:mean)
        background_mode = make_background(pathlist; mode=:mode)
        @test typeof(background) == Array{Float64,2}
        @test typeof(background_mode) == Array{Float64,2}
        @test size(background) == (1024, 1024)
        @test size(background_mode) == (1024, 1024)

        padded = pad_with_mean(img, 2048)
        @test typeof(padded) == Array{Float32,2}
        @test size(padded) == (2048, 2048)
    end

    @testset "dictionary IO" begin
        dict = Dict(uuid1() => Float32[1, 2, 3], uuid1() => Float32[4, 5, 6, 7])
        mktemp() do path, io
            close(io)
            dictsave(path, dict)
            @test dictload(path) == dict
        end
    end

    @testset "array wrapper traits" begin
        wrappers = (
            CuTransferSqrtPart(fill(1.0f0, 2, 3)),
            CuTransfer(fill(ComplexF32(1, 0), 2, 3)),
            CuWavefront(fill(ComplexF32(1, 0), 2, 3)),
            CuLowPassFilter(fill(1.0f0, 2, 3)),
        )

        for wrapper in wrappers
            @test size(wrapper) == (2, 3)
            @test axes(wrapper) == (Base.OneTo(2), Base.OneTo(3))
            @test ndims(wrapper) == 2
            @test Base.IndexStyle(typeof(wrapper)) == Base.IndexCartesian()
        end
        @test eltype(wrappers[1]) == Float32
        @test eltype(wrappers[2]) == ComplexF32
    end

    @testset "CCL CPU helpers" begin
        labels = UInt32[0 0 3; 0 5 5; 7 7 0]
        rects = ParticleHolography.get_bounding_rectangles(labels, [3, 5, 7])
        @test rects == [(3, 1, 3, 1), (2, 2, 3, 2), (1, 3, 2, 3)]

        particle_bbs = ParticleHolography.gen_particle_neighborhoods([(1, 1, 3, 3)], 1)
        ParticleHolography.update_particle_neighborhoods3d!(particle_bbs, [(2, 2, 4, 4), (10, 10, 12, 12)], 2)
        vals = collect(values(particle_bbs))
        @test length(particle_bbs) == 2
        @test any(v -> v == [1, 1, 1, 4, 4, 2], vals)
        @test any(v -> v == [10, 10, 2, 12, 12, 2], vals)
    end

    @testset "particle coordinate helpers on CPU arrays" begin
        id = uuid1()
        particle_bbs = Dict(id => [2, 2, 1, 4, 4, 3])
        vol = fill(1.0f0, 5, 5, 3)
        vol[2:4, 2:4, 2] .= 0.25f0
        coords = particle_coordinates(particle_bbs, vol; profile_smoothing_kernel=[1.0])
        coor_diams = particle_coor_diams(particle_bbs, vol; profile_smoothing_kernel=[1.0], diameter_metrics=_ -> 1.0f0)
        lpf_vol = fill(1.0f0, 5, 5, 3)
        lpf_vol[2:4, 2:4, 3] .= 0.1f0
        coor_diams_lpf = particle_coor_diams(particle_bbs, vol, lpf_vol; profile_smoothing_kernel=[1.0], diameter_metrics=_ -> 2.0f0)

        @test haskey(coords, id)
        @test length(coords[id]) == 3
        @test haskey(coor_diams, id)
        @test length(coor_diams[id]) == 4
        @test haskey(coor_diams_lpf, id)
        @test length(coor_diams_lpf[id]) == 4
        @test_throws ArgumentError particle_coordinates(particle_bbs, falses(5, 5, 3))
        @test_throws ArgumentError particle_coordinates(particle_bbs, fill(1.0f0 + 0im, 5, 5, 3))
        @test_throws ArgumentError particle_coor_diams(particle_bbs, vol, fill(1.0f0, 5, 5))
        @test_throws ArgumentError particle_coor_diams(particle_bbs, vol, falses(5, 5, 3))
        @test_throws ArgumentError particle_coor_diams(particle_bbs, vol, fill(1.0f0 + 0im, 5, 5, 3))
        @test_throws ArgumentError particle_coor_diams(particle_bbs, vol, fill(1.0f0, 4, 5, 3))
    end

    @testset "phdemo-style CPU smoke" begin
        frame_count = 5
        datlen = 16
        raw_frames = [fill(0.45f0 + 0.01f0 * frame, datlen, datlen) for frame in 1:frame_count]
        for (frame, img) in enumerate(raw_frames)
            img[4+frame:8+frame, 5+frame:9+frame] .= 0.2f0
        end
        background = sum(raw_frames) ./ frame_count
        identity_coeffs = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        mktempdir() do dir
            dict_paths = String[]
            for frame in 1:frame_count
                corrected = clamp.(raw_frames[frame] .- background .+ mean(background), 0.0f0, 1.0f0)
                corrected = quadratic_distortion_correction(corrected, identity_coeffs)
                @test size(corrected) == (datlen, datlen)

                vol = fill(1.0f0, datlen, datlen, 5)
                y0 = 4 + frame
                x0 = 5 + frame
                vol[y0:y0+4, x0:x0+4, 2:4] .= 0.2f0
                particle_bbs = particle_bounding_boxes(vol .<= 0.5f0)
                @test length(particle_bbs) == 1

                coords = particle_coor_diams(particle_bbs, vol; profile_smoothing_kernel=[1.0], diameter_metrics=_ -> 5.0f0)
                @test length(coords) == 1
                frame_coords = Dict(uuid4() => first(values(coords)))
                path = joinpath(dir, "frame_$(lpad(frame, 3, '0')).json")
                dictsave(path, frame_coords)
                push!(dict_paths, path)
            end

            dicts = dictload.(dict_paths)
            graphs = [Labonte(dict1, dict2; Dmax=10.0, dim3weight=1) for (dict1, dict2) in zip(dicts[1:end-1], dicts[2:end])]
            paths = enum_edge(graphs[1])
            for graph in graphs[2:end]
                append_path!(paths, graph)
            end

            @test any(path -> length(path) == frame_count, paths)
            @test length(gen_fulldict(dict_paths)) == frame_count
        end
    end

    @testset "tracking" begin
        dict_dir = joinpath(DATA_DIR, "dicts")
        files = sort(joinpath.(dict_dir, filter(endswith(".json"), readdir(dict_dir))))
        dicts = dictload.(files)
        graphs = [Labonte(dict1, dict2) for (dict1, dict2) in zip(dicts[1:end-1], dicts[2:end])]
        @test graphs[1] !== nothing

        paths = enum_edge(graphs[1])
        for graph in graphs[2:end]
            append_path!(paths, graph)
        end
        @test paths !== nothing
        @test gen_fulldict(dicts) == gen_fulldict(files)
        @test node_distance([1, 2, 3], [4, 6, 3]) == 5
    end

    @testset "CPU import CUDA stubs" begin
        stub_names = [
            :cu_transfer_sqrt_arr,
            :cu_transfer,
            :cu_gabor_wavefront,
            :cu_phase_retrieval_holo,
            :cu_get_reconst_vol,
            :cu_get_reconst_xyprojection,
            :cu_get_reconst_vol_and_xyprojection,
            :cu_get_reconst_complex_vol,
            Symbol("cu_asm_prop!"),
            :cu_2d_pad,
            :cu_get_reconst_vol_and_xyprojection_padded,
            :cu_rectangle_filter,
            :cu_super_gaussian_filter,
            :cu_apply_low_pass_filter,
            Symbol("cu_apply_low_pass_filter!"),
            :cu_connected_component_labeling,
            :cu_find_valid_labels,
            :cu_dilate,
            :cu_make_background_mode,
            :particle_bounding_boxes,
            :particle_bounding_boxes_3d,
            :getPIVMap_GPU,
            :get_distortion_coefficients,
        ]

        for name in stub_names
            err = try
                getfield(ParticleHolography, name)()
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            msg = sprint(showerror, err)
            @test occursin("ParticleHolography.$name requires CUDA", msg)
        end

        makie_err = try
            ParticleHolography._save_bundle_adjustment_diagnostics()
            nothing
        catch e
            e
        end
        @test makie_err isa ArgumentError
        @test occursin("require Makie and CairoMakie", sprint(showerror, makie_err))

        old_functional = ParticleHolography._cuda_functional[]
        old_status = ParticleHolography._cuda_status_message[]
        try
            ParticleHolography._cuda_functional[] = () -> error("probe failure")
            @test !ParticleHolography._cuda_available()
            @test occursin("CUDA.functional() failed", ParticleHolography._cuda_status_message[]())

            ParticleHolography._cuda_functional[] = () -> true
            ParticleHolography._cuda_status_message[] = () -> "CUDA probe succeeded."
            err = try
                ParticleHolography._cuda_required_error(:unimplemented_api)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("requires a CUDA extension method", sprint(showerror, err))
        finally
            ParticleHolography._cuda_functional[] = old_functional
            ParticleHolography._cuda_status_message[] = old_status
        end
    end
end

if get(ENV, "PARTICLEHOLOGRAPHY_RUN_GPU_TESTS", "false") == "true"
    include(joinpath(@__DIR__, "gpu", "runtests.jl"))
else
    @info "Skipping GPU tests; set PARTICLEHOLOGRAPHY_RUN_GPU_TESTS=true to run CUDA smoke tests."
end
