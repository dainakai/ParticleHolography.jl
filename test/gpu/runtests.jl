using Test
using UUIDs
using Random

@testset "ParticleHolography CUDA extension" begin
    cuda_loaded = try
        @eval import CUDA
        true
    catch err
        @info "Skipping CUDA tests because CUDA could not be imported" exception=(err, catch_backtrace())
        false
    end

    if cuda_loaded && CUDA.functional()
        using CUDA

        @testset "CUDA wrappers and kernels" begin
            data = CUDA.fill(1.0f0, (4, 4))
            transfer_sqrt_part = CuTransferSqrtPart(data)
            @test transfer_sqrt_part.data == data
            @test size(transfer_sqrt_part) == (4, 4)

            transsqr = cu_transfer_sqrt_arr(4, 0.6328, 10.0)
            @test transsqr isa CuTransferSqrtPart{Float32}
            @test size(transsqr.data) == (4, 4)

            transfer = cu_transfer(100.0, 4, 0.6328, transsqr)
            @test transfer isa CuTransfer{ComplexF32}
            @test size(transfer.data) == (4, 4)

            wf = cu_gabor_wavefront(CUDA.ones(Float32, 4, 4))
            @test wf isa CuWavefront{ComplexF32}

            padded = cu_2d_pad(CuArray(ComplexF32[1 + 1im 2 + 1im; 3 + 1im 4 + 1im]))
            @test size(padded) == (4, 4)

            rect_lpf = cu_rectangle_filter(0.01, 5.3e-7, 4, 6.5e-6)
            super_lpf = cu_super_gaussian_filter(0.01, 5.3e-7, 4, 6.5e-6)
            @test rect_lpf isa CuLowPassFilter{Float32}
            @test super_lpf isa CuLowPassFilter{Float32}

            filtered = cu_apply_low_pass_filter(wf, super_lpf)
            @test filtered isa CuWavefront{ComplexF32}
            cu_apply_low_pass_filter!(wf, rect_lpf)
            @test size(wf.data) == (4, 4)

            complex_vol = cu_get_reconst_complex_vol(wf, transfer, transfer, 2)
            vol, xyprojection = cu_get_reconst_vol_and_xyprojection(wf, transfer, transfer, 2, Float32)
            padded_sqrt = cu_transfer_sqrt_arr(8, 0.6328, 10.0)
            padded_front = cu_transfer(100.0, 8, 0.6328, padded_sqrt)
            padded_slice = cu_transfer(10.0, 8, 0.6328, padded_sqrt)
            padded_vol, padded_xyprojection = cu_get_reconst_vol_and_xyprojection_padded(wf, padded_front, padded_slice, 2, Float32)
            in_wf = CuWavefront(CUDA.ones(ComplexF32, 4, 4))
            out_wf = CuWavefront(CUDA.zeros(ComplexF32, 4, 4))
            cu_asm_prop!(out_wf, in_wf, transsqr, 100.0, 4, 0.6328)
            @test size(complex_vol) == (4, 4, 2)
            @test size(vol) == (4, 4, 2)
            @test size(xyprojection) == (4, 4)
            @test size(padded_vol) == (4, 4, 2)
            @test size(padded_xyprojection) == (4, 4)
            @test sum(abs.(Array(out_wf.data))) > 0

            bg = cu_make_background_mode([UInt8[10 20; 30 40], UInt8[10 25; 30 45], UInt8[10 20; 30 40]])
            @test size(bg) == (2, 2)
            @test isapprox(bg[1, 1], 10 / 255; atol=1e-6)

            host_vol = falses(5, 5, 1)
            host_vol[3, 3, 1] = true
            dilated = cu_dilate(cu(host_vol))
            @test sum(Array(dilated)) == 9

            labels = cu_connected_component_labeling(cu(Bool[1 1 0; 0 1 0; 1 0 1]))
            valid_labels = cu_find_valid_labels(labels)
            @test labels isa CuArray{UInt32,2}
            @test !isempty(valid_labels)
        end

        @testset "CUDA bundle adjustment smoke" begin
            Random.seed!(1234)
            img1 = rand(Float32, 64, 64)
            img2 = circshift(img1, (-8, -8))
            piv = ParticleHolography.getPIVMap_GPU(img1, img2, 64, 16, 8, 16)
            @test size(piv) == (3, 3, 2)
            @test all(isfinite, piv)

            coeffs = ParticleHolography.get_distortion_coefficients(img1, img2; gridSize=16, intrSize=8, srchSize=16)
            @test length(coeffs) == 12
            @test all(isfinite, coeffs)
        end

        @testset "phdemo-style 5 frame CUDA smoke" begin
            datlen = 16
            slices = 4
            frame_count = 5
            λ = 0.6328
            Δx = 10.0
            Δz = 20.0
            z0 = 100.0
            pr_dist = 50.0

            d_sqr = cu_transfer_sqrt_arr(datlen, λ, Δx)
            d_tf = cu_transfer(-z0, datlen, λ, d_sqr)
            d_slice = cu_transfer(-Δz, datlen, λ, d_sqr)
            d_pr = cu_transfer(pr_dist, datlen, λ, d_sqr)
            d_pr_inv = cu_transfer(-pr_dist, datlen, λ, d_sqr)

            mktempdir() do dir
                dict_paths = String[]
                for frame in 1:frame_count
                    img1 = fill(0.5f0, datlen, datlen)
                    img2 = fill(0.5f0, datlen, datlen)
                    img1[4+frame:8+frame, 5:9] .= 0.2f0
                    img2[4+frame:8+frame, 6:10] .= 0.2f0

                    d_holo = cu_phase_retrieval_holo(cu(img1), cu(img2), d_pr, d_pr_inv, 1, datlen)
                    d_vol = cu_get_reconst_vol(d_holo, d_tf, d_slice, slices, Float32)
                    @test size(d_vol) == (datlen, datlen, slices)

                    host_bin_vol = falses(datlen, datlen, slices)
                    host_bin_vol[4+frame:8+frame, 5:9, 2:4] .= true
                    particle_bbs = particle_bounding_boxes(cu(host_bin_vol))
                    @test length(particle_bbs) == 1
                    particle_bbs_3d = particle_bounding_boxes_3d(cu(host_bin_vol))
                    @test length(particle_bbs_3d) == 1
                    coords = particle_coor_diams(particle_bbs, d_vol; profile_smoothing_kernel=[1.0], diameter_metrics=_ -> 5.0f0)
                    @test length(coords) == 1

                    path = joinpath(dir, "frame_$(lpad(frame, 3, '0')).json")
                    dictsave(path, Dict(uuid4() => first(values(coords))))
                    push!(dict_paths, path)

                    xyproj = Array(cu_get_reconst_xyprojection(d_holo, d_tf, d_slice, slices))
                    @test size(xyproj) == (datlen, datlen)
                end

                dicts = dictload.(dict_paths)
                graphs = [Labonte(dict1, dict2; Dmax=20.0, dim3weight=1) for (dict1, dict2) in zip(dicts[1:end-1], dicts[2:end])]
                paths = enum_edge(graphs[1])
                for graph in graphs[2:end]
                    append_path!(paths, graph)
                end

                @test any(path -> length(path) == frame_count, paths)
                @test length(gen_fulldict(dict_paths)) == frame_count
            end
        end
    elseif cuda_loaded
        @testset "non-functional CUDA errors" begin
            err = try
                cu_transfer_sqrt_arr(4, 0.6328, 10.0)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("CUDA.functional()", sprint(showerror, err))
        end
        @info "Skipping CUDA kernel tests because CUDA.functional() returned false."
    end
end
