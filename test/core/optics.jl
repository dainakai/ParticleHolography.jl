using FixedPointNumbers: N0f8
using Statistics: mean

function quantized_float_reference(array)
    return Float32.(round.(UInt8, clamp.(array, 0, 1) .* 255)) ./ 255
end

@testset "optical core" begin
    cpu = backend(:cpu)
    n = 16
    wavelength = 0.6328
    pixel_pitch = 10.0

    # Selecting a backend once controls constructors that omit it.
    grid = propagation_grid(n, wavelength, pixel_pitch)
    @test grid isa PropagationGrid
    @test backendof(grid) isa CPUBackend
    @test size(grid) == (n, n)
    @test grid[1, 1] == 1.0f0
    @test all(>=(0), grid.data)

    front = propagation_kernel(-800.0, wavelength, grid)
    step = propagation_kernel(-20.0, wavelength, grid)
    inverse = propagation_kernel(800.0, wavelength, grid)
    @test front isa PropagationKernel
    @test all(isapprox.(abs.(front.data), 1; atol=2f-6))

    hologram1 = reshape(Float32.(range(0.1, 0.9; length=n^2)), n, n)
    hologram2 = reverse(hologram1; dims=1)
    wavefront = gabor_wavefront(hologram1)
    @test abs2.(wavefront.data) ≈ hologram1 rtol=2f-6

    phase_plan = PhaseRetrievalPlan(front, inverse)
    phase_diagnostic = memory_diagnostic(phase_plan)
    @test phase_diagnostic.scope === :phase_plan
    @test phase_diagnostic.safe === true
    @test_throws ArgumentError PhaseRetrievalPlan(
        cpu, front, inverse; available_memory=1,
    )
    owning = phase_retrieval(phase_plan, hologram1, hologram2; iterations=2)
    aliasing = phase_retrieval!(phase_plan, hologram1, hologram2; iterations=2)
    @test owning.data ≈ aliasing.data rtol=5f-5 atol=5f-6
    @test aliasing.data === phase_plan.light1
    @test all(isfinite, owning.data)

    plan = ReconstructionPlan(front, step)
    frequency_buffer = plan.frequency
    volume, projection = reconstruct_and_projection(plan, wavefront, 4)
    @test size(volume) == (n, n, 4)
    @test size(projection) == (n, n)
    @test eltype(volume) === Float32
    @test all(isfinite, volume)
    @test projection ≈ dropdims(minimum(volume; dims=3); dims=3) rtol=2f-5
    repeated = reconstruct(plan, wavefront, 4)
    @test repeated ≈ volume rtol=2f-5
    @test plan.frequency === frequency_buffer

    quantized = reconstruct(plan, wavefront, 2; output=N0f8, clamp_output=true)
    @test eltype(quantized) === N0f8
    @test all(value -> zero(N0f8) <= value <= typemax(N0f8), quantized)

    complex_volume = reconstruct_complex(plan, wavefront, 3)
    @test size(complex_volume) == (n, n, 3)
    @test eltype(complex_volume) === ComplexF64
    complex32 = reconstruct_complex(plan, wavefront, 3; output=ComplexF32)
    @test ComplexF32.(complex_volume) ≈ complex32 rtol=2f-6 atol=2f-6

    # Every supported volume/MinIP combination uses the same result contract.
    reference_request = ReconstructionRequest(3;
                                              volume=ComplexF32,
                                              min_projection=Float32)
    reference = reconstruct(plan, wavefront, reference_request)
    reference_complex = to_host(reference.volume)
    reference_intensity = Float32.(abs2.(reference_complex))
    reference_projection = dropdims(minimum(reference_intensity; dims=3); dims=3)
    for volume_type in (nothing, N0f8, Float32, ComplexF32, ComplexF64)
        for projection_type in (nothing, N0f8, Float32)
            isnothing(volume_type) && isnothing(projection_type) && continue
            request = ReconstructionRequest(3;
                                            volume=volume_type,
                                            min_projection=projection_type)
            result = reconstruct(plan, wavefront, request)
            if isnothing(volume_type)
                @test isnothing(result.volume)
            elseif volume_type === N0f8
                actual = Float32.(to_host(result.volume))
                @test actual ≈ quantized_float_reference(reference_intensity) atol=1f-6
            elseif volume_type === Float32
                @test to_host(result.volume) ≈ reference_intensity rtol=2f-6 atol=2f-6
            else
                @test eltype(result.volume) === volume_type
                @test ComplexF32.(to_host(result.volume)) ≈ reference_complex rtol=2f-6 atol=2f-6
            end
            if isnothing(projection_type)
                @test isnothing(result.min_projection)
            elseif projection_type === N0f8
                actual = Float32.(to_host(result.min_projection))
                @test actual ≈ quantized_float_reference(reference_projection) atol=1f-6
            else
                @test to_host(result.min_projection) ≈ reference_projection rtol=2f-6 atol=2f-6
            end
        end
    end

    stored_request = ReconstructionRequest(2;
                                           volume=Float32,
                                           min_projection=N0f8)
    stored_plan = ReconstructionPlan(front, step; request=stored_request)
    stored_result = reconstruct(stored_plan, wavefront)
    @test size(stored_result.volume) == (n, n, 2)
    @test eltype(stored_result.min_projection) === N0f8
    @test_throws ArgumentError reconstruct(plan, wavefront)

    diagnostic = memory_diagnostic(plan, stored_request)
    @test diagnostic.safe === true
    @test diagnostic.scope === :outputs
    @test diagnostic.output_bytes == n^2 * (2 * sizeof(Float32) + sizeof(N0f8))
    @test diagnostic.temporary_bytes == n^2 * sizeof(Float32)
    @test occursin("MemoryDiagnostic(cpu", sprint(show, diagnostic))
    @test_throws ArgumentError ReconstructionPlan(
        cpu, front, step; request=stored_request, available_memory=1,
    )
    unchecked = ReconstructionPlan(
        cpu, front, step; request=stored_request, available_memory=1,
        check_memory=false,
    )
    @test unchecked isa ReconstructionPlan

    # Mean padding is explicit, and padded reconstruction stores only the crop.
    padded_data = pad2d(wavefront.data)
    @test size(padded_data) == (2n, 2n)
    @test padded_data[n÷2+1:n÷2+n, n÷2+1:n÷2+n] == wavefront.data
    @test padded_data[1, 1] ≈ mean(wavefront.data)
    zero_padded = pad2d(wavefront.data, (2n, 2n); mode=:zero)
    @test zero_padded[1, 1] == 0

    padded_grid = propagation_grid(cpu, 2n, wavelength, pixel_pitch)
    padded_front = propagation_kernel(cpu, -800.0, wavelength, padded_grid)
    padded_step = propagation_kernel(cpu, -20.0, wavelength, padded_grid)
    padded_request = ReconstructionRequest(3;
                                           volume=Float32,
                                           min_projection=N0f8)
    padded_plan = ReconstructionPlan(
        cpu, padded_front, padded_step;
        request=padded_request, output_shape=size(wavefront),
    )
    cropped = reconstruct_padded(padded_plan, wavefront)
    @test size(cropped.volume) == (n, n, 3)
    @test size(cropped.min_projection) == (n, n)
    full = reconstruct(padded_plan, Wavefront(padded_data), padded_request)
    rows = n ÷ 2 + 1:n ÷ 2 + n
    cols = n ÷ 2 + 1:n ÷ 2 + n
    @test cropped.volume ≈ full.volume[rows, cols, :] rtol=2f-6 atol=2f-6
    @test cropped.min_projection == full.min_projection[rows, cols]

    propagated = asm_propagate(wavefront, grid, 30.0, wavelength)
    @test size(propagated) == size(wavefront)
    @test all(isfinite, propagated.data)

    identity_filter = LowPassFilter(ones(Float32, n, n))
    filtered = apply_low_pass_filter(wavefront, identity_filter)
    @test filtered !== wavefront
    @test filtered.data ≈ wavefront.data rtol=2f-5 atol=2f-6
    @test size(rectangle_filter(800.0, wavelength, n, pixel_pitch)) == (n, n)
    @test size(super_gaussian_filter(800.0, wavelength, n, pixel_pitch)) == (n, n)

    @test_throws ArgumentError propagation_grid(cpu, 0, wavelength, pixel_pitch)
    @test_throws ArgumentError propagation_grid(cpu, n, -wavelength, pixel_pitch)
    @test_throws DimensionMismatch PhaseRetrievalPlan(
        cpu, front, PropagationKernel(ones(ComplexF32, n ÷ 2, n ÷ 2)),
    )
    @test_throws ArgumentError ReconstructionRequest(2; volume=nothing,
                                                     min_projection=nothing)
    @test_throws ArgumentError ReconstructionRequest(2; volume=Float64)
    @test_throws ArgumentError ReconstructionRequest(2; volume=nothing,
                                                     min_projection=Float64)
    @test_throws ArgumentError memory_diagnostic(cpu, (n, n), stored_request;
                                                 safety_factor=0.5)
    @test_throws ArgumentError memory_diagnostic(plan, stored_request;
                                                 output_shape=(n + 1, n))
    @test_throws ArgumentError pad2d(wavefront.data, (n - 1, n))
    @test_throws ArgumentError pad2d(wavefront.data; mode=:edge)
end

@testset "physical reconstruction accuracy" begin
    backend(:cpu)
    n = 32
    wavelength = 0.6328
    pixel_pitch = 10.0
    grid = propagation_grid(n, wavelength, pixel_pitch)

    # A normally incident plane wave has a closed-form propagation phase.
    distance = 37.0
    plane = Wavefront(ones(ComplexF32, n, n))
    propagated_plane = asm_propagate(plane, grid, distance, wavelength)
    expected_phase = ComplexF32(exp(complex(0.0, 2π * distance / wavelength)))
    @test propagated_plane.data ≈ fill(expected_phase, n, n) rtol=3f-5 atol=3f-5

    # Forward and backward angular-spectrum propagation should invert each other.
    structured = Matrix{ComplexF32}(undef, n, n)
    for j in 1:n, i in 1:n
        structured[i, j] = ComplexF32(0.6 + 0.2cos(2π * i / n) +
                                     0.1sin(4π * j / n),
                                     0.15sin(2π * (i + j) / n))
    end
    original = Wavefront(structured)
    round_trip = asm_propagate(
        asm_propagate(original, grid, 250.0, wavelength),
        grid, -250.0, wavelength,
    )
    @test round_trip.data ≈ original.data rtol=5f-5 atol=5f-5

    # A known object propagated to the camera refocuses at the known depth.
    object_data = Matrix{ComplexF32}(undef, n, n)
    centre = (n + 1) / 2
    for j in 1:n, i in 1:n
        radius2 = (i - centre)^2 + (j - centre)^2
        object_data[i, j] = ComplexF32(1 - 0.8exp(-radius2 / 8))
    end
    object_wavefront = Wavefront(object_data)
    object_distance = 400.0
    depth_step = 40.0
    camera_wavefront = asm_propagate(object_wavefront, grid,
                                     object_distance, wavelength)
    front = propagation_kernel(-object_distance + 2depth_step,
                               wavelength, grid)
    step = propagation_kernel(-depth_step, wavelength, grid)
    plan = ReconstructionPlan(front, step)
    request = ReconstructionRequest(5;
                                    volume=ComplexF32,
                                    min_projection=Float32)
    result = reconstruct(plan, camera_wavefront, request)
    errors = [mean(abs2, @view(result.volume[:, :, z]) .- object_data)
              for z in 1:request.slices]
    @test argmin(errors) == 3
    @test errors[3] < 2f-9
    @test errors[3] < minimum(errors[[1, 2, 4, 5]]) / 100
    expected_intensity = Float32.(abs2.(object_data))
    @test Float32.(abs2.(@view result.volume[:, :, 3])) ≈ expected_intensity rtol=5f-5 atol=5f-5
    @test result.min_projection ≈ dropdims(
        minimum(Float32.(abs2.(result.volume)); dims=3); dims=3,
    ) rtol=5f-5 atol=5f-5
end
