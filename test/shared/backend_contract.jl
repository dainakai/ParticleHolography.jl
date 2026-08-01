using Test
using ParticleHolography

function run_backend_contract(accelerator::AbstractBackend)
    cpu = CPUBackend()
    n = 16
    wavelength = 0.6328
    pixel_pitch = 10.0
    image1 = reshape(Float32.(range(0.1, 0.9; length=n^2)), n, n)
    image2 = reverse(image1; dims=2)

    function products(selected)
        backend(backend_name(selected))
        @test backend_name(backend()) === backend_name(selected)
        grid = propagation_grid(n, wavelength, pixel_pitch)
        front = propagation_kernel(-800.0, wavelength, grid)
        step = propagation_kernel(-20.0, wavelength, grid)
        backward = propagation_kernel(800.0, wavelength, grid)
        wavefront = gabor_wavefront(image1)
        phase = phase_retrieval(PhaseRetrievalPlan(selected, front, backward),
                                image1, image2; iterations=2)
        request = ReconstructionRequest(4;
                                        volume=Float32,
                                        min_projection=N0f8)
        plan = ReconstructionPlan(selected, front, step; request)
        reconstruction = reconstruct(plan, wavefront)
        plane = Wavefront(to_backend(selected, ones(ComplexF32, n, n)))
        propagated = asm_propagate(plane, grid, 37.0, wavelength)
        round_trip = asm_propagate(
            asm_propagate(plane, grid, 150.0, wavelength),
            grid, -150.0, wavelength,
        )

        object_data = Matrix{ComplexF32}(undef, n, n)
        centre = (n + 1) / 2
        for col in 1:n, row in 1:n
            radius2 = (row - centre)^2 + (col - centre)^2
            object_data[row, col] = ComplexF32(1 - 0.8exp(-radius2 / 4))
        end
        object_wavefront = Wavefront(to_backend(selected, object_data))
        camera_wavefront = asm_propagate(object_wavefront, grid, 400.0, wavelength)
        focus_front = propagation_kernel(-320.0, wavelength, grid)
        focus_step = propagation_kernel(-40.0, wavelength, grid)
        focus_request = ReconstructionRequest(5; volume=ComplexF32)
        focus_volume = to_host(reconstruct(
            ReconstructionPlan(focus_front, focus_step; request=focus_request),
            camera_wavefront,
        ).volume)
        focus_errors = [sum(abs2, @view(focus_volume[:, :, z]) .- object_data) /
                        length(object_data) for z in 1:focus_request.slices]

        binary = to_backend(selected, falses(8, 8, 2))
        binary[3:5, 3:5, :] .= true
        expanded = dilate(binary)
        padded = pad_with_mean(to_backend(selected, Float32[1 2 3; 4 5 6]), 6)
        synchronize_backend(selected)
        return (grid=to_host(grid).data,
                front=to_host(front).data,
                phase=to_host(phase).data,
                volume=to_host(reconstruction.volume),
                projection=to_host(reconstruction.min_projection),
                propagated=to_host(propagated).data,
                round_trip=to_host(round_trip).data,
                focus_errors,
                dilation=to_host(expanded), padded=to_host(padded),
                diagnostic=memory_diagnostic(plan, request))
    end

    reference = products(cpu)
    actual = products(accelerator)
    @test actual.grid == reference.grid
    @test actual.front ≈ reference.front rtol=2f-5 atol=2f-6
    @test actual.phase ≈ reference.phase rtol=2f-4 atol=2f-5
    @test actual.volume ≈ reference.volume rtol=2f-4 atol=2f-5
    @test Float32.(actual.projection) ≈ Float32.(reference.projection) atol=1f-2
    @test actual.propagated ≈ reference.propagated rtol=2f-4 atol=2f-5
    @test actual.round_trip ≈ reference.round_trip rtol=2f-4 atol=2f-5
    @test argmin(actual.focus_errors) == 3
    @test actual.focus_errors[3] < 2f-9
    @test actual.focus_errors ≈ reference.focus_errors rtol=2f-3 atol=2f-8
    @test actual.dilation == reference.dilation
    @test actual.padded == reference.padded
    @test actual.diagnostic.safe !== false
    return nothing
end
