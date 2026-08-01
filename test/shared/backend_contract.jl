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
        sqrt_part = transfer_sqrt(selected, n, wavelength, pixel_pitch)
        front = transfer(selected, -800.0, wavelength, sqrt_part)
        step = transfer(selected, -20.0, wavelength, sqrt_part)
        backward = transfer(selected, 800.0, wavelength, sqrt_part)
        wavefront = gabor_wavefront(selected, image1)
        phase = phase_retrieval(PhaseRetrievalPlan(selected, front, backward),
                                image1, image2; iterations=2)
        volume, projection = reconstruct_and_projection(
            ReconstructionPlan(selected, front, step), wavefront, 4)
        binary = to_backend(selected, falses(8, 8, 2))
        binary[3:5, 3:5, :] .= true
        expanded = dilate(binary)
        padded = pad_with_mean(to_backend(selected, Float32[1 2 3; 4 5 6]), 6)
        synchronize_backend(selected)
        return (sqrt_part=to_host(sqrt_part).data,
                front=to_host(front).data,
                phase=to_host(phase).data,
                volume=to_host(volume), projection=to_host(projection),
                dilation=to_host(expanded), padded=to_host(padded))
    end

    reference = products(cpu)
    actual = products(accelerator)
    @test actual.sqrt_part == reference.sqrt_part
    @test actual.front ≈ reference.front rtol=2f-5 atol=2f-6
    @test actual.phase ≈ reference.phase rtol=2f-4 atol=2f-5
    @test actual.volume ≈ reference.volume rtol=2f-4 atol=2f-5
    @test actual.projection ≈ reference.projection rtol=2f-4 atol=2f-5
    @test actual.dilation == reference.dilation
    @test actual.padded == reference.padded
    return nothing
end
