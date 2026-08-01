using FixedPointNumbers: N0f8

@testset "optical core" begin
    cpu = CPUBackend()
    n = 16
    wavelength = 0.6328
    pixel_pitch = 10.0
    sqrt_part = transfer_sqrt(cpu, n, wavelength, pixel_pitch)
    @test size(sqrt_part) == (n, n)
    @test sqrt_part[1, 1] == 1.0f0
    @test all(>=(0), sqrt_part.data)

    front = transfer(cpu, -800.0, wavelength, sqrt_part)
    step = transfer(cpu, -20.0, wavelength, sqrt_part)
    inverse = transfer(cpu, 800.0, wavelength, sqrt_part)
    @test all(isapprox.(abs.(front.data), 1; atol=2f-6))

    hologram1 = reshape(Float32.(range(0.1, 0.9; length=n^2)), n, n)
    hologram2 = reverse(hologram1; dims=1)
    wavefront = gabor_wavefront(cpu, hologram1)
    @test abs2.(wavefront.data) ≈ hologram1 rtol=2f-6

    phase_plan = PhaseRetrievalPlan(cpu, front, inverse)
    owning = phase_retrieval(phase_plan, hologram1, hologram2; iterations=2)
    aliasing = phase_retrieval!(phase_plan, hologram1, hologram2; iterations=2)
    @test owning.data ≈ aliasing.data rtol=5f-5 atol=5f-6
    @test aliasing.data === phase_plan.light1
    @test all(isfinite, owning.data)

    plan = ReconstructionPlan(cpu, front, step)
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
    @test eltype(complex_volume) === ComplexF32

    padded = pad2d(wavefront.data)
    @test size(padded) == (2n, 2n)
    @test padded[n÷2+1:n÷2+n, n÷2+1:n÷2+n] == wavefront.data

    propagated = asm_propagate(wavefront, sqrt_part, 30.0, wavelength)
    @test size(propagated) == size(wavefront)
    @test all(isfinite, propagated.data)

    identity_filter = LowPassFilter(ones(Float32, n, n))
    filtered = apply_low_pass_filter(wavefront, identity_filter)
    @test filtered !== wavefront
    @test filtered.data ≈ wavefront.data rtol=2f-5 atol=2f-6
    @test size(rectangle_filter(cpu, 800.0, wavelength, n, pixel_pitch)) == (n, n)
    @test size(super_gaussian_filter(cpu, 800.0, wavelength, n, pixel_pitch)) == (n, n)

    @test_throws ArgumentError transfer_sqrt(cpu, 0, wavelength, pixel_pitch)
    @test_throws ArgumentError transfer_sqrt(cpu, n, -wavelength, pixel_pitch)
    @test_throws DimensionMismatch PhaseRetrievalPlan(cpu, front,
                                                       Transfer(ones(ComplexF32, n ÷ 2, n ÷ 2)))
end
