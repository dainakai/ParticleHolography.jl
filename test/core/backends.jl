@testset "backends and wrappers" begin
    cpu = backend(:cpu)
    @test cpu isa CPUBackend
    @test backend(:auto) isa AbstractBackend
    @test :cpu in available_backends()
    @test isfunctional(cpu)
    @test backend_name(cpu) === :cpu
    @test backendof(zeros(Float32, 2, 2)) isa CPUBackend
    @test to_backend(cpu, ones(Float32, 2, 2)) == ones(Float32, 2, 2)
    @test synchronize_backend(cpu) === nothing
    @test_throws ArgumentError backend(:unknown)
    @test_throws ArgumentError backend(:cpu; device=0)

    real_data = ones(Float32, 3, 4)
    complex_data = ComplexF32.(real_data)
    for wrapped in (TransferSqrtPart(real_data), LowPassFilter(real_data),
                    Transfer(complex_data), Wavefront(complex_data))
        @test size(wrapped) == (3, 4)
        @test axes(wrapped) == axes(parent(wrapped))
        @test wrapped[1, 1] == parent(wrapped)[1, 1]
        @test backendof(wrapped) isa CPUBackend
        @test parent(to_host(wrapped)) == parent(wrapped)
    end

    @test CuTransferSqrtPart === TransferSqrtPart
    @test CuTransfer === Transfer
    @test CuWavefront === Wavefront
    @test CuLowPassFilter === LowPassFilter
end
