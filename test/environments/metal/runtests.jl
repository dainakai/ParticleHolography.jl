using ParticleHolography
using Metal
using Test

Metal.functional() || error("Metal is not functional on the selected macOS runner.")
include(joinpath(@__DIR__, "..", "..", "shared", "backend_contract.jl"))

@testset "Metal backend contract" begin
    @test :metal in available_backends()
    array = Metal.MtlArray(zeros(Float32, 2, 2))
    @test backendof(array) isa ParticleHolography.MetalBackend
    @test backendof(@view array[:, 1:1]) isa ParticleHolography.MetalBackend
    run_backend_contract(backend(:metal))
end
