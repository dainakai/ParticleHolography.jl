using ParticleHolography
using Test

@testset "ParticleHolography v1 CPU core" begin
    include("core/backends.jl")
    include("core/optics.jl")
    include("core/detection.jl")
    include("core/tracking.jl")
    include("core/calibration.jl")
end
