using ParticleHolography
using Plots
using Test
using UUIDs

@testset "Plots extension" begin
    id = uuid4()
    particles = Dict(id => Float32[1, 2, 3])
    full = Dict(id => Float32[1, 1, 2, 3])
    @test particleplot(particles) isa Plots.Plot
    Plots.plot()
    @test particleplot!(particles) isa Plots.Plot
    @test trajectoryplot([[id]], full) isa Plots.Plot
    Plots.plot()
    @test trajectoryplot!([[id]], full) isa Plots.Plot

    mktempdir() do directory
        image1 = reshape(Float32.(1:256), 16, 16) ./ 256
        image2 = reverse(image1; dims=2)
        vectors = zeros(Float32, 2, 2, 2)
        ParticleHolography._save_distortion_diagnostics(
            image1, image2, image2, vectors, vectors;
            save_dir=directory, grid_size=8,
        )
        @test isfile(joinpath(directory, "before_BA.png"))
        @test isfile(joinpath(directory, "after_BA.png"))
        @test isfile(joinpath(directory, "adjusted_image.png"))
    end
end
