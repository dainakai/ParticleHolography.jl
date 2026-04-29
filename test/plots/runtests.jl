using ParticleHolography
using Plots
using Test
using UUIDs

@testset "ParticleHolography Plots extension" begin
    mktempdir() do dir
        first_id = uuid1()
        second_id = uuid1()
        particle_dict = Dict(
            first_id => Float32[1, 2, 3],
            second_id => Float32[4, 5, 6],
        )

        particle_plot = particleplot(particle_dict; scaling=(2.0, 3.0, -4.0), shift=(10.0, 20.0, 30.0))
        @test particle_plot isa Plots.Plot
        @test length(particle_plot.series_list) == 1

        particleplot!(particle_dict)
        @test length(Plots.current().series_list) >= 2

        third_id = uuid1()
        fulldict = Dict(
            first_id => Float32[1, 1, 2, 3],
            second_id => Float32[2, 2, 3, 4],
            third_id => Float32[5, 5, 6, 7],
        )
        paths = [[first_id, second_id], [third_id]]

        trajectory_plot = trajectoryplot(paths, fulldict; framerange=(1, 2))
        @test trajectory_plot isa Plots.Plot
        @test length(trajectory_plot.series_list) == 2

        trajectoryplot!(paths, fulldict; framerange=(1, 2))
        @test length(Plots.current().series_list) >= 4

        outpath = joinpath(dir, "trajectory.png")
        savefig(trajectory_plot, outpath)
        @test isfile(outpath)
        @test filesize(outpath) > 0
    end

    @test_throws AssertionError particleplot(Dict{UUID,Vector{Float64}}())
    @test_throws AssertionError trajectoryplot()
end
