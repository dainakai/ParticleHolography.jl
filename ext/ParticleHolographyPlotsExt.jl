module ParticleHolographyPlotsExt

using ParticleHolography
using Plots
using UUIDs

const particleplot = ParticleHolography.particleplot
const particleplot! = ParticleHolography.particleplot!
const trajectoryplot = ParticleHolography.trajectoryplot
const trajectoryplot! = ParticleHolography.trajectoryplot!

struct ParticlePlot
    args::Tuple
end

Plots.@recipe function f(m::ParticlePlot)
    @assert length(m.args) >= 1 "ParticlePlot requires at least one argument"
    D = m.args[1]
    @assert (D isa Dict{UUID,Vector{Float32}}) "The first argument should be a Dict{UUID, Vector{Float32}}"

    pscale = get(plotattributes, :scaling, (1.0, 1.0, -1.0))
    pshift = get(plotattributes, :shift, (0.0, 0.0, 0.0))

    x = Float32[]
    y = Float32[]
    z = Float32[]
    for (_, value) in D
        push!(x, value[1] * pscale[1] + pshift[1])
        push!(y, value[2] * pscale[2] + pshift[2])
        push!(z, value[3] * pscale[3] + pshift[3])
    end

    seriestype := :scatter
    legend --> false
    dpi --> 600
    markersize --> 2
    markerstrokewidth --> 0
    zflip --> true
    yflip --> true
    camera --> (30, 30)

    x, z, y
end

struct TrajectoryPlot
    args::Tuple
end

Plots.@recipe function f(tp::TrajectoryPlot)
    @assert length(tp.args) >= 2 "TrajectoryPlot requires at least two arguments: paths and fulldict"

    paths, fulldict = tp.args[1:2]

    pscale = get(plotattributes, :scaling, (1.0, 1.0, -1.0))
    pshift = get(plotattributes, :shift, (0.0, 0.0, 0.0))
    colors = get(plotattributes, :colors, palette(:tab10))
    framerange = get(plotattributes, :framerange, (0, 1024))

    legend --> false
    dpi --> 600
    zflip --> true
    yflip --> true
    camera --> (30, 30)

    colorlen = length(colors)

    for (idx, path) in enumerate(paths)
        x = Float64[]
        y = Float64[]
        z = Float64[]
        for label in path
            particleidx = fulldict[label][1]
            if particleidx >= framerange[1] && particleidx <= framerange[2]
                push!(x, fulldict[label][2] * pscale[1] + pshift[1])
                push!(y, fulldict[label][3] * pscale[2] + pshift[2])
                push!(z, fulldict[label][4] * pscale[3] + pshift[3])
            end
        end

        if isempty(x)
            continue
        end

        @series begin
            seriestype := :path3d
            linewidth --> 0.5
            color --> colors[idx%colorlen+1]
            x, z, y
        end

        @series begin
            seriestype := :scatter3d
            markersize --> 1.0
            markerstrokewidth --> 0
            color --> colors[idx%colorlen+1]
            x, z, y
        end
    end
end

particleplot(args...; kwargs...) = Plots.plot(ParticlePlot(args); kwargs...)
particleplot!(args...; kwargs...) = Plots.plot!(ParticlePlot(args); kwargs...)
trajectoryplot(args...; kwargs...) = Plots.plot(TrajectoryPlot(args); kwargs...)
trajectoryplot!(args...; kwargs...) = Plots.plot!(TrajectoryPlot(args); kwargs...)

end
