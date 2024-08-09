using Plots
using UUIDs

"""
    particleplot(data::Dict{UUID, Vector{Float32}}; scaling=(1.0, 1.0, -1.0), shift=(0.0, 0.0, 0.0), kwargs...)
    particleplot!(data::Dict{UUID, Vector{Float32}}; scaling=(1.0, 1.0, -1.0), shift=(0.0, 0.0, 0.0), kwargs...)

Plots the particles in the 3D space. The data should be a dictionary with UUID keys and values as Vector{Float32}, which includes the coordinates of the particles. The scaling and shift parameters are used to transform the coordinates to the observed space.

In this plot, we display the x-y plane in the foreground by mapping the data's z-axis to the y-axis of Plots.scatter(). To align the data's y-axis with the hologram image coordinates, we set zflip = true by default. This results in a left-handed coordinate system, so we additionally set yflip = true to correct for this. The z-coordinate of the data (slice number) is opposite to the optical axis direction shown in the [Introduction](@ref introduction), so we set the z-axis data scaling to -1.0 for correct display.

# Arguments
- `data::Dict{UUID, Vector{Float32}}`: The coordinates of the particles.

# Optional keyword arguments
- `scaling::Tuple{Float32, Float32, Float32} = (1.0, 1.0, -1.0)`: The scaling of the coordinates.
- `shift::Tuple{Float32, Float32, Float32} = (0.0, 0.0, 0.0)`: The shift of the coordinates.
- `kwargs...`: Additional keyword arguments passed to Plots.scatter().
"""
@userplot ParticlePlot

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

"""
    trajectoryplot(paths::Vector{Vector{UUID}}, fulldict::Dict{UUID, Vector{Float32}}; colors=palette(:tab10), framerange=(0, 1024), scaling=(1.0, 1.0, -1.0), shift=(0.0, 0.0, 0.0), kwargs...)
    trajectoryplot!(paths::Vector{Vector{UUID}}, fulldict::Dict{UUID, Vector{Float32}}; colors=palette(:tab10), framerange=(0, 1024), scaling=(1.0, 1.0, -1.0), shift=(0.0, 0.0, 0.0), kwargs...)

Plots particle trajectories with different colors in 3D space. `paths` is a `Vector{Vector{UUID}}`, where each element represents the trajectory of particles considered identical. `fulldict` is a dictionary containing the UUIDs of particles for all conceivable frames as keys and their coordinates as values, generated using [`gen_fulldict`](@ref gen_fulldict). `colors` specifies the palette for trajectory colors, defaulting to `palette(:tab10)`. For `scaling` and `shift`, refer to [`particleplot`](@ref particleplot). `framerange` specifies the range of frames to plot. For example, when generating `fulldict` for N consecutive frames, setting `framerange=(N-10+1, N)` will plot only the trajectories of the last 10 frames. When creating animations with the `@anim` macro, setting `framerange` from `(1,1)` to `(1,N)` will create an N-frame animation.

# Arguments
- `paths::Vector{Vector{UUID}}`: The trajectories of the particles.
- `fulldict::Dict{UUID, Vector{Float32}}`: The dictionary of all particles.

# Optional keyword arguments
- `colors::Vector{Colorant} = palette(:tab10)`: The colors of the trajectories.
- `framerange::Tuple{Int, Int} = (0, 1024)`: The range of frames to plot.
- `scaling::Tuple{Float32, Float32, Float32} = (1.0, 1.0, -1.0)`: The scaling of the coordinates.
- `shift::Tuple{Float32, Float32, Float32} = (0.0, 0.0, 0.0)`: The shift of the coordinates.
- `kwargs...`: Additional keyword arguments passed to Plots.@series.
"""
@userplot TrajectoryPlot

Plots.@recipe function f(tp::TrajectoryPlot)
    @assert length(tp.args) >= 2 "TrajectoryPlot requires at least two arguments: paths and fulldict"

    paths, fulldict = tp.args[1:2]

    pscale = get(plotattributes, :scaling, (1.0, 1.0, -1.0))
    pshift = get(plotattributes, :shift, (0.0, 0.0, 0.0))
    colors = get(plotattributes, :colors, palette(:tab10))
    framerange = get(plotattributes, :framerange, (0, 1024))

    # Default values
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

        # Plot line
        @series begin
            seriestype := :path3d
            linewidth --> 0.5
            color --> colors[idx%colorlen+1]
            x, z, y
        end

        # Plot scatter points
        @series begin
            seriestype := :scatter3d
            markersize --> 1.0
            markerstrokewidth --> 0
            color --> colors[idx%colorlen+1]
            x, z, y
        end
    end
end