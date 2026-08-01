module ParticleHolographyPlotsExt

using ParticleHolography
import Plots

function _particle_coordinates(data, scaling, shift)
    x = Float64[]
    y = Float64[]
    z = Float64[]
    for value in values(data)
        push!(x, value[1] * scaling[1] + shift[1])
        push!(y, value[2] * scaling[2] + shift[2])
        push!(z, value[3] * scaling[3] + shift[3])
    end
    return x, y, z
end

function particleplot(data; scaling=(1.0, 1.0, -1.0), shift=(0.0, 0.0, 0.0), kwargs...)
    x, y, z = _particle_coordinates(data, scaling, shift)
    return Plots.scatter(x, z, y; legend=false, markersize=2,
                         markerstrokewidth=0, zflip=true, yflip=true,
                         camera=(30, 30), kwargs...)
end

function particleplot!(data; scaling=(1.0, 1.0, -1.0), shift=(0.0, 0.0, 0.0), kwargs...)
    x, y, z = _particle_coordinates(data, scaling, shift)
    return Plots.scatter!(x, z, y; legend=false, markersize=2,
                          markerstrokewidth=0, zflip=true, yflip=true,
                          camera=(30, 30), kwargs...)
end

function _trajectoryplot!(plot, paths, full; colors=Plots.palette(:tab10),
                          framerange=(0, typemax(Int)),
                          scaling=(1.0, 1.0, -1.0), shift=(0.0, 0.0, 0.0),
                          kwargs...)
    for (index, path) in enumerate(paths)
        points = [full[id] for id in path if framerange[1] <= full[id][1] <= framerange[2]]
        isempty(points) && continue
        x = [point[2] * scaling[1] + shift[1] for point in points]
        y = [point[3] * scaling[2] + shift[2] for point in points]
        z = [point[4] * scaling[3] + shift[3] for point in points]
        color = colors[mod1(index, length(colors))]
        Plots.plot!(plot, x, z, y; color, linewidth=0.5, label=false, kwargs...)
        Plots.scatter!(plot, x, z, y; color, markersize=1,
                       markerstrokewidth=0, label=false)
    end
    return plot
end

function trajectoryplot(paths, full; kwargs...)
    plot = Plots.plot(; legend=false, zflip=true, yflip=true, camera=(30, 30))
    return _trajectoryplot!(plot, paths, full; kwargs...)
end

trajectoryplot!(paths, full; kwargs...) = _trajectoryplot!(Plots.current(), paths, full; kwargs...)

function save_distortion_diagnostics(image1, image2, corrected, before, after;
                                     save_dir="", grid_size=128,
                                     save_extension="png")
    directory = isempty(save_dir) ? pwd() : save_dir
    mkpath(directory)
    cells = size(before, 1)
    coordinates = collect(grid_size:grid_size:grid_size * cells)

    function diagnostic(second_image, vectors, title)
        left = Plots.heatmap(image1; aspect_ratio=:equal, yflip=true,
                             title="Camera 1", colorbar=false)
        right = Plots.heatmap(second_image; aspect_ratio=:equal, yflip=true,
                              title="Camera 2", colorbar=false)
        field = Plots.quiver(repeat(coordinates, inner=cells), repeat(coordinates, outer=cells),
                             quiver=(vec(vectors[:, :, 1]), vec(vectors[:, :, 2])),
                             aspect_ratio=:equal, yflip=true, title=title,
                             legend=false)
        return Plots.plot(left, right, field; layout=(1, 3), size=(1700, 500))
    end

    Plots.savefig(diagnostic(image2, before, "PIV before correction"),
                  joinpath(directory, "before_BA.$save_extension"))
    Plots.savefig(diagnostic(corrected, after, "PIV after correction"),
                  joinpath(directory, "after_BA.$save_extension"))
    Plots.savefig(Plots.heatmap(corrected; aspect_ratio=:equal, yflip=true,
                               colorbar=false),
                  joinpath(directory, "adjusted_image.$save_extension"))
    return nothing
end

end
