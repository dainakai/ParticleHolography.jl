ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using ParticleHolography
using Plots: cgrad, default, heatmap, plot, savefig, scatter!, theme
import Plots
using Random: MersenneTwister

function generate_workflow_figure()
    backend(:cpu)
    n = 128
    wavelength = 0.6328
    pixel_pitch = 10.0
    object_distance = 8_000.0
    slice_spacing = 400.0
    slices = 9
    focus_slice = fld(slices, 2) + 1

    object = ones(ComplexF32, n, n)
    particles = ((38.0, 42.0, 4.0),
                 (88.0, 48.0, 5.0),
                 (70.0, 91.0, 4.5))
    for row in axes(object, 1), col in axes(object, 2)
        transmission = 1.0
        for (x, y, radius) in particles
            radius2 = (col - x)^2 + (row - y)^2
            particle_width = 0.45radius
            transmission -= 0.85exp(-radius2 / (2particle_width^2))
        end
        object[row, col] = ComplexF32(clamp(transmission, 0.08, 1.0))
    end

    grid = propagation_grid(n, wavelength, pixel_pitch)
    camera_wavefront = asm_propagate(Wavefront(object), grid,
                                     object_distance, wavelength)
    hologram = Float32.(abs2.(camera_wavefront.data))
    front = propagation_kernel(-object_distance + (focus_slice - 1) * slice_spacing,
                               wavelength, grid)
    step = propagation_kernel(-slice_spacing, wavelength, grid)
    request = ReconstructionRequest(slices;
                                    volume=Float32,
                                    min_projection=Float32)
    result = reconstruct(ReconstructionPlan(front, step; request),
                         camera_wavefront)
    volume = result.volume
    minip = result.min_projection

    binary = volume .<= 0.86f0
    boxes = particle_bounding_boxes(binary; rng=MersenneTwister(20260801))
    coordinates = particle_coordinates(
        boxes, volume; profile_smoothing_kernel=nothing,
    )
    isempty(coordinates) && error("Synthetic documentation figure detected no particles.")

    theme(:default)
    default(fontfamily="sans-serif", guidefontsize=10, tickfontsize=8,
            titlefontsize=11, framestyle=:box)
    grayscale = cgrad(:grays)
    p1 = heatmap(hologram;
                 color=grayscale, aspect_ratio=:equal, yflip=true,
                 title="1  Camera hologram", xlabel="x (pixel)", ylabel="y (pixel)",
                 colorbar=false, xlims=(1, n))
    p2 = heatmap(volume[:, :, focus_slice];
                 color=grayscale, aspect_ratio=:equal, yflip=true,
                 title="2  Reconstructed focus plane", xlabel="x (pixel)",
                 ylabel="y (pixel)", colorbar=false, clims=(0.05, 1.05),
                 xlims=(1, n))

    depth_offsets = ((1:slices) .- focus_slice) .* slice_spacing
    xz = permutedims(volume[48, :, :], (2, 1))
    p3 = heatmap(1:n, depth_offsets, xz;
                 color=grayscale, yflip=false,
                 title="3  Depth scan through y = 48", xlabel="x (pixel)", ylabel="offset z (μm)",
                 colorbar=false, clims=(0.05, 1.05))

    p4 = heatmap(minip;
                 color=grayscale, aspect_ratio=:equal, yflip=true,
                 title="4  MinIP and detected particles", xlabel="x (pixel)",
                 ylabel="y (pixel)", colorbar=false, clims=(0.05, 1.05),
                 xlims=(1, n))
    xs = [coordinate[1] for coordinate in values(coordinates)]
    ys = [coordinate[2] for coordinate in values(coordinates)]
    scatter!(p4, xs, ys; marker=:circle, markersize=7,
             markercolor=:transparent, markerstrokecolor=:red,
             markerstrokewidth=2, label=false)

    figure = plot(p1, p2, p3, p4; layout=(2, 2), size=(1040, 780), dpi=140,
                  margin=4 * Plots.mm)
    output = joinpath(@__DIR__, "src", "assets", "hologram-to-particles.png")
    mkpath(dirname(output))
    savefig(figure, output)
    return (path=output, particles=length(coordinates), focus_slice)
end
