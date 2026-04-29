module ParticleHolographyMakieExt

using ParticleHolography
using CairoMakie
using Images
using Makie
using Statistics

function ParticleHolography._save_bundle_adjustment_diagnostics(img1::Array{<:AbstractFloat,2}, img2::Array{<:AbstractFloat,2}, vecArray, coefa; save_dir="", gridSize=128, intrSize=128, srchSize=256, save_extension="png")
    imgLen = size(img1, 1)
    errorVec = ParticleHolography.getErrorVec(vecArray, coefa, gridSize, imgLen)
    println("The mean squared error after iteration: $(mean(sqrt.(errorVec.^2)))")
    n = div(imgLen, gridSize) - 1
    f = Figure(size=(1700, 500), figure_padding=1)
    arrowax = Makie.Axis(f[1, 3], aspect=1, yreversed=false, backgroundcolor="white", title="PIV Vector Field")
    imageax1 = Makie.Axis(f[1, 1], aspect=DataAspect(), yreversed=false, title="Camera 1")
    imageax2 = Makie.Axis(f[1, 2], aspect=DataAspect(), yreversed=false, title="Camera 2")
    imgObservable1 = Observable(rotr90(RGB.(img1, img1, img1)))
    imgObservable2 = Observable(rotr90(RGB.(img2, img2, img2)))
    image!(imageax1, imgObservable1)
    image!(imageax2, imgObservable2)
    vecxObservable = Observable(rotr90(vecArray[:, :, 1]))
    vecyObservable = Observable(-rotr90(vecArray[:, :, 2]))
    strObservable = Observable(vec(sqrt.(vecArray[:, :, 1] .^ 2 .+ vecArray[:, :, 2] .^ 2)))
    xs = [i * gridSize for i in 1:n]
    ys = [i * gridSize for i in 1:n]

    function normalize_values(values)
        min_val = 0
        max_val = maximum(values)
        return (values .- min_val) / (max_val - min_val)
    end

    function color_mapping(norm_values)
        colormap = Makie.to_colormap(:viridis)
        colors = [colormap[Int(round(v * 255))+1] for v in norm_values]
        return colors
    end

    norm_strObservable = lift(x -> normalize_values(x), strObservable)
    arrow_colors = lift(x -> color_mapping(x), norm_strObservable)
    arrows!(arrowax, xs, ys, vecxObservable, vecyObservable, arrowsize=10, lengthscale=20, arrowcolor=arrow_colors, linecolor=arrow_colors)
    firstcolor = Colorbar(f[1, 4], limits=(0, maximum(vec(sqrt.(vecArray[:, :, 1] .^ 2 .+ vecArray[:, :, 2] .^ 2)))), colormap=:viridis)
    Makie.save("./"*save_dir*"/before_BA." * save_extension, f)
    img3 = ParticleHolography.quadratic_distortion_correction(img2, coefa)
    vecArray2 = ParticleHolography.getPIVMap_GPU(img1, img3, imgLen, gridSize, intrSize, srchSize)
    vecxObservable[] = rotr90(vecArray2[:, :, 1])
    vecyObservable[] = -rotr90(vecArray2[:, :, 2])
    strObservable[] = vec(sqrt.(vecArray2[:, :, 1] .^ 2 .+ vecArray2[:, :, 2] .^ 2))
    imgObservable2[] = rotr90(RGB.(img3, img3, img3))
    Makie.delete!(firstcolor)
    Colorbar(f[1, 4], limits=(0, maximum(vec(sqrt.(vecArray2[:, :, 1] .^ 2 .+ vecArray2[:, :, 2] .^ 2)))), colormap=:viridis)
    Makie.save("./"*save_dir*"/after_BA." * save_extension, f)
    Images.save("./"*save_dir*"/adjusted_image.png", img3)
    return nothing
end

end
