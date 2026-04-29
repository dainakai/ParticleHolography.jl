using FixedPointNumbers
using Statistics
using ImageFiltering
using HistogramThresholding
using UUIDs

export particle_bounding_boxes, particle_coordinates, particle_coor_diams
export particle_bounding_boxes_3d, cu_dilate

"""
    tamura(arr)

Calculates the Tamura coefficient of an array. The Tamura coefficient is defined as the standard deviation divided by the mean of the array. Please refer to the use in digital holography https://doi.org/10.1364/OL.36.001945

# Arguments
- `arr::Array{Float32, 2}`: The array for which the Tamura coefficient is calculated.

# Returns
- `Float32`: The Tamura coefficient of the array.
"""
function tamura(arr::AbstractArray{<:AbstractFloat,2})
    return std(arr) / mean(arr)
end

function depth_profile(f::Function, bounding_rect_3d::AbstractArray{<:Union{AbstractFloat,Complex},3})
    return [f(bounding_rect_3d[:, :, i]) for i in axes(bounding_rect_3d, 3)]
end

function getcenterfromslice(arr::AbstractArray{<:AbstractFloat,2})
    x = 0.0
    y = 0.0
    for i in axes(arr, 1)
        for j in axes(arr, 2)
            y += i * arr[i, j]
            x += j * arr[i, j]
        end
    end
    x = x / sum(arr)
    y = y / sum(arr)
    return (x, y)
end

function _validate_particle_volume_type(T::Type, argname::String)
    if T <: Bool
        throw(ArgumentError("`$argname` must contain real-valued intensity voxels. Bool is not supported."))
    elseif !(T <: Real)
        throw(ArgumentError("`$argname` must contain real-valued intensity voxels (for example N0f8, Float32, UInt8, UInt16). Got element type $T."))
    end
    return nothing
end

function _validate_particle_volume_pair(d_vol::AbstractArray{T,3}, d_lpf_vol) where {T}
    _validate_particle_volume_type(T, "d_vol")
    if isnothing(d_lpf_vol)
        return nothing
    end

    if !(d_lpf_vol isa AbstractArray{<:Any,3})
        throw(ArgumentError("`d_lpf_vol` must be `nothing` or `AbstractArray{<:Any,3}`. Got $(typeof(d_lpf_vol))."))
    end

    _validate_particle_volume_type(eltype(d_lpf_vol), "d_lpf_vol")
    if size(d_lpf_vol) != size(d_vol)
        throw(ArgumentError("`d_lpf_vol` must have the same size as `d_vol`. Got $(size(d_lpf_vol)) and $(size(d_vol))."))
    end

    return nothing
end

function _cpu_bounding_rectangles_2d(binary_img::AbstractMatrix{Bool})
    visited = falses(size(binary_img))
    height, width = size(binary_img)
    rectangles = NTuple{4,Int}[]
    neighbors = CartesianIndex[
        CartesianIndex(-1, -1), CartesianIndex(-1, 0), CartesianIndex(-1, 1),
        CartesianIndex(0, -1), CartesianIndex(0, 1),
        CartesianIndex(1, -1), CartesianIndex(1, 0), CartesianIndex(1, 1),
    ]

    for idx in CartesianIndices(binary_img)
        if !binary_img[idx] || visited[idx]
            continue
        end

        queue = [idx]
        visited[idx] = true
        y_min = y_max = idx[1]
        x_min = x_max = idx[2]

        while !isempty(queue)
            current = popfirst!(queue)
            y_min = min(y_min, current[1])
            y_max = max(y_max, current[1])
            x_min = min(x_min, current[2])
            x_max = max(x_max, current[2])

            for delta in neighbors
                next = current + delta
                if 1 <= next[1] <= height && 1 <= next[2] <= width && binary_img[next] && !visited[next]
                    visited[next] = true
                    push!(queue, next)
                end
            end
        end

        push!(rectangles, (x_min, y_min, x_max, y_max))
    end

    return rectangles
end

"""
    particle_bounding_boxes(d_bin_vol::AbstractArray{Bool,3})

Detect particle neighborhoods from a CPU-resident binary volume. This mirrors
the CUDA path by labeling each z-slice in 2D and merging overlapping x-y boxes
across slices.
"""
function particle_bounding_boxes(d_bin_vol::AbstractArray{Bool,3})
    bounding_boxes = _cpu_bounding_rectangles_2d(@view d_bin_vol[:, :, 1])
    particle_bbs = gen_particle_neighborhoods(bounding_boxes, 1)

    for idx in 2:size(d_bin_vol, 3)
        bounding_boxes = _cpu_bounding_rectangles_2d(@view d_bin_vol[:, :, idx])
        update_particle_neighborhoods!(particle_bbs, bounding_boxes, idx)
    end

    finalize_particle_neighborhoods!(particle_bbs)
    return particle_bbs
end

function particle_bounding_boxes_3d(d_bin_vol::AbstractArray{Bool,3})
    bounding_boxes = _cpu_bounding_rectangles_2d(@view d_bin_vol[:, :, 1])
    particle_bbs = gen_particle_neighborhoods(bounding_boxes, 1)

    for idx in 2:size(d_bin_vol, 3)
        bounding_boxes = _cpu_bounding_rectangles_2d(@view d_bin_vol[:, :, idx])
        update_particle_neighborhoods3d!(particle_bbs, bounding_boxes, idx)
    end

    finalize_particle_neighborhoods!(particle_bbs)
    return particle_bbs
end


"""
    particle_coordinates(particle_bbs, d_vol; depth_metrics = tamura, profile_smoothing_kernel = Kernel.gaussian(5,))

Calculates the coordinates of the particles in the reconstructed volume with the bounding boxe dictionary. The depth of the particles is the maximum of the profile that is calculated using the `depth_metrics` function at each slice of the bounding box. The profile is then smoothed using the `profile_smoothing_kernel`. The x and y coordinates are calculated by finding the center of mass of the slice with the detected depth. The low pass filtered volume would be better for coordinate detection. The extracted subvolume is converted to `Float32` internally before evaluating metrics.

# Arguments
- `particle_bbs::Dict{UUID, Vector{Int}}`: The bounding boxes of the particles.
- `d_vol::AbstractArray{T, 3}`: The reconstructed volume. Real-valued voxel types such as `N0f8`, `Float32`, `UInt8`, and `UInt16` are supported. `Bool` and complex inputs are rejected.
- `depth_metrics::Function = tamura`: The function that calculates the depth profile of the particles.
- `profile_smoothing_kernel = Kernel.gaussian((5,))`: The kernel used for smoothing the depth profile.

# Returns
- `Dict{UUID, Vector{Float32}}`: The coordinates of the particles.
"""
function particle_coordinates(particle_bbs::Dict{UUID,Vector{Int}}, d_vol::AbstractArray{T,3}; depth_metrics::Function=tamura, profile_smoothing_kernel=Kernel.gaussian((5,))) where {T}
    _validate_particle_volume_type(T, "d_vol")
    particle_coords = Dict{UUID,Vector{Float32}}()
    for (key, value) in particle_bbs
        @views subvol = Float32.(d_vol[value[2]:value[5], value[1]:value[4], value[3]:value[6]])
        zmetric = depth_profile(depth_metrics, subvol)
        imfilter!(zmetric, zmetric, profile_smoothing_kernel)
        z = argmax(zmetric)
        (x, y) = getcenterfromslice(Array(subvol[:, :, z]))
        particle_coords[key] = [x + value[1] - 1, y + value[2] - 1, z + value[3] - 1]
    end
    return particle_coords
end

function equivalent_diameter(arr::AbstractArray{<:AbstractFloat,2})
    t = find_threshold(arr, HistogramThresholding.Otsu())
    newarr = arr .<= t
    return 2 * sqrt(sum(newarr) / π)
end

"""
    particle_coor_diams(particle_bbs, d_vol, d_lpf_vol = nothing; depth_metrics = tamura, profile_smoothing_kernel = Kernel.gaussian(5,), diameter_metrics = equivalent_diameter)

Calculates the coordinates and diameters of the particles in the reconstructed volume with the bounding boxe dictionary. The depth of the particles is the maximum of the profile that is calculated using the `depth_metrics` function at each slice of the bounding box. The profile is then smoothed using the `profile_smoothing_kernel`. The x and y coordinates are calculated by finding the center of mass of the slice with the detected depth. The low pass filtered volume would be better for coordinate detection. The diameter of the particles is calculated using the `diameter_metrics` function. If the low pass filtered volume is provided, the coordinate is calculated using the low pass filtered volume. The extracted subvolumes are converted to `Float32` internally before evaluating metrics.

# Arguments
- `particle_bbs::Dict{UUID, Vector{Int}}`: The bounding boxes of the particles.
- `d_vol::AbstractArray{T, 3}`: The reconstructed volume. Real-valued voxel types such as `N0f8`, `Float32`, `UInt8`, and `UInt16` are supported. `Bool` and complex inputs are rejected.

# Optional arguments
- `d_lpf_vol = nothing`: The low pass filtered volume. When provided, it must be a 3D `AbstractArray` with a real-valued element type and the same size as `d_vol`.

# Optional keyword arguments
- `depth_metrics::Function = tamura`: The function that calculates the depth profile of the particles.
- `profile_smoothing_kernel = Kernel.gaussian((5,))`: The kernel used for smoothing the depth profile.
- `diameter_metrics::Function = equivalent_diameter`: The function that calculates the diameter of the particles.

# Returns
- `Dict{UUID, Vector{Float32}}`: The coordinates and diameters of the particles.
"""
function particle_coor_diams(particle_bbs::Dict{UUID,Vector{Int}}, d_vol::AbstractArray{T,3}, d_lpf_vol=nothing; depth_metrics::Function=tamura, profile_smoothing_kernel=Kernel.gaussian((5,)), diameter_metrics::Function=equivalent_diameter) where {T}
    _validate_particle_volume_pair(d_vol, d_lpf_vol)
    particle_coords = Dict{UUID,Vector{Float32}}()
    for (key, value) in particle_bbs
        @views subvol = Float32.(d_vol[value[2]:value[5], value[1]:value[4], value[3]:value[6]])
        if !isnothing(d_lpf_vol)
            @views subvol_lpf = Float32.(d_lpf_vol[value[2]:value[5], value[1]:value[4], value[3]:value[6]])
            zmetric = depth_profile(depth_metrics, subvol_lpf)
            imfilter!(zmetric, zmetric, profile_smoothing_kernel)
            z = argmax(zmetric)
            (x, y) = getcenterfromslice(Array(subvol_lpf[:, :, z]))
            diam = diameter_metrics(Array(subvol[:, :, z]))
            particle_coords[key] = [x + value[1] - 1, y + value[2] - 1, z + value[3] - 1, diam]
        else
            zmetric = depth_profile(depth_metrics, subvol)
            imfilter!(zmetric, zmetric, profile_smoothing_kernel)
            z = argmax(zmetric)
            hostsubvol = Array(subvol[:, :, z])
            (x, y) = getcenterfromslice(hostsubvol)
            diam = diameter_metrics(hostsubvol)
            particle_coords[key] = [x + value[1] - 1, y + value[2] - 1, z + value[3] - 1, diam]
        end
    end
    return particle_coords
end
