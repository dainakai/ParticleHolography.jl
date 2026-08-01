using HistogramThresholding
using ImageFiltering
using Statistics
using UUIDs

export dilate, cu_dilate
export particle_bounding_boxes, particle_bounding_boxes_3d
export particle_coordinates, particle_coor_diams
export tamura, depth_profile, equivalent_diameter

"""
    dilate(volume)

Dilate each XY slice with a 3×3 neighbourhood. The operation stays on the
input array's backend and preserves the v0.2 convention that boundary pixels
remain false.
"""
function dilate(volume::AbstractArray{Bool,3})
    height, width, slices = size(volume)
    output = similar(volume, Bool, size(volume))
    fill!(output, false)
    (height < 3 || width < 3) && return output
    @views interior = output[2:height-1, 2:width-1, :]
    for dr in -1:1, dc in -1:1
        @views interior .= interior .| volume[2+dr:height-1+dr,
                                             2+dc:width-1+dc, 1:slices]
    end
    return output
end

function cu_dilate(volume::AbstractArray{Bool,3}; blocksize::Integer=32)
    _legacy(:cu_dilate, :dilate)
    blocksize > 0 || throw(ArgumentError("blocksize must be positive."))
    return dilate(volume)
end

function _slice_bounding_rectangles(volume::AbstractArray{Bool,3}, slice::Int)
    @views labels = connected_component_labeling(volume[:, :, slice])
    return get_bounding_rectangles(labels)
end

function _particle_bounding_boxes(volume::AbstractArray{Bool,3}, adjacent_only::Bool;
                                  rng::AbstractRNG=Random.default_rng())
    size(volume, 3) > 0 || throw(ArgumentError("Binary volume must contain at least one slice."))
    rectangles = _slice_bounding_rectangles(volume, 1)
    neighborhoods = gen_particle_neighborhoods(rectangles, 1; rng)
    for slice in 2:size(volume, 3)
        rectangles = _slice_bounding_rectangles(volume, slice)
        if adjacent_only
            update_particle_neighborhoods3d!(neighborhoods, rectangles, slice; rng)
        else
            update_particle_neighborhoods!(neighborhoods, rectangles, slice; rng)
        end
    end
    return finalize_particle_neighborhoods!(neighborhoods)
end

"""
    particle_bounding_boxes(binary_volume)

Connect slice components whenever their inclusive XY bounding boxes overlap.
This keeps the original non-adjacent-z behaviour used to join fragmented
holographic particle signatures.
"""
particle_bounding_boxes(volume::AbstractArray{Bool,3}; rng::AbstractRNG=Random.default_rng()) =
    _particle_bounding_boxes(volume, false; rng)

"""Variant that joins a component only to a component in the preceding slice."""
particle_bounding_boxes_3d(volume::AbstractArray{Bool,3}; rng::AbstractRNG=Random.default_rng()) =
    _particle_bounding_boxes(volume, true; rng)

"""Tamura focus coefficient (`std / mean`) with a stable all-zero case."""
function tamura(array::AbstractMatrix{<:AbstractFloat})
    μ = mean(array)
    iszero(μ) && return zero(float(eltype(array)))
    return std(array) / μ
end

depth_profile(metric::Function, volume::AbstractArray{<:Union{AbstractFloat,Complex},3}) =
    [metric(@view volume[:, :, slice]) for slice in axes(volume, 3)]

function getcenterfromslice(array::AbstractMatrix{<:AbstractFloat})
    total = sum(array)
    if iszero(total) || !isfinite(total)
        return ((first(axes(array, 2)) + last(axes(array, 2))) / 2,
                (first(axes(array, 1)) + last(axes(array, 1))) / 2)
    end
    x = 0.0
    y = 0.0
    for row in axes(array, 1), col in axes(array, 2)
        value = array[row, col]
        y += row * value
        x += col * value
    end
    return (x / total, y / total)
end

function _validate_particle_volume_type(T::Type, name::String)
    T <: Bool && throw(ArgumentError("`$name` must contain real-valued intensity voxels; Bool is not supported."))
    T <: Real || throw(ArgumentError("`$name` must contain real-valued intensity voxels. Got $T."))
    return nothing
end

function _validate_particle_volume_pair(volume::AbstractArray{T,3}, filtered) where {T}
    _validate_particle_volume_type(T, "volume")
    isnothing(filtered) && return nothing
    filtered isa AbstractArray{<:Any,3} || throw(ArgumentError("filtered volume must be nothing or a three-dimensional array."))
    _validate_particle_volume_type(eltype(filtered), "filtered")
    size(filtered) == size(volume) || throw(DimensionMismatch("Filtered and unfiltered volumes must have the same size."))
    return nothing
end

function _validated_box(box::AbstractVector{<:Integer}, volume_size)
    length(box) == 6 || throw(ArgumentError("Particle bounding boxes must have six entries [xmin, ymin, zmin, xmax, ymax, zmax]."))
    xmin, ymin, zmin, xmax, ymax, zmax = box
    1 <= xmin <= xmax <= volume_size[2] || throw(ArgumentError("Invalid x range $xmin:$xmax for width $(volume_size[2])."))
    1 <= ymin <= ymax <= volume_size[1] || throw(ArgumentError("Invalid y range $ymin:$ymax for height $(volume_size[1])."))
    1 <= zmin <= zmax <= volume_size[3] || throw(ArgumentError("Invalid z range $zmin:$zmax for depth $(volume_size[3])."))
    return (xmin, ymin, zmin, xmax, ymax, zmax)
end

function _host_subvolume(volume::AbstractArray, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box
    @views return Float32.(to_host(volume[ymin:ymax, xmin:xmax, zmin:zmax]))
end

function _focus_slice(volume::Array{Float32,3}, depth_metric::Function, smoothing_kernel)
    profile = Float32.(depth_profile(depth_metric, volume))
    if !isnothing(smoothing_kernel) && length(profile) > 1
        imfilter!(profile, copy(profile), smoothing_kernel)
    end
    return argmax(profile)
end

"""Calculate `[x, y, z]` coordinates for detected particle bounding boxes."""
function particle_coordinates(boxes::Dict{UUID,Vector{Int}},
                              volume::AbstractArray{T,3};
                              depth_metrics::Function=tamura,
                              profile_smoothing_kernel=Kernel.gaussian((5,))) where {T}
    _validate_particle_volume_type(T, "volume")
    coordinates = Dict{UUID,Vector{Float32}}()
    for (id, raw_box) in boxes
        box = _validated_box(raw_box, size(volume))
        subvolume = _host_subvolume(volume, box)
        z = _focus_slice(subvolume, depth_metrics, profile_smoothing_kernel)
        x, y = getcenterfromslice(@view subvolume[:, :, z])
        coordinates[id] = Float32[x + box[1] - 1, y + box[2] - 1, z + box[3] - 1]
    end
    return coordinates
end

function equivalent_diameter(array::AbstractMatrix{<:AbstractFloat})
    isempty(array) && return 0.0
    threshold = find_threshold(array, HistogramThresholding.Otsu())
    return 2 * sqrt(count(<=(threshold), array) / π)
end

"""Calculate `[x, y, z, equivalent_diameter]` for each particle box."""
function particle_coor_diams(boxes::Dict{UUID,Vector{Int}},
                             volume::AbstractArray{T,3}, filtered=nothing;
                             depth_metrics::Function=tamura,
                             profile_smoothing_kernel=Kernel.gaussian((5,)),
                             diameter_metrics::Function=equivalent_diameter) where {T}
    _validate_particle_volume_pair(volume, filtered)
    result = Dict{UUID,Vector{Float32}}()
    for (id, raw_box) in boxes
        box = _validated_box(raw_box, size(volume))
        raw = _host_subvolume(volume, box)
        focus_volume = isnothing(filtered) ? raw : _host_subvolume(filtered, box)
        z = _focus_slice(focus_volume, depth_metrics, profile_smoothing_kernel)
        x, y = getcenterfromslice(@view focus_volume[:, :, z])
        diameter = diameter_metrics(@view raw[:, :, z])
        result[id] = Float32[x + box[1] - 1, y + box[2] - 1,
                             z + box[3] - 1, diameter]
    end
    return result
end
