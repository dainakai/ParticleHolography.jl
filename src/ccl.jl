using Random
using UUIDs

export connected_component_labeling, find_valid_labels, count_labels
export get_bounding_rectangles, gen_particle_neighborhoods
export update_particle_neighborhoods!, update_particle_neighborhoods3d!
export finalize_particle_neighborhoods!
export cu_connected_component_labeling, cu_find_valid_labels

const _NEIGHBORS_8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0), (1, 1))

"""
    connected_component_labeling(image)

Label non-zero pixels using 8-way connectivity. The reference implementation
runs on the host and returns consecutive `UInt32` labels with zero reserved for
the background. Device inputs are copied one image at a time.
"""
function connected_component_labeling(image::AbstractMatrix)
    length(image) <= typemax(UInt32) || throw(ArgumentError("Image is too large for UInt32 labels."))
    foreground = .!iszero.(to_host(image))
    height, width = size(foreground)
    labels = zeros(UInt32, height, width)
    queue = Vector{Int}(undef, length(foreground))
    component = UInt32(0)

    for col in 1:width, row in 1:height
        if !foreground[row, col] || labels[row, col] != 0
            continue
        end
        component == typemax(UInt32) && throw(ArgumentError("Too many connected components for UInt32 labels."))
        component += UInt32(1)
        head = 1
        tail = 1
        queue[1] = LinearIndices(foreground)[row, col]
        labels[row, col] = component

        while head <= tail
            linear = queue[head]
            head += 1
            current = CartesianIndices(foreground)[linear]
            r, c = Tuple(current)
            for (dr, dc) in _NEIGHBORS_8
                nr = r + dr
                nc = c + dc
                if 1 <= nr <= height && 1 <= nc <= width &&
                   foreground[nr, nc] && labels[nr, nc] == 0
                    labels[nr, nc] = component
                    tail += 1
                    queue[tail] = LinearIndices(foreground)[nr, nc]
                end
            end
        end
    end
    return labels
end

function connected_component_labeling(b::AbstractBackend, image::AbstractMatrix)
    return to_backend(b, connected_component_labeling(image))
end

"""Count non-background connected-component labels."""
count_labels(labels::AbstractMatrix{<:Integer}) = length(find_valid_labels(labels))

"""Return sorted, non-zero connected-component labels."""
function find_valid_labels(labels::AbstractMatrix{<:Integer})
    return sort!(collect(Int, filter(!iszero, unique(vec(to_host(labels))))))
end

function get_bounding_rectangles(labels::AbstractMatrix{<:Integer},
                                 valid_labels::AbstractVector{<:Integer}=find_valid_labels(labels))
    host = to_host(labels)
    height, width = size(host)
    label_to_index = Dict(Int(label) => i for (i, label) in enumerate(valid_labels))
    x_min = fill(typemax(Int), length(valid_labels))
    y_min = fill(typemax(Int), length(valid_labels))
    x_max = fill(typemin(Int), length(valid_labels))
    y_max = fill(typemin(Int), length(valid_labels))

    for col in 1:width, row in 1:height
        index = get(label_to_index, Int(host[row, col]), 0)
        index == 0 && continue
        x_min[index] = min(x_min[index], col)
        y_min[index] = min(y_min[index], row)
        x_max[index] = max(x_max[index], col)
        y_max[index] = max(y_max[index], row)
    end
    return [(x_min[i], y_min[i], x_max[i], y_max[i]) for i in eachindex(valid_labels)]
end

function _new_uuid(existing, rng::AbstractRNG)
    while true
        id = uuid4(rng)
        haskey(existing, id) || return id
    end
end

function gen_particle_neighborhoods(bounding_rectangles, slicenum::Integer;
                                    rng::AbstractRNG=Random.default_rng())
    neighborhoods = Dict{UUID,Vector{Int}}()
    for (x_min, y_min, x_max, y_max) in bounding_rectangles
        id = _new_uuid(neighborhoods, rng)
        neighborhoods[id] = [x_min, y_min, Int(slicenum), x_max, y_max, Int(slicenum)]
    end
    return neighborhoods
end

function judge_overlap2d(rect1, rect2)
    x_min1, y_min1, x_max1, y_max1 = rect1
    x_min2, y_min2, x_max2, y_max2 = rect2
    return x_min1 <= x_max2 && x_max1 >= x_min2 &&
           y_min1 <= y_max2 && y_max1 >= y_min2
end

function judge_overlap3d(rect1, rect2)
    x_min1, y_min1, z_min1, x_max1, y_max1, z_max1 = rect1
    x_min2, y_min2, z_min2, x_max2, y_max2, z_max2 = rect2
    return x_min1 <= x_max2 && x_max1 >= x_min2 &&
           y_min1 <= y_max2 && y_max1 >= y_min2 &&
           z_min1 <= z_max2 && z_max1 >= z_min2
end

function new_rect(rect1, rect2)
    return [min(rect1[1], rect2[1]), min(rect1[2], rect2[2]),
            max(rect1[3], rect2[3]), max(rect1[4], rect2[4])]
end

function _merge_rectangle!(neighborhoods::Dict{UUID,Vector{Int}}, rectangle,
                           slicenum::Int, candidates;
                           rng::AbstractRNG=Random.default_rng())
    overlaps = UUID[]
    for id in candidates
        box = neighborhoods[id]
        judge_overlap2d(rectangle, (box[1], box[2], box[4], box[5])) && push!(overlaps, id)
    end

    if isempty(overlaps)
        id = _new_uuid(neighborhoods, rng)
        neighborhoods[id] = [rectangle[1], rectangle[2], slicenum,
                             rectangle[3], rectangle[4], slicenum]
        return id
    end

    target = first(overlaps)
    merged = neighborhoods[target]
    merged[1] = min(merged[1], rectangle[1])
    merged[2] = min(merged[2], rectangle[2])
    merged[3] = min(merged[3], slicenum)
    merged[4] = max(merged[4], rectangle[3])
    merged[5] = max(merged[5], rectangle[4])
    merged[6] = max(merged[6], slicenum)

    for id in Iterators.drop(overlaps, 1)
        other = neighborhoods[id]
        merged[1] = min(merged[1], other[1])
        merged[2] = min(merged[2], other[2])
        merged[3] = min(merged[3], other[3])
        merged[4] = max(merged[4], other[4])
        merged[5] = max(merged[5], other[5])
        merged[6] = max(merged[6], other[6])
        delete!(neighborhoods, id)
    end
    return target
end

"""Merge a slice's rectangles into all prior XY-overlapping neighborhoods."""
function update_particle_neighborhoods!(neighborhoods::Dict{UUID,Vector{Int}},
                                        bounding_rectangles, slicenum::Integer;
                                        rng::AbstractRNG=Random.default_rng())
    for rectangle in bounding_rectangles
        _merge_rectangle!(neighborhoods, rectangle, Int(slicenum), collect(keys(neighborhoods)); rng)
    end
    return neighborhoods
end

"""Merge rectangles only with neighborhoods present in the immediately prior slice."""
function update_particle_neighborhoods3d!(neighborhoods::Dict{UUID,Vector{Int}},
                                          bounding_rectangles, slicenum::Integer;
                                          rng::AbstractRNG=Random.default_rng())
    previous = [id for (id, box) in neighborhoods if box[6] == slicenum - 1]
    for rectangle in bounding_rectangles
        _merge_rectangle!(neighborhoods, rectangle, Int(slicenum), previous; rng)
    end
    return neighborhoods
end

function delete_duplicates!(neighborhoods::Dict{UUID,Vector{Int}})
    changed = true
    while changed
        changed = false
        ids = collect(keys(neighborhoods))
        for i in 1:length(ids), j in i + 1:length(ids)
            haskey(neighborhoods, ids[i]) && haskey(neighborhoods, ids[j]) || continue
            firstbox = neighborhoods[ids[i]]
            secondbox = neighborhoods[ids[j]]
            if judge_overlap3d(firstbox, secondbox)
                firstbox[1] = min(firstbox[1], secondbox[1])
                firstbox[2] = min(firstbox[2], secondbox[2])
                firstbox[3] = min(firstbox[3], secondbox[3])
                firstbox[4] = max(firstbox[4], secondbox[4])
                firstbox[5] = max(firstbox[5], secondbox[5])
                firstbox[6] = max(firstbox[6], secondbox[6])
                delete!(neighborhoods, ids[j])
                changed = true
                break
            end
        end
    end
    return neighborhoods
end

"""Remove duplicate, one-slice, highly elongated, and area-smaller-than-10 boxes."""
function finalize_particle_neighborhoods!(neighborhoods::Dict{UUID,Vector{Int}})
    delete_duplicates!(neighborhoods)
    delete_ids = UUID[]
    for (id, box) in neighborhoods
        width = box[4] - box[1] + 1
        height = box[5] - box[2] + 1
        depth = box[6] - box[3] + 1
        aspect = width / height
        if depth <= 1 || aspect > 3 || aspect < 1 / 3 || width * height < 10
            push!(delete_ids, id)
        end
    end
    foreach(id -> delete!(neighborhoods, id), delete_ids)
    return neighborhoods
end

function cu_connected_component_labeling(image::AbstractMatrix)
    _legacy(:cu_connected_component_labeling, :connected_component_labeling)
    b = backend(:cuda)
    return connected_component_labeling(b, image)
end

function cu_find_valid_labels(labels::AbstractMatrix{<:Integer})
    _legacy(:cu_find_valid_labels, :find_valid_labels)
    return find_valid_labels(labels)
end
