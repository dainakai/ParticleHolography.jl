using UUIDs
using Random

# Tested
export cu_connected_component_labeling, count_labels, cu_find_valid_labels, get_bounding_rectangles, gen_particle_neighborhoods, update_particle_neighborhoods!, finalize_particle_neighborhoods!

function count_labels(labels)
    components = 0
    for i in 0:length(labels)-1
        if labels[i+1] == i + 1
            components += 1
        end
    end
    return components
end

function get_bounding_rectangles(labels::Array{UInt32,2}, valid_labels::Vector{Int64})
    height, width = size(labels)
    label_to_index = Dict(l => i for (i, l) in enumerate(valid_labels))

    # Initialize arrays to store min and max coordinates for each label
    x_min = fill(typemax(Int), length(valid_labels))
    y_min = fill(typemax(Int), length(valid_labels))
    x_max = fill(typemin(Int), length(valid_labels))
    y_max = fill(typemin(Int), length(valid_labels))

    # Iterate through the labels array once
    for j in 1:width, i in 1:height
        label = labels[i, j]
        if haskey(label_to_index, label)
            idx = label_to_index[label]
            x_min[idx] = min(x_min[idx], j)
            y_min[idx] = min(y_min[idx], i)
            x_max[idx] = max(x_max[idx], j)
            y_max[idx] = max(y_max[idx], i)
        end
    end

    # Construct the bounding rectangles
    return [(x_min[i], y_min[i], x_max[i], y_max[i]) for i in 1:length(valid_labels)]
end

function gen_particle_neighborhoods(bounding_rectangles, slicenum)
    rng = MersenneTwister(1234)
    formatted_output = Dict{UUID,Vector{Int}}()

    for item in bounding_rectangles
        x_min, y_min, x_max, y_max = item
        uuid = uuid1(rng)
        formatted_output[uuid] = [x_min, y_min, slicenum, x_max, y_max, slicenum]
    end

    return formatted_output
end

function judge_overlap2d(rect1, rect2)
    x_min1, y_min1, x_max1, y_max1 = rect1
    x_min2, y_min2, x_max2, y_max2 = rect2

    if x_min1 < x_max2 && x_max1 > x_min2 && y_min1 < y_max2 && y_max1 > y_min2
        return true
    else
        return false
    end
end

function judge_overlap3d(rect1, rect2)
    x_min1, y_min1, z_min1, x_max1, y_max1, z_max1 = rect1
    x_min2, y_min2, z_min2, x_max2, y_max2, z_max2 = rect2

    if x_min1 < x_max2 && x_max1 > x_min2 && y_min1 < y_max2 && y_max1 > y_min2 && z_min1 < z_max2 && z_max1 > z_min2
        return true
    else
        return false
    end
end

function new_rect(rect1, rect2)
    x_min1, y_min1, x_max1, y_max1 = rect1
    x_min2, y_min2, x_max2, y_max2 = rect2

    x_min = min(x_min1, x_min2)
    y_min = min(y_min1, y_min2)
    x_max = max(x_max1, x_max2)
    y_max = max(y_max1, y_max2)

    return [x_min, y_min, x_max, y_max]
end

"""
    update_particle_neighborhoods!(particle_neighborhoods, bounding_rectangles, slicenum)

Updates the particle neighborhoods with new bounding rectangles from a new slice. If a bounding rectangle overlaps with an existing particle neighborhood in the x-y plane, the neighborhood is updated to include the new rectangle and the z-range is adjusted accordingly. If there is no overlap, a new particle neighborhood is created. CAUTION: This function does NOT support multiple particles along the z-axis, and it is recommended to use `update_particle_neighborhoods3d!` instead. If there are multiple particles along the z-axis, they may be merged into one particle neighborhood.

# Arguments
- `particle_neighborhoods`: The current particle neighborhoods.
- `bounding_rectangles`: The new bounding rectangles to be added.
- `slicenum`: The slice number of the new bounding rectangles.
# Returns
- `nothing`
"""
function update_particle_neighborhoods!(particle_neighborhoods, bounding_rectangles, slicenum)
    rng = MersenneTwister(1234)

    for br in bounding_rectangles
        overlapflag = false
        for item in particle_neighborhoods
            if judge_overlap2d(br, (item[2][1], item[2][2], item[2][4], item[2][5]))
                newbr = new_rect(br, (item[2][1], item[2][2], item[2][4], item[2][5]))
                item[2][1] = newbr[1]
                item[2][2] = newbr[2]
                item[2][4] = newbr[3]
                item[2][5] = newbr[4]

                if slicenum < item[2][3]
                    item[2][3] = slicenum
                elseif slicenum > item[2][6]
                    item[2][6] = slicenum
                end

                overlapflag = true
            end
        end
        if !overlapflag
            uuid = uuid1(rng)
            particle_neighborhoods[uuid] = [br[1], br[2], slicenum, br[3], br[4], slicenum]
        end
    end

    for item in particle_neighborhoods
        if abs(item[2][1] - item[2][4]) == 1 && abs(item[2][2] - item[2][5]) == 1 && abs(item[2][3] - item[2][6]) == 1
            delete!(particle_neighborhoods, item[1])
        end
    end

    return nothing
end


"""
    update_particle_neighborhoods3d!(particle_neighborhoods, bounding_rectangles, slicenum)
Updates the particle neighborhoods with new bounding rectangles from a new slice. If a bounding rectangle overlaps with an existing particle neighborhood in the x-y plane, the neighborhood is updated to include the new rectangle and the z-range is adjusted accordingly. If there is no overlap, a new particle neighborhood is created. This function supports multiple particles along the z-axis.

# Arguments
- `particle_neighborhoods`: The current particle neighborhoods.
- `bounding_rectangles`: The new bounding rectangles to be added.
- `slicenum`: The slice number of the new bounding rectangles.
# Returns
- `nothing`
"""
function update_particle_neighborhoods3d!(particle_neighborhoods, bounding_rectangles, slicenum)
    rng = MersenneTwister(1234)

    subpns = filter(((k,v), )-> v[end]==slicenum-1, particle_neighborhoods)

    for br in bounding_rectangles
        overlapflag = false
        for item in subpns
            if judge_overlap2d(br, (item[2][1], item[2][2], item[2][4], item[2][5]))
                newbr = new_rect(br, (item[2][1], item[2][2], item[2][4], item[2][5]))
                item[2][1] = newbr[1]
                item[2][2] = newbr[2]
                item[2][4] = newbr[3]
                item[2][5] = newbr[4]
                item[2][6] = slicenum

                overlapflag = true
            end
        end
        if !overlapflag
            uuid = uuid1(rng)
            particle_neighborhoods[uuid] = [br[1], br[2], slicenum, br[3], br[4], slicenum]
        end
    end

    for item in particle_neighborhoods
        if abs(item[2][1] - item[2][4]) == 1 && abs(item[2][2] - item[2][5]) == 1 && abs(item[2][3] - item[2][6]) == 1
            delete!(particle_neighborhoods, item[1])
        end
    end

    return nothing
end

"""
    finalize_particle_neighborhoods!(particle_neighborhoods)

Finalizes the particle neighborhoods by removing duplicates and particles that are too small or too elongated in the x-y plane. Detailed criteria are as follows:

* Duplicate bounding boxes
* Bounding boxes with a length-to-width (x-y) ratio greater than 3 or less than 1/3
* Bounding boxes with an area less than ``\\sqrt{10}`` pixels
* Bounding boxes with a depth of 1

# Arguments
- `particle_neighborhoods`: The particle neighborhoods to be finalized.

# Returns
- `nothing`
"""
function finalize_particle_neighborhoods!(particle_neighborhoods)
    ParticleHolography.delete_duplicates!(particle_neighborhoods)
    for item in particle_neighborhoods
        if abs(item[2][3] - item[2][6]) == 1
            delete!(particle_neighborhoods, item[1])
        end
        if abs(item[2][1] - item[2][4]) / abs(item[2][2] - item[2][5]) > 3 || abs(item[2][1] - item[2][4]) / abs(item[2][2] - item[2][5]) < 1 / 3
            delete!(particle_neighborhoods, item[1])
        end
        if abs(item[2][1] - item[2][4]) * abs(item[2][2] - item[2][5]) < 10
            delete!(particle_neighborhoods, item[1])
        end
    end

    return nothing
end

function delete_duplicates!(particle_neighborhoods)
    changed = false
    for item in particle_neighborhoods
        for item2 in particle_neighborhoods
            if item != item2
                if judge_overlap3d((item[2][1], item[2][2], item[2][3], item[2][4], item[2][5], item[2][6]), (item2[2][1], item2[2][2], item2[2][3], item2[2][4], item2[2][5], item2[2][6]))
                    newbr = new_rect((item[2][1], item[2][2], item[2][4], item[2][5]), (item2[2][1], item2[2][2], item2[2][4], item2[2][5]))
                    item[2][1] = newbr[1]
                    item[2][2] = newbr[2]
                    item[2][4] = newbr[3]
                    item[2][5] = newbr[4]
                    item[2][3] = min(item[2][3], item2[2][3])
                    item[2][6] = max(item[2][6], item2[2][6])
                    delete!(particle_neighborhoods, item2[1])
                    changed = true
                    break
                end
            end
        end
    end

    if changed
        ParticleHolography.delete_duplicates!(particle_neighborhoods)
    end

    return nothing
end
