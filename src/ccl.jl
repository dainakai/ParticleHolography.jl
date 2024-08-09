using CUDA
using UUIDs
using Random

# Tested
export cu_connected_component_labeling, count_labels, cu_find_valid_labels, get_bounding_rectangles, gen_particle_neighborhoods, update_particle_neighborhoods!, finalize_particle_neighborhoods!

# CUDA 8-way Connected Component Labelling
# Please refer to: https://github.com/FolkeV/CUDA_CCL

# ---------- reduction.cuh ----------
# ---------- Find the root of a chain ----------
@inline function find_root(labels, label)
    # Resolve label
    next = labels[label+1]

    # Follow chain
    while label != next
        # Move to next
        label = next
        next = labels[label+1]
    end

    return label
end

# ---------- Label Reduction ----------
@inline function reduction(g_labels, label1, label2)
    # Get next labels
    next1 = (label1 != label2) ? g_labels[label1+1] : 1
    next2 = (label1 != label2) ? g_labels[label2+1] : 1

    # Find label1
    while (label1 != label2) && (label1 != next1)
        # Adopt label
        label1 = next1

        # Fetch next label
        next1 = g_labels[label1+1]
    end

    # Find label2
    while (label1 != label2) && (label2 != next2)
        # Adopt label
        label2 = next2

        # Fetch next label
        next2 = g_labels[label2+1]
    end

    label3 = 0
    # While labels are different
    while label1 != label2
        # label2 should be smallest
        if label1 < label2
            # Swap labels
            tmp = label1
            label1 = label2
            label2 = tmp
        end
        # AtomicMin label1 to label2
        label3 = CUDA.@atomic g_labels[label1+1] = min(g_labels[label1+1], label2)
        label1 = (label1 == label3) ? label2 : label3
    end

    return label1
end

# ---------- ccl.cu ----------
function init_labels(g_labels, g_image, numCols, numRows)
    # Calculate index
    ix = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
    iy = threadIdx().y + (blockIdx().y - 1) * blockDim().y - 1

    # Check thread range
    if (ix < numCols) && (iy < numRows)
        pyx = g_image[iy*numCols+ix+1]

        # Neighbour Connections
        nym1x = (iy > 0) ? (pyx == g_image[(iy-1)*numCols+ix+1]) : false
        nyxm1 = (ix > 0) ? (pyx == g_image[(iy)*numCols+ix-1+1]) : false
        nym1xm1 = ((iy > 0) && (ix > 0)) ? (pyx == g_image[(iy-1)*numCols+ix-1+1]) : false
        nym1xp1 = ((iy > 0) && (ix < numCols - 1)) ? (pyx == g_image[(iy-1)*numCols+ix+1+1]) : false

        # Initialise Label
        # Label will be chosen in the following order:
        # NW > N > NE > E > current position
        label = (nyxm1) ? iy * numCols + ix - 1 : iy * numCols + ix
        label = (nym1xp1) ? (iy - 1) * numCols + ix + 1 : label
        label = (nym1x) ? (iy - 1) * numCols + ix : label
        label = (nym1xm1) ? (iy - 1) * numCols + ix - 1 : label

        # Write to Global Memory
        @inbounds g_labels[iy*numCols+ix+1] = label
    end

    return nothing
end

function resolve_labels(g_labels, numCols, numRows)
    # Calculate index
    ix = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
    iy = threadIdx().y + (blockIdx().y - 1) * blockDim().y - 1
    id = ix + iy * numCols

    # Check thread range
    if id < numCols * numRows
        # Resolve label
        g_labels[id+1] = find_root(g_labels, g_labels[id+1])
    end

    return nothing
end

function label_reduction(g_labels, g_image, numCols, numRows)
    # Calculate index
    ix = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
    iy = threadIdx().y + (blockIdx().y - 1) * blockDim().y - 1

    # Check thread range
    if (ix < numCols) && (iy < numRows)
        # Compare image values
        pyx = g_image[iy*numCols+ix+1]
        nym1x = (iy > 0) ? (pyx == g_image[(iy-1)*numCols+ix+1]) : false

        if !nym1x
            # Neighbouring values
            nym1xm1 = ((iy > 0) && (ix > 0)) ? (pyx == g_image[(iy-1)*numCols+ix-1+1]) : false
            nyxm1 = (ix > 0) ? (pyx == g_image[(iy)*numCols+ix-1+1]) : false
            nym1xp1 = ((iy > 0) && (ix < numCols - 1)) ? (pyx == g_image[(iy-1)*numCols+ix+1+1]) : false

            if nym1xp1
                # Check criticals
                # There are three cases that need a reduction
                if (nym1xm1 && nyxm1) || (nym1xm1 && !nyxm1)
                    # Get labels
                    label1 = g_labels[(iy)*numCols+ix+1]
                    label2 = g_labels[(iy-1)*numCols+ix+1+1]

                    # Reduction
                    reduction(g_labels, label1, label2)
                end

                if !nym1xm1 && nyxm1
                    # Get labels
                    label1 = g_labels[(iy)*numCols+ix+1]
                    label2 = g_labels[(iy)*numCols+ix-1+1]

                    # Reduction
                    reduction(g_labels, label1, label2)
                end
            end
        end
    end

    return nothing
end

function resolve_background(g_labels, g_image, width, height)
    # Calculate index
    ix = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    iy = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    # Check thread range
    # if id <= width * height
    if (ix <= width) && (iy <= height)
        # Resolve label
        g_labels[iy, ix] = (g_image[iy, ix] > 0) ? g_labels[iy, ix] + 1 : 0
    end

    return nothing
end

"""
    cu_connected_component_labeling(input_img)

8-way connected component labeling on binary image based on the article by Playne and Hawick https://ieeexplore.ieee.org/document/8274991 and the implementation by FolkeV https://github.com/FolkeV/CUDA_CCL. It works using the CUDA.jl package and NVIDIA GPUs.

# Arguments
- `input_img::CuArray{Float32, 2}`: Input binary image. 

# Returns
- `output_img::CuArray{UInt32, 2}`: Output labeled image.

"""
function cu_connected_component_labeling(input_img)
    @assert length(input_img) <= 2^32 - 1 "Image is too large. Maximum length is 2^32-1."
    output_img = CUDA.zeros(UInt32, size(input_img))

    height, width = size(input_img)

    block = (4, 32)
    grid = cld.((width, height), block)

    # Initialize labels
    @cuda threads = block blocks = grid init_labels(output_img, input_img, width, height)

    # Analysis
    @cuda threads = block blocks = grid resolve_labels(output_img, width, height)

    # Label reduction
    @cuda threads = block blocks = grid label_reduction(output_img, input_img, width, height)

    # Analysis
    @cuda threads = block blocks = grid resolve_labels(output_img, width, height)

    # Force background to have level 0
    @cuda threads = block blocks = grid resolve_background(output_img, input_img, width, height)

    return output_img
end

function count_labels(labels)
    components = 0
    for i in 0:length(labels)-1
        if labels[i+1] == i + 1
            components += 1
        end
    end
    return components
end

# Post processing
function find_indices(labels, indices, length)
    id = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if id <= length
        if labels[id] == id
            indices[id] = id
        end
    end
    return nothing
end

function cu_find_valid_labels(labels::CuArray{UInt32,2})
    d_indices = CUDA.zeros(UInt32, length(labels))
    @cuda threads = 1024 blocks = cld(length(labels), 1024) find_indices(labels, d_indices, length(labels))
    return Array(findall(!iszero, d_indices))
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