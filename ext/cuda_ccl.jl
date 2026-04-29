# CUDA 8-way Connected Component Labelling
# Please refer to: https://github.com/FolkeV/CUDA_CCL

@inline function find_root(labels, label)
    next = labels[label+1]

    while label != next
        label = next
        next = labels[label+1]
    end

    return label
end

@inline function reduction(g_labels, label1, label2)
    next1 = (label1 != label2) ? g_labels[label1+1] : 1
    next2 = (label1 != label2) ? g_labels[label2+1] : 1

    while (label1 != label2) && (label1 != next1)
        label1 = next1
        next1 = g_labels[label1+1]
    end

    while (label1 != label2) && (label2 != next2)
        label2 = next2
        next2 = g_labels[label2+1]
    end

    label3 = 0
    while label1 != label2
        if label1 < label2
            tmp = label1
            label1 = label2
            label2 = tmp
        end
        label3 = CUDA.@atomic g_labels[label1+1] = min(g_labels[label1+1], label2)
        label1 = (label1 == label3) ? label2 : label3
    end

    return label1
end

function init_labels(g_labels, g_image, numCols, numRows)
    ix = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
    iy = threadIdx().y + (blockIdx().y - 1) * blockDim().y - 1

    if (ix < numCols) && (iy < numRows)
        pyx = g_image[iy*numCols+ix+1]

        nym1x = (iy > 0) ? (pyx == g_image[(iy-1)*numCols+ix+1]) : false
        nyxm1 = (ix > 0) ? (pyx == g_image[(iy)*numCols+ix-1+1]) : false
        nym1xm1 = ((iy > 0) && (ix > 0)) ? (pyx == g_image[(iy-1)*numCols+ix-1+1]) : false
        nym1xp1 = ((iy > 0) && (ix < numCols - 1)) ? (pyx == g_image[(iy-1)*numCols+ix+1+1]) : false

        label = (nyxm1) ? iy * numCols + ix - 1 : iy * numCols + ix
        label = (nym1xp1) ? (iy - 1) * numCols + ix + 1 : label
        label = (nym1x) ? (iy - 1) * numCols + ix : label
        label = (nym1xm1) ? (iy - 1) * numCols + ix - 1 : label

        @inbounds g_labels[iy*numCols+ix+1] = label
    end

    return nothing
end

function resolve_labels(g_labels, numCols, numRows)
    ix = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
    iy = threadIdx().y + (blockIdx().y - 1) * blockDim().y - 1
    id = ix + iy * numCols

    if id < numCols * numRows
        g_labels[id+1] = find_root(g_labels, g_labels[id+1])
    end

    return nothing
end

function label_reduction(g_labels, g_image, numCols, numRows)
    ix = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
    iy = threadIdx().y + (blockIdx().y - 1) * blockDim().y - 1

    if (ix < numCols) && (iy < numRows)
        pyx = g_image[iy*numCols+ix+1]
        nym1x = (iy > 0) ? (pyx == g_image[(iy-1)*numCols+ix+1]) : false

        if !nym1x
            nym1xm1 = ((iy > 0) && (ix > 0)) ? (pyx == g_image[(iy-1)*numCols+ix-1+1]) : false
            nyxm1 = (ix > 0) ? (pyx == g_image[(iy)*numCols+ix-1+1]) : false
            nym1xp1 = ((iy > 0) && (ix < numCols - 1)) ? (pyx == g_image[(iy-1)*numCols+ix+1+1]) : false

            if nym1xp1
                if (nym1xm1 && nyxm1) || (nym1xm1 && !nyxm1)
                    label1 = g_labels[(iy)*numCols+ix+1]
                    label2 = g_labels[(iy-1)*numCols+ix+1+1]
                    reduction(g_labels, label1, label2)
                end

                if !nym1xm1 && nyxm1
                    label1 = g_labels[(iy)*numCols+ix+1]
                    label2 = g_labels[(iy)*numCols+ix-1+1]
                    reduction(g_labels, label1, label2)
                end
            end
        end
    end

    return nothing
end

function resolve_background(g_labels, g_image, width, height)
    ix = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    iy = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if (ix <= width) && (iy <= height)
        g_labels[iy, ix] = (g_image[iy, ix] > 0) ? g_labels[iy, ix] + 1 : 0
    end

    return nothing
end

"""
    cu_connected_component_labeling(input_img)

8-way connected component labeling on binary image based on the article by Playne and Hawick https://ieeexplore.ieee.org/document/8274991 and the implementation by FolkeV https://github.com/FolkeV/CUDA_CCL. It works using the CUDA.jl package and NVIDIA GPUs.
"""
function cu_connected_component_labeling(input_img::CuArray)
    _assert_cuda_functional(:cu_connected_component_labeling)
    @assert length(input_img) <= 2^32 - 1 "Image is too large. Maximum length is 2^32-1."
    output_img = CUDA.zeros(UInt32, size(input_img))

    height, width = size(input_img)

    block = (4, 32)
    grid = cld.((width, height), block)

    @cuda threads=block blocks=grid init_labels(output_img, input_img, width, height)
    @cuda threads=block blocks=grid resolve_labels(output_img, width, height)
    @cuda threads=block blocks=grid label_reduction(output_img, input_img, width, height)
    @cuda threads=block blocks=grid resolve_labels(output_img, width, height)
    @cuda threads=block blocks=grid resolve_background(output_img, input_img, width, height)

    return output_img
end

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
    _assert_cuda_functional(:cu_find_valid_labels)
    d_indices = CUDA.zeros(UInt32, length(labels))
    @cuda threads=1024 blocks=cld(length(labels), 1024) find_indices(labels, d_indices, length(labels))
    return Array(findall(!iszero, d_indices))
end
