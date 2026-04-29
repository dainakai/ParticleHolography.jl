# COV_EXCL_START
function _cu_dilate_3d!(dilated, vol, datlen, slices)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    z = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if x > 1 && x < datlen && y > 1 && y < datlen && z > 0 && z <= slices
        @inbounds dilated[y,x,z] = vol[y-1,x-1,z] || vol[y-1,x,z] || vol[y-1,x+1,z] || vol[y,x-1,z] || vol[y,x,z] || vol[y,x+1,z] || vol[y+1,x-1,z] || vol[y+1,x,z] || vol[y+1,x+1,z]
    end
    return nothing
end
# COV_EXCL_STOP

"""
    cu_dilate(vol; blocksize=32)

Perform a 3D dilation on the reconstructed image stack `vol` and return the dilated stack.
"""
function cu_dilate(vol::CuArray{Bool,3}; blocksize=32)
    _assert_cuda_functional(:cu_dilate)
    datlen = size(vol, 1)
    slices = size(vol, 3)
    dilated = CUDA.fill(false, (datlen, datlen, slices))
    threads = (blocksize, blocksize, 1)
    blocks = cld.((datlen, datlen, slices), threads)
    @cuda threads=threads blocks=blocks _cu_dilate_3d!(dilated, vol, datlen, slices)
    return dilated
end

"""
    particle_bounding_boxes(d_bin_vol)

Detects the particles in a binary volume and returns the bounding boxes of the particles.
"""
function particle_bounding_boxes(d_bin_vol::CuArray{Bool,3})
    _assert_cuda_functional(:particle_bounding_boxes)
    @views labeledimg = cu_connected_component_labeling(d_bin_vol[:, :, 1])
    valid_labels = cu_find_valid_labels(labeledimg)
    bounding_boxes = get_bounding_rectangles(Array(labeledimg), valid_labels)
    particle_bbs = gen_particle_neighborhoods(bounding_boxes, 1)

    slices = size(d_bin_vol, 3)
    if slices > 1
        for idx in 2:slices
            @views labeledimg = cu_connected_component_labeling(d_bin_vol[:, :, idx])
            valid_labels = cu_find_valid_labels(labeledimg)
            bounding_boxes = get_bounding_rectangles(Array(labeledimg), valid_labels)
            update_particle_neighborhoods!(particle_bbs, bounding_boxes, idx)
        end
    end

    finalize_particle_neighborhoods!(particle_bbs)
    return particle_bbs
end

function particle_bounding_boxes_3d(d_bin_vol::CuArray{Bool,3})
    _assert_cuda_functional(:particle_bounding_boxes_3d)
    @views labeledimg = cu_connected_component_labeling(d_bin_vol[:, :, 1])
    valid_labels = cu_find_valid_labels(labeledimg)
    bounding_boxes = get_bounding_rectangles(Array(labeledimg), valid_labels)
    particle_bbs = gen_particle_neighborhoods(bounding_boxes, 1)

    slices = size(d_bin_vol, 3)
    if slices > 1
        for idx in 2:slices
            @views labeledimg = cu_connected_component_labeling(d_bin_vol[:, :, idx])
            valid_labels = cu_find_valid_labels(labeledimg)
            bounding_boxes = get_bounding_rectangles(Array(labeledimg), valid_labels)
            update_particle_neighborhoods3d!(particle_bbs, bounding_boxes, idx)
        end
    end

    finalize_particle_neighborhoods!(particle_bbs)
    return particle_bbs
end
