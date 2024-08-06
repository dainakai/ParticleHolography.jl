using CUDA
using FixedPointNumbers
using Statistics
using ImageFiltering
using HistogramThresholding
using UUIDs

export particle_bounding_boxes, particle_coordinates, particle_coor_diams

function _dilate_3d!(dilated, vol, datlen, slices)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    z = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if x>1 && x<datlen && y>1 && y<datlen && z>0 && z<=slices
        @inbounds dilated[y,x,z] = vol[y-1,x-1,z] || vol[y-1,x,z] || vol[y-1,x+1,z] || vol[y,x-1,z] || vol[y,x,z] || vol[y,x+1,z] || vol[y+1,x-1,z] || vol[y+1,x,z] || vol[y+1,x+1,z]
    end
    return nothing
end

function cu_dilate(vol::CuArray{Bool,3})
    datlen = size(vol, 1)
    slices = size(vol, 3)
    dilated = CUDA.fill(false, (datlen, datlen, slices))
    threads = (32,32,1)
    blocks = cld.((datlen, datlen, slices), threads)
    @cuda threads=threads blocks=blocks _dilate_3d!(dilated, vol, datlen, slices)
    return dilated
end


"""
    particle_bounding_boxes(d_bin_vol)

Detects the particles in a binary volume and returns the bounding boxes of the particles.
This function performs three-dimensional element connected labeling on binary volumes. However, please note that it does not perform strict adjacent connectivity in the optical axis (z) direction. Strict adjacent connectivity in the optical axis direction may result in artifacts not being connected, potentially leading to the detection of many ghost particles.
This function assumes that two particles never overlap at exactly the same X-Y coordinates. All elements that overlap in X-Y coordinates are considered connected. Therefore, this method is not suitable for accurate position detection of particles that overlap in X-Y coordinates.
Additionally, this processing may have some effects, such as slightly elongating the bounding box of particles in the optical axis direction. However, this has minimal impact on the accuracy of particle position detection.

# Arguments
- `d_bin_vol::CuArray{Bool, 3}`: The binary volume of reconstructed holographic volume. Binarization can be done by thresholding the reconstructed volume. `true` values represent the particles and neighboring voxels and vice versa.

# Returns
- `Dict{UUID, Vector{Int}}`: The bounding boxes of the particles.
"""
function particle_bounding_boxes(d_bin_vol::CuArray{Bool, 3})
    @views labeledimg = cu_connected_component_labeling(d_bin_vol[:,:,1])
    valid_labels = cu_find_valid_labels(labeledimg)
    bounding_boxes = get_bounding_rectangles(Array(labeledimg), valid_labels)
    particle_bbs = gen_particle_neighborhoods(bounding_boxes, 1)

    slices = size(d_bin_vol, 3)
    if slices > 1
        for idx in 2:slices
            @views labeledimg = cu_connected_component_labeling(d_bin_vol[:,:,idx])
            valid_labels = cu_find_valid_labels(labeledimg)
            bounding_boxes = get_bounding_rectangles(Array(labeledimg), valid_labels)
            update_particle_neighborhoods!(particle_bbs, bounding_boxes, idx)
        end
    end

    finalize_particle_neighborhoods!(particle_bbs)
    return particle_bbs
end

"""
    tamura(arr)

Calculates the Tamura coefficient of an array. The Tamura coefficient is defined as the standard deviation divided by the mean of the array. Please refer to the use in digital holography https://doi.org/10.1364/OL.36.001945

# Arguments
- `arr::Array{Float32, 2}`: The array for which the Tamura coefficient is calculated.

# Returns
- `Float32`: The Tamura coefficient of the array.
"""
function tamura(arr::AbstractArray{<:AbstractFloat,2})
    return std(arr)/mean(arr)
end

function depth_profile(f::Function, bounding_rect_3d::AbstractArray{<:Union{AbstractFloat, Complex}, 3})
    return [f(bounding_rect_3d[:,:,i]) for i in axes(bounding_rect_3d, 3)]
end

function getcenterfromslice(arr::AbstractArray{<:AbstractFloat, 2})
    x = 0.0
    y = 0.0
    for i in axes(arr, 1)
        for j in axes(arr, 2)
            y += i * arr[i,j]
            x += j * arr[i,j]
        end
    end
    x = x / sum(arr)
    y = y / sum(arr)
    return (x, y)
end


"""
    particle_coordinates(particle_bbs, d_vol; depth_metrics = tamura, profile_smoothing_kernel = Kernel.gaussian(5,))

Calculates the coordinates of the particles in the reconstructed volume with the bounding boxe dictionary. The depth of the particles is the maximum of the profile that is calculated using the `depth_metrics` function at each slice of the bounding box. The profile is then smoothed using the `profile_smoothing_kernel`. The x and y coordinates are calculated by finding the center of mass of the slice with the detected depth. The low pass filtered volume would be better for coordinate detection.

# Arguments
- `particle_bbs::Dict{UUID, Vector{Int}}`: The bounding boxes of the particles.
- `d_vol::CuArray{N0f8, 3}`: The reconstructed volume.
- `depth_metrics::Function = tamura`: The function that calculates the depth profile of the particles.
- `profile_smoothing_kernel = Kernel.gaussian((5,))`: The kernel used for smoothing the depth profile.

# Returns
- `Dict{UUID, Vector{Float32}}`: The coordinates of the particles.
"""
function particle_coordinates(particle_bbs::Dict{UUID, Vector{Int}}, d_vol::CuArray{N0f8, 3}; depth_metrics::Function = tamura, profile_smoothing_kernel = Kernel.gaussian((5,)))
    particle_coords = Dict{UUID, Vector{Float32}}()
    for (key, value) in particle_bbs
        @views subvol = Float32.(d_vol[value[2]:value[5], value[1]:value[4], value[3]:value[6]])
        zmetric = depth_profile(depth_metrics, subvol)
        imfilter!(zmetric, zmetric, profile_smoothing_kernel)
        z = argmax(zmetric)
        (x, y) = getcenterfromslice(Array(subvol[:,:,z]))
        particle_coords[key] = [x + value[1] - 1, y + value[2] - 1, z + value[3] - 1]
    end
    return particle_coords
end

function equivalent_diameter(arr::AbstractArray{<:AbstractFloat, 2})
    t = find_threshold(arr, algorithm::HistogramThresholding.Otsu())
    newarr = arr .<= t
    return 2 * sqrt(sum(newarr)/π)
end

"""
    particle_coor_diams(particle_bbs, d_vol; d_lpf_vol = nothing, depth_metrics = tamura, profile_smoothing_kernel = Kernel.gaussian(5,), diameter_metrics = equivalent_diameter)

Calculates the coordinates and diameters of the particles in the reconstructed volume with the bounding boxe dictionary. The depth of the particles is the maximum of the profile that is calculated using the `depth_metrics` function at each slice of the bounding box. The profile is then smoothed using the `profile_smoothing_kernel`. The x and y coordinates are calculated by finding the center of mass of the slice with the detected depth. The low pass filtered volume would be better for coordinate detection. The diameter of the particles is calculated using the `diameter_metrics` function. If the low pass filtered volume is provided, the coordinate is calculated using the low pass filtered volume.

# Arguments
- `particle_bbs::Dict{UUID, Vector{Int}}`: The bounding boxes of the particles.
- `d_vol::CuArray{N0f8, 3}`: The reconstructed volume.
- `d_lpf_vol::CuArray{N0f8, 3} = nothing`: The low pass filtered volume.
- `depth_metrics::Function = tamura`: The function that calculates the depth profile of the particles.
- `profile_smoothing_kernel = Kernel.gaussian((5,))`: The kernel used for smoothing the depth profile.
- `diameter_metrics::Function = equivalent_diameter`: The function that calculates the diameter of the particles.

# Returns
- `Dict{UUID, Vector{Float32}}`: The coordinates and diameters of the particles.
"""
function particle_coor_diams(particle_bbs::Dict{UUID, Vector{Int}}, d_vol::CuArray{N0f8, 3}; d_lpf_vol::CuArray{N0f8, 3} = nothing, depth_metrics::Function = tamura, profile_smoothing_kernel = Kernel.gaussian((5,)), diameter_metrics::Function = equivalent_diameter)
    particle_coords = Dict{UUID, Vector{Float32}}()
    for (key, value) in particle_bbs
        @views subvol = Float32.(d_vol[value[2]:value[5], value[1]:value[4], value[3]:value[6]])
        if !isnothing(d_lpf_vol)
            @views subvol_lpf = Float32.(d_lpf_vol[value[2]:value[5], value[1]:value[4], value[3]:value[6]])
            zmetric = depth_profile(depth_metrics, subvol_lpf)
            imfilter!(zmetric, zmetric, profile_smoothing_kernel)
            z = argmax(zmetric)
            (x, y) = getcenterfromslice(Array(subvol_lpf[:,:,z]))
            diam = diameter_metrics(subvol[:,:,z])
            particle_coords[key] = [x + value[1] - 1, y + value[2] - 1, z + value[3] - 1, diam]
        else
            zmetric = depth_profile(depth_metrics, subvol)
            imfilter!(zmetric, zmetric, profile_smoothing_kernel)
            z = argmax(zmetric)
            (x, y) = getcenterfromslice(Array(subvol[:,:,z]))
            diam = diameter_metrics(subvol[:,:,z])
            particle_coords[key] = [x + value[1] - 1, y + value[2] - 1, z + value[3] - 1, diam]
        end
    end
    return particle_coords
end

