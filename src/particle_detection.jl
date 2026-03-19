using CUDA
using FixedPointNumbers
using Statistics
using ImageFiltering
using HistogramThresholding
using UUIDs

export particle_bounding_boxes, particle_coordinates, particle_coor_diams
export particle_bounding_boxes_3d, cu_dilate

# COV_EXCL_START
function _cu_dilate_3d!(dilated, vol, datlen, slices)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    z = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if x>1 && x<datlen && y>1 && y<datlen && z>0 && z<=slices
        @inbounds dilated[y,x,z] = vol[y-1,x-1,z] || vol[y-1,x,z] || vol[y-1,x+1,z] || vol[y,x-1,z] || vol[y,x,z] || vol[y,x+1,z] || vol[y+1,x-1,z] || vol[y+1,x,z] || vol[y+1,x+1,z]
    end
    return nothing
end
# COV_EXCL_STOP

"""
    cu_dilate(vol; blocksize=32)

Perform a 3D dilation on the reconstructed image stack `vol` and return the dilated stack.

# Arguments
- `vol::CuArray{Bool,3}`: The input 3D binary volume to be dilated. It can be the result of thresholding a reconstructed volume.
- `blocksize::Int`: The block size for CUDA kernel execution. Default is 32.
"""
function cu_dilate(vol::CuArray{Bool,3}; blocksize=32)
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
This function performs three-dimensional element connected labeling on binary volumes. However, please note that it does not perform strict adjacent connectivity in the optical axis (z) direction. Strict adjacent connectivity in the optical axis direction may result in artifacts not being connected, potentially leading to the detection of many ghost particles.
This function assumes that two particles never overlap at exactly the same X-Y coordinates. All elements that overlap in X-Y coordinates are considered connected. Therefore, this method is not suitable for accurate position detection of particles that overlap in X-Y coordinates.
Additionally, this processing may have some effects, such as slightly elongating the bounding box of particles in the optical axis direction. However, this has minimal impact on the accuracy of particle position detection.

# Arguments
- `d_bin_vol::CuArray{Bool, 3}`: The binary volume of reconstructed holographic volume. Binarization can be done by thresholding the reconstructed volume. `true` values represent the particles and neighboring voxels and vice versa.

# Returns
- `Dict{UUID, Vector{Int}}`: The bounding boxes of the particles.
"""
function particle_bounding_boxes(d_bin_vol::CuArray{Bool,3})
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

function _validate_particle_volume_pair(d_vol::CuArray{T,3}, d_lpf_vol) where {T}
    _validate_particle_volume_type(T, "d_vol")
    if isnothing(d_lpf_vol)
        return nothing
    end

    if !(d_lpf_vol isa CuArray{<:Any,3})
        throw(ArgumentError("`d_lpf_vol` must be `nothing` or `CuArray{<:Any,3}`. Got $(typeof(d_lpf_vol))."))
    end

    _validate_particle_volume_type(eltype(d_lpf_vol), "d_lpf_vol")
    if size(d_lpf_vol) != size(d_vol)
        throw(ArgumentError("`d_lpf_vol` must have the same size as `d_vol`. Got $(size(d_lpf_vol)) and $(size(d_vol))."))
    end

    return nothing
end


"""
    particle_coordinates(particle_bbs, d_vol; depth_metrics = tamura, profile_smoothing_kernel = Kernel.gaussian(5,))

Calculates the coordinates of the particles in the reconstructed volume with the bounding boxe dictionary. The depth of the particles is the maximum of the profile that is calculated using the `depth_metrics` function at each slice of the bounding box. The profile is then smoothed using the `profile_smoothing_kernel`. The x and y coordinates are calculated by finding the center of mass of the slice with the detected depth. The low pass filtered volume would be better for coordinate detection. The extracted subvolume is converted to `Float32` internally before evaluating metrics.

# Arguments
- `particle_bbs::Dict{UUID, Vector{Int}}`: The bounding boxes of the particles.
- `d_vol::CuArray{T, 3}`: The reconstructed volume. Real-valued voxel types such as `N0f8`, `Float32`, `UInt8`, and `UInt16` are supported. `Bool` and complex inputs are rejected.
- `depth_metrics::Function = tamura`: The function that calculates the depth profile of the particles.
- `profile_smoothing_kernel = Kernel.gaussian((5,))`: The kernel used for smoothing the depth profile.

# Returns
- `Dict{UUID, Vector{Float32}}`: The coordinates of the particles.
"""
function particle_coordinates(particle_bbs::Dict{UUID,Vector{Int}}, d_vol::CuArray{T,3}; depth_metrics::Function=tamura, profile_smoothing_kernel=Kernel.gaussian((5,))) where {T}
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
- `d_vol::CuArray{T, 3}`: The reconstructed volume. Real-valued voxel types such as `N0f8`, `Float32`, `UInt8`, and `UInt16` are supported. `Bool` and complex inputs are rejected.

# Optional arguments
- `d_lpf_vol = nothing`: The low pass filtered volume. When provided, it must be a 3D `CuArray` with a real-valued element type and the same size as `d_vol`.

# Optional keyword arguments
- `depth_metrics::Function = tamura`: The function that calculates the depth profile of the particles.
- `profile_smoothing_kernel = Kernel.gaussian((5,))`: The kernel used for smoothing the depth profile.
- `diameter_metrics::Function = equivalent_diameter`: The function that calculates the diameter of the particles.

# Returns
- `Dict{UUID, Vector{Float32}}`: The coordinates and diameters of the particles.
"""
function particle_coor_diams(particle_bbs::Dict{UUID,Vector{Int}}, d_vol::CuArray{T,3}, d_lpf_vol=nothing; depth_metrics::Function=tamura, profile_smoothing_kernel=Kernel.gaussian((5,)), diameter_metrics::Function=equivalent_diameter) where {T}
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
