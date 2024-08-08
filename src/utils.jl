using Images
using StatsBase
using ProgressMeter
using Colors
using FixedPointNumbers
using JSON
using FileIO

export load_gray2float, find_external_contours, draw_contours!, make_background, pad_with_mean, dictsave, dictload

"""
    load_gray2float(path)

Load a grayscale image from a file and return it as a Array{Float32, 2} array.

# Arguments
- `path::String`: The path to the image file.

# Returns
- `Array{Float32, 2}`: The image as a Float32 array.
"""
function load_gray2float(path::String)
    out = Float32.(channelview(Gray.(load(path))))
end


###################################   Particle Detection   ###################################
# See the document on the following page for details
# https://juliaimages.org/latest/examples/contours/contour_detection/
##############################################################################################
#              N          NE      E       SE      S       SW        W      NW
# direction between two pixels

# rotate direction clockwise
function _clockwise(dir)
    return (dir)%8 + 1
end

# rotate direction counterclockwise
function _counterclockwise(dir)
    return (dir+6)%8 + 1
end

# move from current pixel to next in given direction
function _move(pixel, image, dir, dir_delta)
    newp = pixel + dir_delta[dir]
    height, width = size(image)
    if (0 < newp[1] <= height) &&  (0 < newp[2] <= width)
        if image[newp]!=0
            return newp
        end
    end
    return CartesianIndex(0, 0)
end

# finds direction between two given pixels
function _from_to(from, to, dir_delta)
    delta = to-from
    return findall(x->x == delta, dir_delta)[1]
end

function _detect_move(image, p0, p2, nbd, border, done, dir_delta)
    dir = _from_to(p0, p2, dir_delta)
    moved = _clockwise(dir)
    p1 = CartesianIndex(0, 0)
    while moved != dir ## 3.1
        newp = _move(p0, image, moved, dir_delta)
        if newp[1]!=0
            p1 = newp
            break
        end
        moved = _clockwise(moved)
    end

    if p1 == CartesianIndex(0, 0)
        return
    end

    p2 = p1 ## 3.2
    p3 = p0 ## 3.2
    done .= false
    while true
        dir = _from_to(p3, p2, dir_delta)
        moved = _counterclockwise(dir)
        p4 = CartesianIndex(0, 0)
        done .= false
        while true ## 3.3
            p4 = _move(p3, image, moved, dir_delta)
            if p4[1] != 0
                break
            end
            done[moved] = true
            moved = _counterclockwise(moved)
        end
        push!(border, p3) ## 3.4
        if p3[1] == size(image, 1) || done[3]
            image[p3] = -nbd
        elseif image[p3] == 1
            image[p3] = nbd
        end

        if (p4 == p0 && p3 == p1) ## 3.5
            break
        end
        p2 = p3
        p3 = p4
    end
end

"""
    find_external_contours(image)

Finds non-hole contours in binary images. This function is excuted on the CPU. Equivalent to CV_RETR_EXTERNAL and CV_CHAIN_APPROX_NONE modes of the findContours() function provided in OpenCV.

# Arguments
- `image`: The binary image. the image should be a 2D array of 0 and 1.

# Returns
- `Vector{Vector{CartesianIndex}}`: A vector of contours. Each contour is a vector of CartesianIndex.
"""
function find_external_contours(image)
    nbd = 1
    lnbd = 1
    image = Float64.(image)
    contour_list =  Vector{typeof(CartesianIndex[])}()
    done = [false, false, false, false, false, false, false, false]

    # Clockwise Moore neighborhood.
    dir_delta = [CartesianIndex(-1, 0) , CartesianIndex(-1, 1), CartesianIndex(0, 1), CartesianIndex(1, 1), CartesianIndex(1, 0), CartesianIndex(1, -1), CartesianIndex(0, -1), CartesianIndex(-1,-1)]

    height, width = size(image)

    for i=1:height
        lnbd = 1
        for j=1:width
            fji = image[i, j]
            is_outer = (image[i, j] == 1 && (j == 1 || image[i, j-1] == 0)) ## 1 (a)
            #is_hole = (image[i, j] >= 1 && (j == width || image[i, j+1] == 0))

            if is_outer #|| is_hole
                # 2
                border = CartesianIndex[]

                from = CartesianIndex(i, j)

                if is_outer
                    nbd += 1
                    from -= CartesianIndex(0, 1)

                else
                    nbd += 1
                    if fji > 1
                        lnbd = fji
                    end
                    from += CartesianIndex(0, 1)
                end

                p0 = CartesianIndex(i,j)
                _detect_move(image, p0, from, nbd, border, done, dir_delta) ## 3
                if isempty(border) ##TODO
                    push!(border, p0)
                    image[p0] = -nbd
                end
                push!(contour_list, border)
            end
            if fji != 0 && fji != 1
                lnbd = abs(fji)
            end

        end
    end

    return contour_list
end

# a contour is a vector of 2 int arrays
function _draw_contour!(image, color, contour)
    for ind in contour
        image[ind] = color
    end
end

function draw_contours!(image, color, contours)
    for cnt in contours
        _draw_contour!(image, color, cnt)
    end
end

"""
    make_background(pathlist; mode=:mode)

Make a background image from a list of image paths. The background image is calculated by taking the mean or mode of the images in the list. The default mode is :mode.

# Arguments
- `pathlist::Vector{String}`: A list of image paths. `glob()` can be used to generate this list.
- `mode::Symbol`: The mode to use for calculating the background. Options are :mean or :mode. Default is :mode.

# Returns
- `Array{Float64, 2}`: The background image.
"""
function make_background(pathlist::Vector{String}; mode=:mode)
    if mode == :mean
        background = zeros(Float64, size(load_gray2float(pathlist[1])))
        @showprogress desc="Background calculating..." for path in pathlist
            background .= background .+ Float64.(load_gray2float(path))
        end
        background /= length(pathlist)
        return background

    elseif mode == :mode
        votevol = zeros(Int, (256,size(load_gray2float(pathlist[1]))...))
        datlen = size(load(pathlist[1]))[1]
        @showprogress desc="Background calculating..." for path in pathlist
            img = Int.(reinterpret.(UInt8, channelview(Gray.(load(path)))))
            for x in 1:datlen
                for y in 1:datlen
                    votevol[img[y,x]+1,y,x] += 1
                end
            end
        end
        background = [(value[1]-1.0)./255.0 for value in argmax(votevol, dims=1)[1,:,:]]
        return background
    end
end

function pad_with_mean(img, padsize)
    @assert padsize > size(img, 1) && padsize > size(img, 2) "Padsize should be larger than the image size. padsize: $padsize, img size: $(size(img))"
    meanval = mean(img)
    output = fill(meanval, (padsize, padsize))
    output[div(padsize,2)-div(size(img,1),2)+1:div(padsize,2)-div(size(img,1),2)+size(img,1), div(padsize,2)-div(size(img,2),2)+1:div(padsize,2)-div(size(img,2),2)+size(img,2)] .= img
    return output
end


"""
    dictsave(filename, dict)

Save a particle dictionary to a file in JSON format. The dictionary should have UUID keys and values as Vector{Float32}, which includes the coordinates (and diameters) of the particles.

# Arguments
- `filename::String`: The path to the file.
- `dict::Dict`: The dictionary to save.

# Returns
- `nothing`
"""
function dictsave(filename, dict::Dict)
    open(filename, "w") do io
        JSON.print(io, dict)
    end
    return nothing
end

"""
    dictload(filename)

Load a particle dictionary from a file in JSON format. The dictionary should have UUID keys and values as Vector{Float32}, which includes the coordinates (and diameters) of the particles.

# Arguments
- `filename::String`: The path to the file.

# Returns
- `Dict`: The loaded dictionary.
"""
function dictload(filename)
    data = JSON.parsefile(filename)
    dict = Dict{UUID, Vector{Float32}}()
    for (key, value) in data
        dict[UUID(key)] = value
    end
    return dict
end