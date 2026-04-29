using Statistics

export quadratic_distortion_correction, get_distortion_coefficients

"""
    quadratic_distortion_correction(img, coefa)

Correct the quadratic distortion in the grayscale image `img` using the coefficients `coefa`. `img` have to be square, and `coefa` have to be a vector of 12 coefficients.

# Arguments
- `img::Array{<:AbstractFloat,2}`: The image to correct.
- `coefa::Vector{<:AbstractFloat}`: The coefficients to correct the distortion.

# Returns
- `Array{AbstractFloat,2}`: The corrected image.
"""
function quadratic_distortion_correction(img::Array{<:AbstractFloat,2}, coefa::Vector{<:AbstractFloat})
    @assert size(img)[1] == size(img)[2] "The image must be square. Got $(size(img))."
    @assert length(coefa) == 12 "The coefficients must be 12. Got $(length(coefa))."

    n = size(img)[1]
    bkg = mean(img)
    refX = Array{Int}(undef, n * n)
    refY = Array{Int}(undef, n * n)
    out = Array{Float64}(undef, n, n)

    for i in 1:n
        for j in 1:n
            refX[(i-1)*n+j] = Int(round(coefa[1] + coefa[2] * j + coefa[3] * i + coefa[4] * j^2 + coefa[5] * i * j + coefa[6] * i^2))
            refY[(i-1)*n+j] = Int(round(coefa[7] + coefa[8] * j + coefa[9] * i + coefa[10] * j^2 + coefa[11] * i * j + coefa[12] * i^2))
        end
    end

    for i in 1:n
        for j in 1:n
            if (refX[(i-1)*n+j] >= 1) && (refX[(i-1)*n+j] <= n) && (refY[(i-1)*n+j] >= 1) && (refY[(i-1)*n+j] <= n)
                out[i, j] = img[refY[(i-1)*n+j], refX[(i-1)*n+j]]
            else
                out[i, j] = bkg
            end
        end
    end
    return out
end

"""
    modified_Cholesky_decomposition(A)

This function decomposes the target matrix A to A = LDL^T. The diagonal components of the return matrix are the inverse of D, and the lower left components correspond to L.
対象行列Aを A = LDL^T に変換します。戻り値行列の対角成分は D の逆数、左下成分は L に一致します。

# Arguments
- `A::Array{<:AbstractFloat,2}`: The target matrix.

# Returns
- `Array{AbstractFloat,2}`: The decomposed matrix.
"""
function modified_Cholesky_decomposition(A::Array{<:AbstractFloat,2})
    ndims(A) == 2 ? 1 : error("modified_Cholesky_decomposition : Input matrix is not in 2 dims.")
    n = size(A)[1]
    n == size(A)[2] ? 1 : error("modified_Cholesky_decomposition : Input matrix is not square.")
    # println("Input : $(n)*$(n) A")
    matA = copy(A)
    vecw = Array{AbstractFloat}(undef, n)
    for j in 1:n
        for i in 1:j-1
            vecw[i] = matA[j, i]
            for k in 1:i-1
                vecw[i] -= matA[i, k] * vecw[k]
            end
            matA[j, i] = vecw[i] * matA[i, i]
        end
        t = matA[j, j]
        for k in 1:j-1
            t -= matA[j, k] * vecw[k]
        end
        matA[j, j] = 1.0 / t
    end
    # matA の対角成分は matD の逆数。matA_ji (i<j) は matL_ji。
    return matA
end

"""
    simultanious_equation_solver(chlskyMat,yacob,errorArray)

Solve the equation `chlskyMat` x = - `yacob` `errorArray` using the Cholesky decomposition matrix `chlskyMat`, Jacobian `yacob`, and error vector `errorArray`.

# Arguments
- `chlskyMat::Array{<:AbstractFloat,2}`: The Cholesky decomposition matrix.
- `yacob::Array{<:AbstractFloat,2}`: The Jacobian matrix.
- `errorArray::Array{<:AbstractFloat,1}`: The error vector.

# Returns
- `Array{AbstractFloat,1}`: The solution vector.
"""
function simultanious_equation_solver(chlsky_mat::Array{<:AbstractFloat,2}, yacob::Array{<:AbstractFloat,2}, error_array::Array{<:AbstractFloat,1})
    vecB = -transpose(yacob) * error_array
    n = size(vecB)[1]

    matL = chlsky_mat


    vecX = zeros(n)
    vecY = zeros(n)

    for k in 1:n
        vecY[k] = vecB[k]
        for i in 1:k-1
            vecY[k] -= matL[k, i] * vecY[i]
        end
    end
    for mink in 1:n
        k = n + 1 - mink
        vecX[k] = vecY[k] * matL[k, k]
        for i in k+1:n
            vecX[k] -= matL[i, k] * vecX[i]
        end
    end
    return vecX
end

"""
    getYacobian(imgSize = 1024, gridSize = 128)

Get the Jacobian matrix for the quadratic distortion correction. The default image size is 1024, and the grid size is 128.

# Arguments
- `imgSize::Int`: The size of the image.
- `gridSize::Int`: The size of the grid.

# Returns
- `Array{Float64,2}`: The Jacobian matrix.
"""
function getYacobian(imgSize::Int=1024, gridSize::Int=128)
    x = collect(gridSize+0.5:gridSize:imgSize)
    y = collect(gridSize+0.5:gridSize:imgSize)
    n = size(x)[1]
    na = 12

    yacob = Array{Float64}(undef, 2 * n * n, na)
    for j in 1:n
        for i in 1:n
            idx = i + (j - 1) * n
            yacob[2idx-1, 1] = -1.0
            yacob[2idx, 1] = 0.0

            yacob[2idx-1, 2] = -x[i]
            yacob[2idx, 2] = 0.0

            yacob[2idx-1, 3] = -y[j]
            yacob[2idx, 3] = 0.0

            yacob[2idx-1, 4] = -x[i]^2
            yacob[2idx, 4] = 0.0

            yacob[2idx-1, 5] = -x[i] * y[j]
            yacob[2idx, 5] = 0.0

            yacob[2idx-1, 6] = -y[j]^2
            yacob[2idx, 6] = 0.0

            yacob[2idx-1, 7] = 0.0
            yacob[2idx, 7] = -1.0

            yacob[2idx-1, 8] = 0.0
            yacob[2idx, 8] = -x[i]

            yacob[2idx-1, 9] = 0.0
            yacob[2idx, 9] = -y[j]

            yacob[2idx-1, 10] = 0.0
            yacob[2idx, 10] = -x[i]^2

            yacob[2idx-1, 11] = 0.0
            yacob[2idx, 11] = -x[i] * y[j]

            yacob[2idx-1, 12] = 0.0
            yacob[2idx, 12] = -y[j]^2
        end
    end
    return yacob
end

"""
    getErrorVec(vecMap, coefa, gridSize = 128, imgSize = 1024)

    Return the vector `errorVec` obtained by calculating the difference between `targetX` `targetY` and `procX` `procY` obtained by converting `targetX` `targetY` from the `gridx` `gridy` set in the first image to the second image. Also returns `gridx` `gridy` for Jacobian acquisition.

# Arguments
- `vecMap`: The PIV map.
- `coefa`: The coefficients for the quadratic distortion correction.
- `gridSize`: The size of the grid.
- `imgSize`: The size of the image.
"""
function getErrorVec(vecMap, coefa, gridSize=128, imgSize=1024)
    n = div(imgSize, gridSize) - 1
    gridx = collect(gridSize+0.5:gridSize:imgSize)
    gridy = collect(gridSize+0.5:gridSize:imgSize)
    targetX = Array{Float64}(undef, n * n)
    targetY = Array{Float64}(undef, n * n)
    procX = Array{Float64}(undef, n * n)
    procY = Array{Float64}(undef, n * n)

    for y in 1:n
        for x in 1:n
            targetX[x+n*(y-1)] = gridx[x] + vecMap[y, x, 1]
            targetY[x+n*(y-1)] = gridy[y] + vecMap[y, x, 2]
            procX[x+n*(y-1)] = coefa[1] + coefa[2] * gridx[x] + coefa[3] * gridy[y] + coefa[4] * gridx[x]^2 + coefa[5] * gridx[x] * gridy[y] + coefa[6] * gridy[y]^2
            procY[x+n*(y-1)] = coefa[7] + coefa[8] * gridx[x] + coefa[9] * gridy[y] + coefa[10] * gridx[x]^2 + coefa[11] * gridx[x] * gridy[y] + coefa[12] * gridy[y]^2
        end
    end

    errorVec = Array{Float64}(undef, 2 * n * n)

    for idx in 1:n*n
        errorVec[2*idx-1] = targetX[idx] - procX[idx]
        errorVec[2*idx] = targetY[idx] - procY[idx]
    end

    return errorVec
end

"""
    get_distortion_coeficients(img1, img2, gridSize = 128, intrSize = 128, srchSize = 256)

Get the distortion coefficients from the two images `img1` and `img2`. The default grid size is 128, the default search size is 256, and the default image size is 128.

# Arguments
- `img1::Array{<:AbstractFloat,2}`: The first image.
- `img2::Array{<:AbstractFloat,2}`: The second image.
- `gridSize::Int`: The size of the grid.
- `intrSize::Int`: The size of the search.
- `srchSize::Int`: The size of the search.

# Returns
- `Vector{<:AbstractFloat}`: The distortion coefficients.
"""
function get_distortion_coefficients(args...; kwargs...)
    _cuda_required_error(:get_distortion_coefficients)
end
