using Logging

# COV_EXCL_START
function CuGetVector!(vecArray::CuDeviceArray{Float32,3}, corArray::CuDeviceArray{Float32,2}, gridNum::Int64, corArrSize::Int64, intrSize::Int64)
    gridIdxx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    gridIdxy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if gridIdxx <= gridNum - 1 && gridIdxy <= gridNum - 1
        x0::Int64 = 0
        y0::Int64 = 0

        tmp::Float32 = 0.0
        for i in 1:corArrSize
            for j in 1:corArrSize
                if corArray[corArrSize*(gridIdxy-1)+i, corArrSize*(gridIdxx-1)+j] > tmp
                    x0 = corArrSize * (gridIdxx - 1) + j
                    y0 = corArrSize * (gridIdxy - 1) + i
                    tmp = corArray[corArrSize*(gridIdxy-1)+i, corArrSize*(gridIdxx-1)+j]
                end
            end
        end

        valy1x0::Float32 = corArray[y0+1, x0]
        valy0x0::Float32 = corArray[y0, x0]
        valyInv1x0::Float32 = corArray[y0-1, x0]
        valy0x1::Float32 = corArray[y0, x0+1]
        valy0xInv1::Float32 = corArray[y0, x0-1]

        if (valy1x0 - 2.0 * valy0x0 + valyInv1x0 == 0.0) || (valy0x1 - 2.0 * valy0x0 + valy0xInv1 == 0.0)
            valy0x0 += 0.00001
        end

        vecArray[gridIdxy, gridIdxx, 1] = Float32(x0) - (valy0x1 - valy0xInv1) / (valy0x1 - 2.0 * valy0x0 + valy0xInv1) / 2.0 - Float32(intrSize) / 2.0 - 1.0 - (gridIdxx - 1) * corArrSize
        vecArray[gridIdxy, gridIdxx, 2] = Float32(y0) - (valy1x0 - valyInv1x0) / (valy1x0 - 2.0 * valy0x0 + valyInv1x0) / 2.0 - intrSize / 2.0 - 1.0 - (gridIdxy - 1) * corArrSize
    end
    return nothing
end

function CuGetCrossCor!(corArray::CuDeviceArray{Float32,2}, img1::CuDeviceArray{Float32,2}, img2::CuDeviceArray{Float32,2}, gridIdxy::Int64, gridNum::Int64, srchSize::Int64, intrSize::Int64, gridSize::Int64)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if x <= (srchSize - intrSize + 1) * (gridNum - 1) && y <= (srchSize - intrSize + 1)
        gridIdxx = div(x - 1, (srchSize - intrSize + 1)) + 1
        idxx = x - (gridIdxx - 1) * (srchSize - intrSize + 1)
        idxy = y

        a1::Int64 = gridIdxy * gridSize - div(intrSize, 2)
        a2::Int64 = gridIdxx * gridSize - div(intrSize, 2)
        b1::Int64 = (gridIdxy - 1) * gridSize + idxy - 1
        b2::Int64 = (gridIdxx - 1) * gridSize + idxx - 1

        meanA::Float32 = 0.0
        meanB::Float32 = 0.0
        num::Float32 = 0.0
        denomA::Float32 = 0.0
        denomB::Float32 = 0.0

        for i in 1:intrSize
            for j in 1:intrSize
                meanA += img1[a1+i, a2+j]
                meanB += img2[b1+i, b2+j]
            end
        end
        meanA /= Float32(intrSize^2)
        meanB /= Float32(intrSize^2)

        for i in 1:intrSize
            for j in 1:intrSize
                num += (img1[a1+i, a2+j] - meanA) * (img2[b1+i, b2+j] - meanB)
                denomA += (img1[a1+i, a2+j] - meanA)^2
                denomB += (img2[b1+i, b2+j] - meanB)^2
            end
        end

        corArray[y+(srchSize-intrSize+1)*(gridIdxy-1), x] = num / (CUDA.sqrt(denomA) * CUDA.sqrt(denomB))
    end
    return nothing
end
# COV_EXCL_STOP

function getPIVMap_GPU(image1, image2, imgLen=1024, gridSize=128, intrSize=128, srchSize=256)
    _assert_cuda_functional(:getPIVMap_GPU)
    gridNum = div(imgLen, gridSize)
    corArray = CuArray{Float32}(undef, ((srchSize - intrSize + 1) * (gridNum - 1), (srchSize - intrSize + 1) * (gridNum - 1)))
    vecArray = CuArray{Float32}(undef, (gridNum - 1, gridNum - 1, 2))

    blockSize = 16
    threads1 = (blockSize, blockSize)
    blocks1 = (cld((srchSize - intrSize + 1) * (gridNum - 1), blockSize), cld((srchSize - intrSize + 1), blockSize))
    blocks2 = (cld(gridNum - 1, blockSize), cld(gridNum - 1, blockSize))

    d_img1 = cu(image1)
    d_img2 = cu(image2)

    for idx in 1:gridNum-1
        @cuda threads=threads1 blocks=blocks1 CuGetCrossCor!(corArray, d_img1, d_img2, idx, gridNum, srchSize, intrSize, gridSize)
    end

    @cuda threads=threads1 blocks=blocks2 CuGetVector!(vecArray, corArray, gridNum, srchSize - intrSize + 1, intrSize)
    output = Array(vecArray)

    return output
end

function get_distortion_coefficients(img1::Array{<:AbstractFloat,2}, img2::Array{<:AbstractFloat,2}; verbose=false, save_dir="", gridSize=128, intrSize=128, srchSize=256, save_extension="png")
    _assert_cuda_functional(:get_distortion_coefficients)
    @assert size(img1) == size(img2) "The size of the images must be the same. Got $(size(img1)) and $(size(img2))."
    imgLen = size(img1)[1]
    vecArray = getPIVMap_GPU(img1, img2, imgLen, gridSize, intrSize, srchSize)
    yacobian = getYacobian(imgLen, gridSize)
    hMat = transpose(yacobian) * yacobian
    coefa = fill(1.0, 12)
    itr = 1
    while itr <= 10
        errorVec = getErrorVec(vecArray, coefa, gridSize, imgLen)
        deltaCoefa = simultanious_equation_solver(modified_Cholesky_decomposition(hMat), yacobian, errorVec)
        coefa += deltaCoefa
        itr += 1
    end

    if verbose
        @info "verbose=true is set. Plotting the results. It may take a while."
        _save_bundle_adjustment_diagnostics(img1, img2, vecArray, coefa; save_dir=save_dir, gridSize=gridSize, intrSize=intrSize, srchSize=srchSize, save_extension=save_extension)
    end
    return coefa
end
