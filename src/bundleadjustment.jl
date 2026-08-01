using LinearAlgebra
using Logging
using Statistics

export quadratic_distortion_correction, get_distortion_coefficients, piv_map

"""Apply the package's 12-coefficient quadratic camera-distortion mapping."""
function quadratic_distortion_correction(image::AbstractMatrix{<:Real},
                                         coefficients::AbstractVector{<:Real})
    size(image, 1) == size(image, 2) || throw(DimensionMismatch("The image must be square. Got $(size(image))."))
    length(coefficients) == 12 || throw(ArgumentError("Twelve distortion coefficients are required."))
    host = to_host(image)
    n = size(host, 1)
    background = mean(host)
    output = Matrix{Float64}(undef, n, n)
    c = Float64.(coefficients)

    for row in 1:n, col in 1:n
        source_x = round(Int, c[1] + c[2] * col + c[3] * row +
                              c[4] * col^2 + c[5] * row * col + c[6] * row^2)
        source_y = round(Int, c[7] + c[8] * col + c[9] * row +
                              c[10] * col^2 + c[11] * row * col + c[12] * row^2)
        output[row, col] = 1 <= source_x <= n && 1 <= source_y <= n ?
                           host[source_y, source_x] : background
    end
    return output
end

function modified_Cholesky_decomposition(matrix::AbstractMatrix{<:Real})
    size(matrix, 1) == size(matrix, 2) || throw(DimensionMismatch("Matrix must be square."))
    result = Float64.(matrix)
    n = size(result, 1)
    work = zeros(Float64, n)
    for j in 1:n
        for i in 1:j-1
            work[i] = result[j, i]
            for k in 1:i-1
                work[i] -= result[i, k] * work[k]
            end
            result[j, i] = work[i] * result[i, i]
        end
        diagonal = result[j, j]
        for k in 1:j-1
            diagonal -= result[j, k] * work[k]
        end
        abs(diagonal) > eps(Float64) || throw(LinearAlgebra.PosDefException(j))
        result[j, j] = inv(diagonal)
    end
    return result
end

function simultanious_equation_solver(cholesky::AbstractMatrix{<:Real},
                                      jacobian::AbstractMatrix{<:Real},
                                      error::AbstractVector{<:Real})
    rhs = -transpose(jacobian) * error
    n = length(rhs)
    y = zeros(Float64, n)
    x = zeros(Float64, n)
    for k in 1:n
        y[k] = rhs[k] - sum(cholesky[k, i] * y[i] for i in 1:k-1; init=0.0)
    end
    for k in n:-1:1
        x[k] = y[k] * cholesky[k, k] -
               sum(cholesky[i, k] * x[i] for i in k+1:n; init=0.0)
    end
    return x
end

function getYacobian(image_size::Int=1024, grid_size::Int=128)
    grid = collect(grid_size + 0.5:grid_size:image_size)
    n = length(grid)
    jacobian = Matrix{Float64}(undef, 2n^2, 12)
    for row in 1:n, col in 1:n
        index = col + (row - 1) * n
        x = grid[col]
        y = grid[row]
        jacobian[2index-1, :] .= (-1, -x, -y, -x^2, -x*y, -y^2, 0, 0, 0, 0, 0, 0)
        jacobian[2index, :] .= (0, 0, 0, 0, 0, 0, -1, -x, -y, -x^2, -x*y, -y^2)
    end
    return jacobian
end

function getErrorVec(vector_map, coefficients, grid_size=128, image_size=1024)
    n = div(image_size, grid_size) - 1
    size(vector_map) == (n, n, 2) || throw(DimensionMismatch("PIV map must have size ($n, $n, 2)."))
    grid = collect(grid_size + 0.5:grid_size:image_size)
    error = Vector{Float64}(undef, 2n^2)
    c = Float64.(coefficients)
    for row in 1:n, col in 1:n
        index = col + (row - 1) * n
        x = grid[col]
        y = grid[row]
        target_x = x + vector_map[row, col, 1]
        target_y = y + vector_map[row, col, 2]
        mapped_x = c[1] + c[2]*x + c[3]*y + c[4]*x^2 + c[5]*x*y + c[6]*y^2
        mapped_y = c[7] + c[8]*x + c[9]*y + c[10]*x^2 + c[11]*x*y + c[12]*y^2
        error[2index-1] = target_x - mapped_x
        error[2index] = target_y - mapped_y
    end
    return error
end

function _validate_piv(image1, image2, grid_size, interrogation_size, search_size)
    size(image1) == size(image2) || throw(DimensionMismatch("PIV images must have the same size."))
    size(image1, 1) == size(image1, 2) || throw(DimensionMismatch("PIV images must be square."))
    all(>(0), (grid_size, interrogation_size, search_size)) || throw(ArgumentError("PIV sizes must be positive."))
    interrogation_size <= search_size || throw(ArgumentError("interrogation_size cannot exceed search_size."))
    iseven(interrogation_size) || throw(ArgumentError("interrogation_size must be even."))
    div(size(image1, 1), grid_size) >= 2 || throw(ArgumentError("grid_size leaves no PIV grid cells."))
    return nothing
end

function _normalized_cross_correlation(template, candidate)
    mean_template = mean(template)
    mean_candidate = mean(candidate)
    numerator = 0.0f0
    denominator_template = 0.0f0
    denominator_candidate = 0.0f0
    for index in eachindex(template, candidate)
        left = Float32(template[index] - mean_template)
        right = Float32(candidate[index] - mean_candidate)
        numerator += left * right
        denominator_template += left^2
        denominator_candidate += right^2
    end
    denominator = sqrt(denominator_template) * sqrt(denominator_candidate)
    return iszero(denominator) ? 0.0f0 : numerator / denominator
end

function _subpixel_peak(correlation::AbstractMatrix{<:Real}, row::Int, col::Int)
    if row == first(axes(correlation, 1)) || row == last(axes(correlation, 1)) ||
       col == first(axes(correlation, 2)) || col == last(axes(correlation, 2))
        return (0.0f0, 0.0f0)
    end
    center = Float32(correlation[row, col])
    denom_x = Float32(correlation[row, col+1] - 2center + correlation[row, col-1])
    denom_y = Float32(correlation[row+1, col] - 2center + correlation[row-1, col])
    offset_x = iszero(denom_x) ? 0.0f0 : Float32(correlation[row, col+1] - correlation[row, col-1]) / (2denom_x)
    offset_y = iszero(denom_y) ? 0.0f0 : Float32(correlation[row+1, col] - correlation[row-1, col]) / (2denom_y)
    return (offset_x, offset_y)
end

function _piv_map_cpu(image1, image2, grid_size::Int, interrogation_size::Int,
                      search_size::Int)
    left = Float32.(to_host(image1))
    right = Float32.(to_host(image2))
    image_size = size(left, 1)
    grid_count = div(image_size, grid_size)
    cells = grid_count - 1
    correlation_size = search_size - interrogation_size + 1
    output = Array{Float32}(undef, cells, cells, 2)
    correlation = Matrix{Float32}(undef, correlation_size, correlation_size)

    for grid_row in 1:cells, grid_col in 1:cells
        template_top = grid_row * grid_size - div(interrogation_size, 2) + 1
        template_left = grid_col * grid_size - div(interrogation_size, 2) + 1
        template_rows = template_top:template_top + interrogation_size - 1
        template_cols = template_left:template_left + interrogation_size - 1
        @views template = left[template_rows, template_cols]

        for shift_row in 1:correlation_size, shift_col in 1:correlation_size
            candidate_top = (grid_row - 1) * grid_size + shift_row
            candidate_left = (grid_col - 1) * grid_size + shift_col
            @views candidate = right[candidate_top:candidate_top + interrogation_size - 1,
                                      candidate_left:candidate_left + interrogation_size - 1]
            correlation[shift_row, shift_col] = _normalized_cross_correlation(template, candidate)
        end

        peak = argmax(correlation)
        offset_x, offset_y = _subpixel_peak(correlation, peak[1], peak[2])
        output[grid_row, grid_col, 1] = peak[2] - offset_x - interrogation_size / 2 - 1
        output[grid_row, grid_col, 2] = peak[1] - offset_y - interrogation_size / 2 - 1
    end
    return output
end

_piv_map(::AbstractBackend, image1, image2, grid_size, interrogation_size, search_size) =
    _piv_map_cpu(image1, image2, grid_size, interrogation_size, search_size)

"""
    piv_map(backend, image1, image2; grid_size=128, interrogation_size=128, search_size=256)

Compute the same brute-force normalized-cross-correlation PIV map used in
v0.2. CPU is the portable reference; CUDA.jl supplies the accelerated method.
Metal currently uses the documented CPU fallback for this calibration step.
"""
function piv_map(b::AbstractBackend, image1::AbstractMatrix{<:Real},
                 image2::AbstractMatrix{<:Real}; grid_size::Int=128,
                 interrogation_size::Int=128, search_size::Int=256)
    _validate_piv(image1, image2, grid_size, interrogation_size, search_size)
    return _piv_map(b, image1, image2, grid_size, interrogation_size, search_size)
end

piv_map(image1::AbstractMatrix{<:Real}, image2::AbstractMatrix{<:Real};
        backend::AbstractBackend=_DEFAULT_BACKEND[], kwargs...) =
    piv_map(backend, image1, image2; kwargs...)

function _save_distortion_diagnostics(args...; kwargs...)
    extension = Base.get_extension(@__MODULE__, :ParticleHolographyPlotsExt)
    isnothing(extension) && throw(ArgumentError("verbose diagnostics require Plots.jl. Install it and run `using Plots` before calling get_distortion_coefficients(...; verbose=true)."))
    return extension.save_distortion_diagnostics(args...; kwargs...)
end

"""Estimate the 12 quadratic distortion coefficients from a stereo image pair."""
function get_distortion_coefficients(image1::AbstractMatrix{<:Real},
                                     image2::AbstractMatrix{<:Real};
                                     backend::AbstractBackend=_DEFAULT_BACKEND[],
                                     verbose::Bool=false, save_dir::AbstractString="",
                                     grid_size::Int=128, interrogation_size::Int=128,
                                     search_size::Int=256, save_extension::AbstractString="png",
                                     # v0.2 keyword spellings
                                     gridSize::Union{Nothing,Int}=nothing,
                                     intrSize::Union{Nothing,Int}=nothing,
                                     srchSize::Union{Nothing,Int}=nothing)
    grid_size = something(gridSize, grid_size)
    interrogation_size = something(intrSize, interrogation_size)
    search_size = something(srchSize, search_size)
    vector_map = piv_map(backend, image1, image2; grid_size,
                         interrogation_size, search_size)
    image_size = size(image1, 1)
    jacobian = getYacobian(image_size, grid_size)
    normal_matrix = transpose(jacobian) * jacobian
    decomposition = modified_Cholesky_decomposition(normal_matrix)
    coefficients = ones(Float64, 12)
    for _ in 1:10
        error = getErrorVec(vector_map, coefficients, grid_size, image_size)
        coefficients .+= simultanious_equation_solver(decomposition, jacobian, error)
    end

    if verbose
        error = getErrorVec(vector_map, coefficients, grid_size, image_size)
        @info "Mean absolute calibration residual" value=mean(abs, error)
        corrected = quadratic_distortion_correction(image2, coefficients)
        corrected_map = piv_map(backend, image1, corrected; grid_size,
                                interrogation_size, search_size)
        _save_distortion_diagnostics(to_host(image1), to_host(image2), corrected,
                                     vector_map, corrected_map;
                                     save_dir, grid_size, save_extension)
    end
    return coefficients
end

# v0.2 internal name retained for downstream scripts.
getPIVMap_GPU(image1, image2, image_size=size(image1, 1), grid_size=128,
              interrogation_size=128, search_size=256) =
    piv_map(backend(:cuda), image1, image2; grid_size, interrogation_size, search_size)
