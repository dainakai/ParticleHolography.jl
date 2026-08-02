module ParticleHolographyCUDAExt

using ParticleHolography
import CUDA

import ParticleHolography: CUDABackend, backendof, isfunctional
import ParticleHolography: _activate!, _available_memory, _copy_to_host!
import ParticleHolography: _piv_map, _synchronize, _to_backend, _to_host

function isfunctional(backend::CUDABackend)
    return try
        CUDA.functional() || return false
        isnothing(backend.device) || backend.device < length(CUDA.devices())
    catch
        false
    end
end

function _activate!(backend::CUDABackend)
    !isnothing(backend.device) && CUDA.device!(backend.device)
    return nothing
end

function _to_backend(backend::CUDABackend, array::CUDA.AnyCuArray)
    _activate!(backend)
    same_device = isnothing(backend.device) ||
                  CUDA.deviceid(CUDA.device(array)) == backend.device
    return same_device ? array : CUDA.CuArray(Array(array))
end

function _to_backend(backend::CUDABackend, array::AbstractArray)
    _activate!(backend)
    return CUDA.CuArray(array)
end

_to_backend(::CUDABackend, value) = value
_to_host(array::CUDA.AnyCuArray) = Array(array)
backendof(array::CUDA.AnyCuArray) =
    CUDABackend(CUDA.deviceid(CUDA.device(array)))
_available_memory(backend::CUDABackend) = (_activate!(backend); Int(CUDA.free_memory()))
_copy_to_host!(destination::Array, source::CUDA.AnyCuArray) = copyto!(destination, source)

function _synchronize(backend::CUDABackend)
    _activate!(backend)
    CUDA.synchronize()
    return nothing
end

function _cross_correlation_kernel!(correlation, image1, image2, grid_row,
                                    grid_count, search_size,
                                    interrogation_size, grid_size)
    x = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    y = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y
    correlation_size = search_size - interrogation_size + 1
    cells = grid_count - 1
    if x <= correlation_size * cells && y <= correlation_size
        grid_col = div(x - 1, correlation_size) + 1
        shift_col = x - (grid_col - 1) * correlation_size
        shift_row = y
        template_top = grid_row * grid_size - div(interrogation_size, 2)
        template_left = grid_col * grid_size - div(interrogation_size, 2)
        candidate_top = (grid_row - 1) * grid_size + shift_row - 1
        candidate_left = (grid_col - 1) * grid_size + shift_col - 1

        mean_template = 0.0f0
        mean_candidate = 0.0f0
        for row in 1:interrogation_size, col in 1:interrogation_size
            mean_template += image1[template_top + row, template_left + col]
            mean_candidate += image2[candidate_top + row, candidate_left + col]
        end
        count = Float32(interrogation_size^2)
        mean_template /= count
        mean_candidate /= count

        numerator = 0.0f0
        denominator_template = 0.0f0
        denominator_candidate = 0.0f0
        for row in 1:interrogation_size, col in 1:interrogation_size
            left = image1[template_top + row, template_left + col] - mean_template
            right = image2[candidate_top + row, candidate_left + col] - mean_candidate
            numerator += left * right
            denominator_template += left^2
            denominator_candidate += right^2
        end
        denominator = sqrt(denominator_template) * sqrt(denominator_candidate)
        correlation[y + correlation_size * (grid_row - 1), x] =
            iszero(denominator) ? 0.0f0 : numerator / denominator
    end
    return nothing
end

function _vector_kernel!(vectors, correlation, cells, correlation_size,
                         interrogation_size)
    grid_col = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    grid_row = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y
    if grid_col <= cells && grid_row <= cells
        best_value = -Inf32
        best_row = 1
        best_col = 1
        row_offset = correlation_size * (grid_row - 1)
        col_offset = correlation_size * (grid_col - 1)
        for row in 1:correlation_size, col in 1:correlation_size
            value = correlation[row_offset + row, col_offset + col]
            if value > best_value
                best_value = value
                best_row = row
                best_col = col
            end
        end

        offset_x = 0.0f0
        offset_y = 0.0f0
        if 1 < best_col < correlation_size
            center = correlation[row_offset + best_row, col_offset + best_col]
            minus = correlation[row_offset + best_row, col_offset + best_col - 1]
            plus = correlation[row_offset + best_row, col_offset + best_col + 1]
            denominator = plus - 2center + minus
            !iszero(denominator) && (offset_x = (plus - minus) / (2denominator))
        end
        if 1 < best_row < correlation_size
            center = correlation[row_offset + best_row, col_offset + best_col]
            minus = correlation[row_offset + best_row - 1, col_offset + best_col]
            plus = correlation[row_offset + best_row + 1, col_offset + best_col]
            denominator = plus - 2center + minus
            !iszero(denominator) && (offset_y = (plus - minus) / (2denominator))
        end
        vectors[grid_row, grid_col, 1] = best_col - offset_x - interrogation_size / 2 - 1
        vectors[grid_row, grid_col, 2] = best_row - offset_y - interrogation_size / 2 - 1
    end
    return nothing
end

function _piv_map(backend::CUDABackend, image1, image2, grid_size,
                  interrogation_size, search_size)
    _activate!(backend)
    device1 = _to_backend(backend, Float32.(image1))
    device2 = _to_backend(backend, Float32.(image2))
    grid_count = div(size(image1, 1), grid_size)
    cells = grid_count - 1
    correlation_size = search_size - interrogation_size + 1
    correlation = CUDA.CuArray{Float32}(undef, correlation_size * cells,
                                        correlation_size * cells)
    vectors = CUDA.CuArray{Float32}(undef, cells, cells, 2)
    threads = (16, 16)
    correlation_blocks = (cld(correlation_size * cells, threads[1]),
                          cld(correlation_size, threads[2]))
    for grid_row in 1:cells
        CUDA.@cuda threads=threads blocks=correlation_blocks _cross_correlation_kernel!(
            correlation, device1, device2, grid_row, grid_count, search_size,
            interrogation_size, grid_size)
    end
    vector_blocks = (cld(cells, threads[1]), cld(cells, threads[2]))
    CUDA.@cuda threads=threads blocks=vector_blocks _vector_kernel!(
        vectors, correlation, cells, correlation_size, interrogation_size)
    CUDA.synchronize()
    return Array(vectors)
end

end
