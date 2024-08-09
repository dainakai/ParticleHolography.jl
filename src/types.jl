using CUDA

"""
    CuTransferSqrtPart{T<: AbstractFloat}

A struct that holds the data for the square root part of the transfer function.

# Fields
- `data::CuArray{T,2}`: The data for the square root part of the transfer function.
"""
struct CuTransferSqrtPart{T<:AbstractFloat}
    data::CuArray{T,2}
end

"""
    CuTransfer{T<: Complex}

A struct that holds the data for the transfer function.

# Fields
- `data::CuArray{T,2}`: The data for the transfer function.
"""
struct CuTransfer{T<:Complex}
    data::CuArray{T,2}
end

"""
    CuWavefront{T<: Complex}

A struct that holds the data for the wavefront.

# Fields
- `data::CuArray{T,2}`: The data for the wavefront.
"""
struct CuWavefront{T<:Complex}
    data::CuArray{T,2}
end


"""
    CuLowPassFilter{T<: AbstractFloat}

A struct that holds the data for the low pass filter. This can be multiplied with the Fourier transform of wavefront to get the low pass filtered wavefront after propagation.

# Fields
- `data::CuArray{T,2}`: The data for the low pass filter.
"""
struct CuLowPassFilter{T<:AbstractFloat}
    data::CuArray{T,2}
end