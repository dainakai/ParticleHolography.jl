using CUDA

"""
    CuTransferSqrtPart{T<: AbstractFloat}

A struct that holds the data for the square root part of the transfer function.

# Fields
- `data::CuArray{T,2}`: The data for the square root part of the transfer function.
"""
struct CuTransferSqrtPart{T<: AbstractFloat}
    data::CuArray{T,2}
end

"""
    CuTransfer{T<: Complex}

A struct that holds the data for the transfer function.

# Fields
- `data::CuArray{T,2}`: The data for the transfer function.
"""
struct CuTransfer{T<: Complex}
    data::CuArray{T,2}
end

"""
    CuWavefront{T<: Complex}

A struct that holds the data for the wavefront.

# Fields
- `data::CuArray{T,2}`: The data for the wavefront.
"""
struct CuWavefront{T<: Complex}
    data::CuArray{T,2}
end