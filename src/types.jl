export CuTransferSqrtPart, CuTransfer, CuWavefront, CuLowPassFilter

"""
    CuTransferSqrtPart{T<: AbstractFloat}

A struct that holds the data for the square root part of the transfer function.

# Fields
- `data::AbstractArray{T,2}`: The data for the square root part of the transfer function.
"""
struct CuTransferSqrtPart{T<:AbstractFloat}
    data::AbstractArray{T,2}
end

"""
    CuTransfer{T<: Complex}

A struct that holds the data for the transfer function.

# Fields
- `data::AbstractArray{T,2}`: The data for the transfer function.
"""
struct CuTransfer{T<:Complex}
    data::AbstractArray{T,2}
end

"""
    CuWavefront{T<: Complex}

A struct that holds the data for the wavefront.

# Fields
- `data::AbstractArray{T,2}`: The data for the wavefront.
"""
struct CuWavefront{T<:Complex}
    data::AbstractArray{T,2}
end


"""
    CuLowPassFilter{T<: AbstractFloat}

A struct that holds the data for the low pass filter. This can be multiplied with the Fourier transform of wavefront to get the low pass filtered wavefront after propagation.

# Fields
- `data::AbstractArray{T,2}`: The data for the low pass filter.
"""
struct CuLowPassFilter{T<:AbstractFloat}
    data::AbstractArray{T,2}
end

for T in (CuTransferSqrtPart, CuTransfer, CuWavefront, CuLowPassFilter)
    @eval begin
        Base.size(x::$T) = size(x.data)
        Base.axes(x::$T) = axes(x.data)
        Base.eltype(x::$T) = eltype(x.data)
        Base.ndims(x::$T) = ndims(x.data)
        Base.IndexStyle(::Type{<:$T}) = IndexCartesian()
    end
end
