export TransferSqrtPart, Transfer, Wavefront, LowPassFilter
export CuTransferSqrtPart, CuTransfer, CuWavefront, CuLowPassFilter

"""FFT-native square-root term used to construct angular-spectrum transfers."""
struct TransferSqrtPart{T<:AbstractFloat,A<:AbstractMatrix{T}} <: AbstractMatrix{T}
    data::A
end

"""Angular-spectrum transfer function stored in FFT-native frequency order."""
struct Transfer{T<:Complex,A<:AbstractMatrix{T}} <: AbstractMatrix{T}
    data::A
end

"""Complex optical wavefront on any supported execution backend."""
struct Wavefront{T<:Complex,A<:AbstractMatrix{T}} <: AbstractMatrix{T}
    data::A
end

"""Frequency-domain low-pass filter stored in FFT-native order."""
struct LowPassFilter{T<:AbstractFloat,A<:AbstractMatrix{T}} <: AbstractMatrix{T}
    data::A
end

for Wrapper in (TransferSqrtPart, Transfer, Wavefront, LowPassFilter)
    @eval begin
        Base.size(x::$Wrapper) = size(x.data)
        Base.axes(x::$Wrapper) = axes(x.data)
        Base.eltype(x::$Wrapper) = eltype(x.data)
        Base.ndims(x::$Wrapper) = ndims(x.data)
        Base.length(x::$Wrapper) = length(x.data)
        Base.getindex(x::$Wrapper, I...) = getindex(x.data, I...)
        Base.setindex!(x::$Wrapper, value, I...) = setindex!(x.data, value, I...)
        Base.parent(x::$Wrapper) = x.data
        Base.copy(x::$Wrapper) = $Wrapper(copy(x.data))
    end
end

backendof(x::Union{TransferSqrtPart,Transfer,Wavefront,LowPassFilter}) = backendof(x.data)
to_host(x::TransferSqrtPart) = TransferSqrtPart(to_host(x.data))
to_host(x::Transfer) = Transfer(to_host(x.data))
to_host(x::Wavefront) = Wavefront(to_host(x.data))
to_host(x::LowPassFilter) = LowPassFilter(to_host(x.data))

# v0.2 compatibility aliases. New code should use the backend-neutral names.
const CuTransferSqrtPart = TransferSqrtPart
const CuTransfer = Transfer
const CuWavefront = Wavefront
const CuLowPassFilter = LowPassFilter
