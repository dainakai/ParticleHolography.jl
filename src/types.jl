export PropagationGrid, PropagationKernel, Wavefront, LowPassFilter
export TransferSqrtPart, Transfer
export CuTransferSqrtPart, CuTransfer, CuWavefront, CuLowPassFilter

"""Distance-independent spatial-frequency grid for angular-spectrum propagation."""
struct PropagationGrid{T<:AbstractFloat,A<:AbstractMatrix{T}} <: AbstractMatrix{T}
    data::A
end

"""Distance-dependent angular-spectrum multiplier in FFT-native order."""
struct PropagationKernel{T<:Complex,A<:AbstractMatrix{T}} <: AbstractMatrix{T}
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

for Wrapper in (PropagationGrid, PropagationKernel, Wavefront, LowPassFilter)
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

backendof(x::Union{PropagationGrid,PropagationKernel,Wavefront,LowPassFilter}) = backendof(x.data)
to_host(x::PropagationGrid) = PropagationGrid(to_host(x.data))
to_host(x::PropagationKernel) = PropagationKernel(to_host(x.data))
to_host(x::Wavefront) = Wavefront(to_host(x.data))
to_host(x::LowPassFilter) = LowPassFilter(to_host(x.data))

# Compatibility aliases. New code should use PropagationGrid/PropagationKernel.
const TransferSqrtPart = PropagationGrid
const Transfer = PropagationKernel
const CuTransferSqrtPart = PropagationGrid
const CuTransfer = PropagationKernel
const CuWavefront = Wavefront
const CuLowPassFilter = LowPassFilter
