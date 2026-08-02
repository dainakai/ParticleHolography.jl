export AbstractBackend, CPUBackend
export backend, backend_name, backendof, available_backends, isfunctional
export to_backend, to_host, synchronize_backend

"""Abstract execution backend used by ParticleHolography plans."""
abstract type AbstractBackend end

"""CPU execution through ordinary Julia arrays and FFTW."""
struct CPUBackend <: AbstractBackend end

"""
    CUDABackend(; device=nothing)

NVIDIA CUDA backend. Load CUDA.jl (`using CUDA`) before constructing this
backend through [`backend`](@ref). `device` is a zero-based CUDA device index.
"""
struct CUDABackend <: AbstractBackend
    device::Union{Nothing,Int}
    function CUDABackend(device::Union{Nothing,Integer})
        !isnothing(device) && device < 0 && throw(ArgumentError("CUDA device must be a non-negative index."))
        return new(isnothing(device) ? nothing : Int(device))
    end
end
CUDABackend(; device=nothing) = CUDABackend(device)

"""
    MetalBackend(; device=nothing)

Apple Metal backend. Load Metal.jl (`using Metal`) before constructing this
backend through [`backend`](@ref). Metal currently exposes one logical device;
the optional field is reserved for a future multi-device API.
"""
struct MetalBackend <: AbstractBackend
    device::Union{Nothing,Int}
    function MetalBackend(device::Union{Nothing,Integer})
        !isnothing(device) && device < 0 && throw(ArgumentError("Metal device must be a non-negative index."))
        return new(isnothing(device) ? nothing : Int(device))
    end
end
MetalBackend(; device=nothing) = MetalBackend(device)

const _DEFAULT_BACKEND = Ref{AbstractBackend}(CPUBackend())
const _DEFAULT_BACKEND_LOCK = ReentrantLock()

backend_name(::CPUBackend) = :cpu
backend_name(::CUDABackend) = :cuda
backend_name(::MetalBackend) = :metal

Base.show(io::IO, b::AbstractBackend) = print(io, nameof(typeof(b)), "()")
Base.show(io::IO, b::CUDABackend) = isnothing(b.device) ? print(io, "CUDABackend()") : print(io, "CUDABackend(device=", b.device, ")")
Base.show(io::IO, b::MetalBackend) = isnothing(b.device) ? print(io, "MetalBackend()") : print(io, "MetalBackend(device=", b.device, ")")

"""Return whether a backend is usable in the current Julia process."""
isfunctional(::AbstractBackend) = false
isfunctional(::CPUBackend) = true

function _backend_unavailable_message(kind::Symbol)
    if kind === :cuda
        return "CUDA backend is unavailable. Install CUDA.jl, run `using CUDA`, and verify `CUDA.functional()` before selecting `backend(:cuda)`."
    elseif kind === :metal
        return "Metal backend is unavailable. On an Apple-silicon Mac with macOS 14+, install Metal.jl, run `using Metal`, and verify `Metal.functional()` before selecting `backend(:metal)`."
    end
    return "Unknown backend: $kind"
end

"""
    backend()

Return the process-wide default backend. A new Julia process starts with CPU.
Call `backend(:cpu)`, `backend(:cuda)`, or `backend(:metal)` to change it.
Explicit backend arguments remain available for side-by-side comparisons.
"""
backend() = _DEFAULT_BACKEND[]

function _construct_backend(kind::Symbol; device=nothing)
    if kind === :cpu
        isnothing(device) || throw(ArgumentError("CPUBackend does not accept a device index."))
        return CPUBackend()
    elseif kind === :cuda
        candidate = CUDABackend(; device)
        isfunctional(candidate) || throw(ArgumentError(_backend_unavailable_message(:cuda)))
        return candidate
    elseif kind === :metal
        candidate = MetalBackend(; device)
        isfunctional(candidate) || throw(ArgumentError(_backend_unavailable_message(:metal)))
        return candidate
    elseif kind === :auto
        isnothing(device) || throw(ArgumentError("`device` cannot be combined with backend(:auto)."))
        cuda = CUDABackend()
        isfunctional(cuda) && return cuda
        metal = MetalBackend()
        isfunctional(metal) && return metal
        return CPUBackend()
    end
    throw(ArgumentError("Backend must be :cpu, :cuda, :metal, or :auto. Got $kind."))
end

"""
    backend(kind; device=nothing)

Select and return the process-wide default execution backend. `kind` is `:cpu`,
`:cuda`, `:metal`, or `:auto`. Subsequent calls that omit a backend use this
selection. `:auto` prefers CUDA, then Metal, and always falls back to CPU.

Changing the default while concurrent tasks are running is unsupported. Pass an
explicit backend object to each call when CPU and GPU work must coexist.
"""
function backend(kind::Symbol; device=nothing)
    selected = _construct_backend(kind; device)
    lock(_DEFAULT_BACKEND_LOCK) do
        _DEFAULT_BACKEND[] = selected
    end
    return selected
end

"""Return the symbols of backends usable in the current process."""
function available_backends()
    result = Symbol[:cpu]
    isfunctional(CUDABackend()) && push!(result, :cuda)
    isfunctional(MetalBackend()) && push!(result, :metal)
    return result
end

backendof(::Array) = CPUBackend()
backendof(::AbstractArray) = CPUBackend()

_activate!(::AbstractBackend) = nothing

function _to_backend(b::AbstractBackend, x)
    throw(ArgumentError(_backend_unavailable_message(backend_name(b))))
end
_to_backend(::CPUBackend, x::Array) = x
_to_backend(::CPUBackend, x::AbstractArray) = Array(x)
_to_backend(::CPUBackend, x) = x

"""Copy `x` to `b`. An input already on `b` may be returned without copying."""
function to_backend(b::AbstractBackend, x)
    _activate!(b)
    return _to_backend(b, x)
end

"""Copy `x` to the currently selected [`backend()`](@ref)."""
to_backend(x) = to_backend(backend(), x)

_to_host(x::Array) = x
_to_host(x::AbstractArray) = Array(x)
_to_host(x) = x

_copy_to_host!(destination::Array, source::AbstractArray) = copyto!(destination, to_host(source))

"""Copy an array or wrapped optical value to host memory."""
to_host(x) = _to_host(x)

_synchronize(::CPUBackend) = nothing
function _synchronize(b::AbstractBackend)
    throw(ArgumentError(_backend_unavailable_message(backend_name(b))))
end

"""Wait for queued work on `b` to complete."""
function synchronize(b::AbstractBackend)
    _activate!(b)
    return _synchronize(b)
end

"""
    synchronize_backend(backend)

Wait for queued work on an execution backend. Use this before timing GPU work.
The longer name avoids collisions with functions exported by CUDA.jl/Metal.jl.
"""
synchronize_backend(b::AbstractBackend) = synchronize(b)
synchronize_backend() = synchronize_backend(backend())

_available_memory(::AbstractBackend) = nothing
_available_memory(::CPUBackend) = Int(Sys.free_memory())
_available_memory(::MetalBackend) = Int(Sys.free_memory())

_memory_kind(::AbstractBackend) = :unknown
_memory_kind(::CPUBackend) = :host
_memory_kind(::CUDABackend) = :device
_memory_kind(::MetalBackend) = :unified
