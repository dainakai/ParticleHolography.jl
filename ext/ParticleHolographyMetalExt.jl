module ParticleHolographyMetalExt

using ParticleHolography
import Metal

import ParticleHolography: MetalBackend, backendof, isfunctional
import ParticleHolography: _activate!, _synchronize, _to_backend, _to_host

function isfunctional(backend::MetalBackend)
    (isnothing(backend.device) || backend.device == 0) || return false
    return try
        Metal.functional()
    catch
        false
    end
end

function _activate!(backend::MetalBackend)
    (isnothing(backend.device) || backend.device == 0) ||
        throw(ArgumentError("Metal.jl currently exposes only device 0."))
    return nothing
end

_to_backend(backend::MetalBackend, array::Metal.WrappedMtlArray) = (_activate!(backend); array)
_to_backend(backend::MetalBackend, array::AbstractArray) = (_activate!(backend); Metal.MtlArray(array))
_to_backend(::MetalBackend, value) = value
_to_host(array::Metal.WrappedMtlArray) = Array(array)
backendof(::Metal.WrappedMtlArray) = MetalBackend()

function _synchronize(backend::MetalBackend)
    _activate!(backend)
    Metal.synchronize()
    return nothing
end

end
