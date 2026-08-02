using AbstractFFTs
using FFTW
using FixedPointNumbers: N0f8
using LinearAlgebra: mul!
using Statistics: mean

export propagation_grid, propagation_kernel, gabor_wavefront, pad2d
export transfer_sqrt, transfer
export PhaseRetrievalPlan, phase_retrieval!, phase_retrieval
export ReconstructionRequest, ReconstructionResult, MemoryDiagnostic, memory_diagnostic
export N0f8
export ReconstructionPlan, reconstruct!, reconstruct, reconstruct_complex!, reconstruct_complex
export xyprojection!, xyprojection, reconstruct_and_projection!, reconstruct_and_projection
export reconstruct_padded, reconstruct_and_projection_padded, asm_propagate!, asm_propagate
export cu_transfer_sqrt_arr, cu_transfer, cu_gabor_wavefront, cu_phase_retrieval_holo
export cu_get_reconst_vol, cu_get_reconst_complex_vol, cu_get_reconst_xyprojection
export cu_get_reconst_vol_and_xyprojection, cu_get_reconst_vol_and_xyprojection_padded
export cu_asm_prop!, cu_2d_pad

const _OpticalReal = Union{Float16,Float32,Float64}

function _validate_optical_parameters(wavelength::Real, pixel_pitch::Real)
    isfinite(wavelength) && wavelength > 0 || throw(ArgumentError("wavelength must be finite and positive. Got $wavelength."))
    isfinite(pixel_pitch) && pixel_pitch > 0 || throw(ArgumentError("pixel_pitch must be finite and positive. Got $pixel_pitch."))
    return nothing
end

function _validate_shape(shape::Tuple{Int,Int})
    all(>(0), shape) || throw(ArgumentError("Array dimensions must be positive. Got $shape."))
    return shape
end

_validate_shape(datlen::Int) = _validate_shape((datlen, datlen))

@inline function _fft_frequency_index(i::Int, n::Int)
    k = i - 1
    return k <= fld(n - 1, 2) ? k : k - n
end

"""
    propagation_grid([backend], shape, wavelength, pixel_pitch)

Construct the distance-independent spatial-frequency grid used by the
angular-spectrum method. `shape` may be a square side length or
`(height, width)`. All length parameters must use the same unit.
"""
function propagation_grid(b::AbstractBackend, shape::Tuple{Int,Int}, wavelength::Real, pixel_pitch::Real)
    _validate_shape(shape)
    _validate_optical_parameters(wavelength, pixel_pitch)
    height, width = shape
    λ = Float64(wavelength)
    dx = Float64(pixel_pitch)
    host = Matrix{Float32}(undef, height, width)
    for j in 1:width, i in 1:height
        fy = _fft_frequency_index(i, height) / (height * dx)
        fx = _fft_frequency_index(j, width) / (width * dx)
        host[i, j] = Float32(1 - (λ * fx)^2 - (λ * fy)^2)
    end
    minimum(host) >= 0 || throw(DomainError(minimum(host), "The sampling parameters include evanescent spatial frequencies. Increase pixel_pitch or use a smaller wavelength."))
    return PropagationGrid(to_backend(b, host))
end

propagation_grid(b::AbstractBackend, datlen::Int, wavelength::Real, pixel_pitch::Real) =
    propagation_grid(b, _validate_shape(datlen), wavelength, pixel_pitch)

propagation_grid(shape, wavelength::Real, pixel_pitch::Real;
                 backend::AbstractBackend=_DEFAULT_BACKEND[]) =
    propagation_grid(backend, shape, wavelength, pixel_pitch)

function transfer_sqrt(args...; kwargs...)
    Base.depwarn("`transfer_sqrt` is renamed to `propagation_grid` in v1.", :transfer_sqrt)
    return propagation_grid(args...; kwargs...)
end

"""
    propagation_kernel([backend], distance, wavelength, grid)

Construct the angular-spectrum propagation multiplier for `distance`. The
kernel is stored in the order consumed directly by `plan_fft`.
"""
function propagation_kernel(b::AbstractBackend, distance::Real, wavelength::Real,
                            grid::PropagationGrid)
    isfinite(distance) || throw(ArgumentError("distance must be finite. Got $distance."))
    isfinite(wavelength) && wavelength > 0 || throw(ArgumentError("wavelength must be finite and positive. Got $wavelength."))
    phase_scale = 2π * Float64(distance) / Float64(wavelength)
    sqrt_host = to_host(grid.data)
    minimum(sqrt_host) >= 0 || throw(DomainError(minimum(sqrt_host), "transfer square-root term contains negative values."))
    host = ComplexF32.(exp.(complex.(0.0, phase_scale .* sqrt.(Float64.(sqrt_host)))))
    return PropagationKernel(to_backend(b, host))
end

propagation_kernel(distance::Real, wavelength::Real, grid::PropagationGrid;
                   backend::AbstractBackend=_DEFAULT_BACKEND[]) =
    propagation_kernel(backend, distance, wavelength, grid)

function transfer(args...; kwargs...)
    Base.depwarn("`transfer` is renamed to `propagation_kernel` in v1.", :transfer)
    return propagation_kernel(args...; kwargs...)
end

function _real_input(b::AbstractBackend, input::AbstractMatrix)
    eltype(input) <: Real || throw(ArgumentError("Hologram intensity must be real-valued. Got $(eltype(input))."))
    if backendof(input) isa CPUBackend
        minimum(input) >= 0 || throw(DomainError(minimum(input), "Hologram intensity cannot be negative."))
    end
    return to_backend(b, Float32.(input))
end

"""Create a complex Gabor wavefront from a non-negative intensity image."""
function gabor_wavefront(b::AbstractBackend, hologram::AbstractMatrix)
    intensity = _real_input(b, hologram)
    return Wavefront(ComplexF32.(sqrt.(intensity)))
end

gabor_wavefront(hologram::AbstractMatrix; backend::AbstractBackend=_DEFAULT_BACKEND[]) =
    gabor_wavefront(backend, hologram)

"""
    pad2d(input[, target_shape]; mode=:mean)

Centre `input` in a larger 2-D array. The default target is twice each input
dimension. `mode=:mean` fills the border with the input mean; `mode=:zero`
uses zero. The result stays on the input backend.
"""
function pad2d(input::AbstractMatrix,
               target_shape::Tuple{Int,Int}=(2 * size(input, 1), 2 * size(input, 2));
               mode::Symbol=:mean)
    all(target_shape .>= size(input)) || throw(ArgumentError(
        "target_shape must be at least the input size; got $target_shape and $(size(input)).",
    ))
    mode in (:mean, :zero) || throw(ArgumentError("padding mode must be :mean or :zero. Got $mode."))
    output = similar(input, eltype(input), target_shape)
    fill_value = mode === :mean ? mean(to_host(input)) : zero(eltype(input))
    fill!(output, convert(eltype(input), fill_value))
    height, width = size(input)
    rows = fld(target_shape[1] - height, 2) + 1:fld(target_shape[1] - height, 2) + height
    cols = fld(target_shape[2] - width, 2) + 1:fld(target_shape[2] - width, 2) + width
    @views output[rows, cols] .= input
    return output
end

"""Reusable Gerchberg-Saxton workspace for a pair of holograms."""
mutable struct PhaseRetrievalPlan{B<:AbstractBackend,A<:AbstractMatrix,R<:AbstractMatrix,PF,PI}
    backend::B
    forward_transfer::A
    backward_transfer::A
    light1::A
    light2::A
    frequency1::A
    frequency2::A
    sqrt_intensity1::R
    sqrt_intensity2::R
    phase1::R
    phase2::R
    fft_plan::PF
    ifft_plan::PI
    check_memory::Bool
    memory_safety_factor::Float64
end

function PhaseRetrievalPlan(b::AbstractBackend, forward::PropagationKernel,
                            backward::PropagationKernel;
                            check_memory::Bool=true,
                            memory_safety_factor::Real=1.2,
                            available_memory::Union{Nothing,Integer}=_available_memory(b))
    size(forward) == size(backward) || throw(DimensionMismatch("Forward and backward transfers must have the same size. Got $(size(forward)) and $(size(backward))."))
    diagnostic = _phase_memory_diagnostic(b, size(forward);
                                          safety_factor=memory_safety_factor,
                                          available_bytes=available_memory)
    _enforce_memory(diagnostic, check_memory)
    _activate!(b)
    tf = ComplexF32.(to_backend(b, forward.data))
    tb = ComplexF32.(to_backend(b, backward.data))
    light1 = similar(tf)
    light2 = similar(tf)
    frequency1 = similar(tf)
    frequency2 = similar(tf)
    sqrt1 = similar(tf, Float32)
    sqrt2 = similar(tf, Float32)
    phase1 = similar(tf, Float32)
    phase2 = similar(tf, Float32)
    fill!(light1, 0)
    fft_plan = plan_fft(light1)
    ifft_plan = plan_ifft(light1)
    return PhaseRetrievalPlan(b, tf, tb, light1, light2, frequency1, frequency2,
                              sqrt1, sqrt2, phase1, phase2, fft_plan, ifft_plan,
                              check_memory, Float64(memory_safety_factor))
end

PhaseRetrievalPlan(forward::PropagationKernel, backward::PropagationKernel;
                   backend::AbstractBackend=backendof(forward), kwargs...) =
    PhaseRetrievalPlan(backend, forward, backward; kwargs...)

"""
    phase_retrieval!(plan, hologram1, hologram2; iterations=10)

Run Gerchberg-Saxton phase retrieval using buffers owned by `plan`. The returned
`Wavefront` aliases the plan workspace and is overwritten by the next call.
Use [`phase_retrieval`](@ref) when an owning copy is required.
"""
function phase_retrieval!(plan::PhaseRetrievalPlan, hologram1::AbstractMatrix,
                          hologram2::AbstractMatrix; iterations::Integer=10)
    iterations >= 0 || throw(ArgumentError("iterations must be non-negative. Got $iterations."))
    size(hologram1) == size(hologram2) == size(plan.forward_transfer) ||
        throw(DimensionMismatch("Holograms and transfers must share size $(size(plan.forward_transfer)); got $(size(hologram1)) and $(size(hologram2))."))
    _activate!(plan.backend)
    intensity1 = _real_input(plan.backend, hologram1)
    intensity2 = _real_input(plan.backend, hologram2)
    plan.sqrt_intensity1 .= sqrt.(intensity1)
    plan.sqrt_intensity2 .= sqrt.(intensity2)
    plan.light1 .= ComplexF32.(plan.sqrt_intensity1)

    for _ in 1:iterations
        mul!(plan.frequency1, plan.fft_plan, plan.light1)
        plan.frequency1 .*= plan.forward_transfer
        mul!(plan.light2, plan.ifft_plan, plan.frequency1)
        plan.phase2 .= angle.(plan.light2)
        plan.light2 .= plan.sqrt_intensity2 .* complex.(cos.(plan.phase2), sin.(plan.phase2))

        mul!(plan.frequency2, plan.fft_plan, plan.light2)
        plan.frequency2 .*= plan.backward_transfer
        mul!(plan.light1, plan.ifft_plan, plan.frequency2)
        plan.phase1 .= angle.(plan.light1)
        plan.light1 .= plan.sqrt_intensity1 .* complex.(cos.(plan.phase1), sin.(plan.phase1))
    end
    return Wavefront(plan.light1)
end

"""Run phase retrieval with `plan` and return an owning wavefront copy."""
function phase_retrieval(plan::PhaseRetrievalPlan, hologram1::AbstractMatrix,
                         hologram2::AbstractMatrix; iterations::Integer=10)
    result = phase_retrieval!(plan, hologram1, hologram2; iterations)
    return Wavefront(copy(result.data))
end

function phase_retrieval(b::AbstractBackend, hologram1::AbstractMatrix,
                         hologram2::AbstractMatrix,
                         forward::PropagationKernel,
                         backward::PropagationKernel; iterations::Integer=10)
    plan = PhaseRetrievalPlan(b, forward, backward)
    return phase_retrieval(plan, hologram1, hologram2; iterations)
end

"""
    ReconstructionRequest(slices; volume=Float32, min_projection=nothing)

Describe the outputs to produce during one depth scan. Real-valued `volume`
types store intensity, complex types store the propagated wavefront, and
`nothing` avoids allocating that output. `volume` accepts `N0f8`, `Float32`,
`ComplexF32`, `ComplexF64`, or `nothing`; `min_projection` accepts `N0f8`,
`Float32`, or `nothing`.
"""
struct ReconstructionRequest
    slices::Int
    volume::Union{Nothing,DataType}
    min_projection::Union{Nothing,DataType}

    function ReconstructionRequest(slices::Integer;
                                   volume::Union{Nothing,DataType}=Float32,
                                   min_projection::Union{Nothing,DataType}=nothing)
        count = _validate_slices(slices)
        isnothing(volume) && isnothing(min_projection) && throw(ArgumentError(
            "At least one of `volume` or `min_projection` must be requested.",
        ))
        _validate_volume_type(volume)
        _validate_projection_type(min_projection)
        return new(count, volume, min_projection)
    end
end

function _validate_volume_type(T)
    isnothing(T) && return nothing
    isconcretetype(T) || throw(ArgumentError("Volume output type must be concrete. Got $T."))
    T in (N0f8, Float32, ComplexF32, ComplexF64) ||
        throw(ArgumentError("Volume output must be N0f8, Float32, ComplexF32, ComplexF64, or nothing. Got $T."))
    return nothing
end

function _validate_projection_type(T)
    isnothing(T) && return nothing
    isconcretetype(T) || throw(ArgumentError("MinIP output type must be concrete. Got $T."))
    T in (N0f8, Float32) ||
        throw(ArgumentError("MinIP output must be N0f8, Float32, or nothing. Got $T."))
    return nothing
end

"""Outputs generated together by one reconstruction depth scan."""
struct ReconstructionResult{V,P}
    volume::V
    min_projection::P
end

"""
Conservative memory estimate for a reconstruction plan and output request.
`required_bytes` is the new allocation covered by `scope`; `safe` is `missing`
when the backend cannot report available memory.
"""
struct MemoryDiagnostic
    backend::Symbol
    memory_kind::Symbol
    scope::Symbol
    working_shape::Tuple{Int,Int}
    output_shape::Tuple{Int,Int}
    slices::Int
    plan_bytes::Int
    output_bytes::Int
    temporary_bytes::Int
    required_bytes::Int
    available_bytes::Union{Nothing,Int}
    safety_factor::Float64
    safe::Union{Missing,Bool}
end

function _format_memory(bytes::Integer)
    value = Float64(bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit = 1
    while value >= 1024 && unit < length(units)
        value /= 1024
        unit += 1
    end
    return string(round(value; digits=unit == 1 ? 0 : 2), " ", units[unit])
end

function Base.show(io::IO, diagnostic::MemoryDiagnostic)
    print(io, "MemoryDiagnostic(", diagnostic.backend,
          ", required=", _format_memory(diagnostic.required_bytes),
          ", available=", isnothing(diagnostic.available_bytes) ? "unknown" :
                           _format_memory(diagnostic.available_bytes),
          ", safe=", diagnostic.safe, ")")
end

function _byte_count(shape::Tuple{Int,Int}, slices::Int, T::DataType)
    try
        return Base.checked_mul(Base.checked_mul(shape[1], shape[2]),
                                Base.checked_mul(slices, sizeof(T)))
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError("Requested array size exceeds addressable memory."))
    end
end

function _checked_add_bytes(values::Integer...)
    try
        return foldl(Base.checked_add, values; init=0)
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError("Requested arrays exceed addressable memory."))
    end
end

function _plan_memory_bytes(shape::Tuple{Int,Int})
    # Four resident complex planes, one intensity plane, and two conservative
    # complex-plane equivalents for backend-specific FFT workspaces.
    return _checked_add_bytes(_byte_count(shape, 1, NTuple{6,ComplexF32}),
                              _byte_count(shape, 1, Float32))
end

function _output_memory_bytes(request::Union{Nothing,ReconstructionRequest},
                              output_shape::Tuple{Int,Int})
    isnothing(request) && return (0, 0)
    output = 0
    temporary = 0
    if !isnothing(request.volume)
        output = _checked_add_bytes(
            output,
            _byte_count(output_shape, request.slices, request.volume),
        )
    end
    if !isnothing(request.min_projection)
        output = _checked_add_bytes(
            output,
            _byte_count(output_shape, 1, request.min_projection),
        )
        request.min_projection === N0f8 &&
            (temporary = _checked_add_bytes(
                temporary,
                _byte_count(output_shape, 1, Float32),
            ))
    end
    return output, temporary
end

"""
    memory_diagnostic(backend, working_shape, request;
                      output_shape=working_shape, include_plan=true,
                      safety_factor=1.2)

Estimate new plan, output, and conversion allocations before reconstruction.
The estimate is conservative because FFT libraries do not expose every
internal allocation. Metal reports unified host/device memory.
"""
function memory_diagnostic(b::AbstractBackend, working_shape::Tuple{Int,Int},
                           request::Union{Nothing,ReconstructionRequest}=nothing;
                           output_shape::Tuple{Int,Int}=working_shape,
                           include_plan::Bool=true,
                           safety_factor::Real=1.2,
                           available_bytes::Union{Nothing,Integer}=_available_memory(b))
    _validate_shape(working_shape)
    _validate_shape(output_shape)
    all(output_shape .<= working_shape) || throw(ArgumentError(
        "output_shape must fit inside working_shape; got $output_shape and $working_shape.",
    ))
    safety = Float64(safety_factor)
    isfinite(safety) && safety >= 1 || throw(ArgumentError(
        "safety_factor must be finite and at least 1. Got $safety_factor.",
    ))
    plan_bytes = include_plan ? _plan_memory_bytes(working_shape) : 0
    output_bytes, temporary_bytes = _output_memory_bytes(request, output_shape)
    working_shape != output_shape &&
        (temporary_bytes = _checked_add_bytes(
            temporary_bytes,
            _byte_count(working_shape, 1, ComplexF32),
        ))
    if !isnothing(request) && request.volume === ComplexF64 && b isa MetalBackend
        temporary_bytes = _checked_add_bytes(
            temporary_bytes,
            _byte_count(output_shape, 1, ComplexF32),
        )
    end
    required = _checked_add_bytes(plan_bytes, output_bytes, temporary_bytes)
    available = isnothing(available_bytes) ? nothing : Int(available_bytes)
    !isnothing(available) && available >= 0 || isnothing(available) ||
        throw(ArgumentError("available_bytes cannot be negative."))
    safe = isnothing(available) ? missing : required * safety <= available
    return MemoryDiagnostic(backend_name(b), _memory_kind(b),
                            include_plan ? :plan_and_outputs : :outputs,
                            working_shape, output_shape,
                            isnothing(request) ? 0 : request.slices,
                            plan_bytes, output_bytes, temporary_bytes, required,
                            available, safety, safe)
end

function _phase_memory_diagnostic(b::AbstractBackend, shape::Tuple{Int,Int};
                                  safety_factor::Real=1.2,
                                  available_bytes::Union{Nothing,Integer}=_available_memory(b))
    _validate_shape(shape)
    safety = Float64(safety_factor)
    isfinite(safety) && safety >= 1 || throw(ArgumentError(
        "safety_factor must be finite and at least 1. Got $safety_factor.",
    ))
    # Two kernels, four resident complex work planes, two FFT workspace plane
    # equivalents, four real work planes, and two converted input planes.
    plan_bytes = _checked_add_bytes(
        _byte_count(shape, 1, NTuple{8,ComplexF32}),
        _byte_count(shape, 1, NTuple{6,Float32}),
    )
    available = isnothing(available_bytes) ? nothing : Int(available_bytes)
    !isnothing(available) && available >= 0 || isnothing(available) ||
        throw(ArgumentError("available_bytes cannot be negative."))
    safe = isnothing(available) ? missing : plan_bytes * safety <= available
    return MemoryDiagnostic(backend_name(b), _memory_kind(b), :phase_plan,
                            shape, shape, 0, plan_bytes, 0, 0, plan_bytes,
                            available, safety, safe)
end

"""Return a conservative memory estimate for an existing phase-retrieval plan."""
function memory_diagnostic(plan::PhaseRetrievalPlan;
                           safety_factor::Real=plan.memory_safety_factor,
                           available_bytes::Union{Nothing,Integer}=_available_memory(plan.backend))
    return _phase_memory_diagnostic(plan.backend, size(plan.forward_transfer);
                                    safety_factor, available_bytes)
end

function _enforce_memory(diagnostic::MemoryDiagnostic, check_memory::Bool)
    check_memory || return diagnostic
    diagnostic.safe === false && throw(ArgumentError(
        "The requested operation is estimated to require $(_format_memory(diagnostic.required_bytes)) " *
        "of new $(diagnostic.memory_kind) memory with a ×$(diagnostic.safety_factor) safety " *
        "factor, but only $(_format_memory(something(diagnostic.available_bytes, 0))) is " *
        "available. Reduce image size/slices, request a smaller output type, omit an output, " *
        "or pass `check_memory=false` to override this diagnostic.",
    ))
    return diagnostic
end

"""Reusable FFT plans and buffers for a uniformly-spaced reconstruction volume."""
mutable struct ReconstructionPlan{B<:AbstractBackend,A<:AbstractMatrix,R<:AbstractMatrix,PF,PI}
    backend::B
    front_transfer::A
    slice_transfer::A
    frequency::A
    propagated::A
    intensity::R
    fft_plan::PF
    ifft_plan::PI
    request::Union{Nothing,ReconstructionRequest}
    check_memory::Bool
    memory_safety_factor::Float64
end

function ReconstructionPlan(b::AbstractBackend, front::PropagationKernel,
                            step::PropagationKernel;
                            request::Union{Nothing,ReconstructionRequest}=nothing,
                            check_memory::Bool=true,
                            memory_safety_factor::Real=1.2,
                            output_shape::Tuple{Int,Int}=size(front),
                            available_memory::Union{Nothing,Integer}=_available_memory(b))
    size(front) == size(step) || throw(DimensionMismatch("Front and slice transfers must have the same size. Got $(size(front)) and $(size(step))."))
    diagnostic = memory_diagnostic(b, size(front), request;
                                   output_shape,
                                   include_plan=true,
                                   safety_factor=memory_safety_factor,
                                   available_bytes=available_memory)
    _enforce_memory(diagnostic, check_memory)
    _activate!(b)
    tf = ComplexF32.(to_backend(b, front.data))
    ts = ComplexF32.(to_backend(b, step.data))
    frequency = similar(tf)
    propagated = similar(tf)
    intensity = similar(tf, Float32)
    fill!(propagated, 0)
    fft_plan = plan_fft(propagated)
    ifft_plan = plan_ifft(propagated)
    return ReconstructionPlan(b, tf, ts, frequency, propagated, intensity,
                              fft_plan, ifft_plan, request, check_memory,
                              Float64(memory_safety_factor))
end

ReconstructionPlan(front::PropagationKernel, step::PropagationKernel;
                   backend::AbstractBackend=backendof(front), kwargs...) =
    ReconstructionPlan(backend, front, step; kwargs...)

function memory_diagnostic(plan::ReconstructionPlan,
                           request::Union{Nothing,ReconstructionRequest}=plan.request;
                           output_shape::Tuple{Int,Int}=size(plan.intensity),
                           include_plan::Bool=false,
                           safety_factor::Real=plan.memory_safety_factor,
                           available_bytes::Union{Nothing,Integer}=_available_memory(plan.backend))
    return memory_diagnostic(plan.backend, size(plan.intensity), request;
                             output_shape, include_plan, safety_factor,
                             available_bytes)
end

function _reset_frequency!(plan::ReconstructionPlan, wavefront::Wavefront)
    size(wavefront) == size(plan.front_transfer) ||
        throw(DimensionMismatch("Wavefront and transfers must have size $(size(plan.front_transfer)); got $(size(wavefront))."))
    _activate!(plan.backend)
    data = to_backend(plan.backend, wavefront.data)
    eltype(data) === ComplexF32 || (data = ComplexF32.(data))
    mul!(plan.frequency, plan.fft_plan, data)
    plan.frequency .*= plan.front_transfer
    return nothing
end

function _next_slice!(plan::ReconstructionPlan, slice::Int; intensity::Bool=true)
    slice > 1 && (plan.frequency .*= plan.slice_transfer)
    mul!(plan.propagated, plan.ifft_plan, plan.frequency)
    intensity && (plan.intensity .= Float32.(abs2.(plan.propagated)))
    return nothing
end

@inline function _normedfloat_to_N0f8(value::AbstractFloat)
    return reinterpret(N0f8, round(UInt8, clamp(value, 0, 1) * 255))
end

function _copy_intensity!(output::AbstractMatrix, intensity::AbstractMatrix; clamp_output::Bool)
    T = eltype(output)
    if T <: AbstractFloat
        if clamp_output
            output .= T.(clamp.(intensity, 0, 1))
        else
            output .= T.(intensity)
        end
    elseif T === N0f8
        output .= _normedfloat_to_N0f8.(intensity)
    else
        throw(ArgumentError("Reconstruction output type must be an AbstractFloat or N0f8. Got $T."))
    end
    return nothing
end

function _validate_slices(slices::Integer)
    slices > 0 || throw(ArgumentError("slices must be positive. Got $slices."))
    return Int(slices)
end

function _center_ranges(working_shape::Tuple{Int,Int}, output_shape::Tuple{Int,Int})
    all(output_shape .<= working_shape) || throw(DimensionMismatch(
        "Output plane $output_shape does not fit inside working plane $working_shape.",
    ))
    first_row = fld(working_shape[1] - output_shape[1], 2) + 1
    first_col = fld(working_shape[2] - output_shape[2], 2) + 1
    return (first_row:first_row + output_shape[1] - 1,
            first_col:first_col + output_shape[2] - 1)
end

function _allocate_volume(plan::ReconstructionPlan, T::DataType,
                          output_shape::Tuple{Int,Int}, slices::Int)
    dims = (output_shape..., slices)
    if T === ComplexF64 && plan.backend isa MetalBackend
        # Apple GPU arithmetic is ComplexF32. Preserve the requested storage
        # type on the host without trying to execute Float64 Metal kernels.
        return Array{ComplexF64}(undef, dims)
    elseif T <: Complex
        return similar(plan.propagated, T, dims)
    end
    return similar(plan.intensity, T, dims)
end

function _allocate_projection(plan::ReconstructionPlan, T::DataType,
                              output_shape::Tuple{Int,Int})
    if T === N0f8
        result = similar(plan.intensity, N0f8, output_shape)
        work = similar(plan.intensity, Float32, output_shape)
        return result, work
    end
    result = similar(plan.intensity, Float32, output_shape)
    return result, result
end

function _copy_complex_slice!(destination::AbstractMatrix, source::AbstractMatrix,
                              staging::Union{Nothing,Matrix{ComplexF32}})
    if isnothing(staging)
        destination .= eltype(destination).(source)
    else
        _copy_to_host!(staging, source)
        destination .= ComplexF64.(staging)
    end
    return nothing
end

function _reconstruct_result(plan::ReconstructionPlan, wavefront::Wavefront,
                             request::ReconstructionRequest;
                             output_shape::Tuple{Int,Int}=size(plan.intensity),
                             clamp_output::Bool=false,
                             check_memory::Bool=plan.check_memory)
    size(wavefront) == size(plan.intensity) || throw(DimensionMismatch(
        "Wavefront and plan must have the same working size $(size(plan.intensity)); got $(size(wavefront)).",
    ))
    diagnostic = memory_diagnostic(plan, request;
                                   output_shape,
                                   include_plan=false,
                                   safety_factor=plan.memory_safety_factor)
    _enforce_memory(diagnostic, check_memory)

    volume = isnothing(request.volume) ? nothing :
             _allocate_volume(plan, request.volume, output_shape, request.slices)
    projection, projection_work = isnothing(request.min_projection) ?
                                  (nothing, nothing) :
                                  _allocate_projection(plan, request.min_projection,
                                                       output_shape)
    rows, cols = _center_ranges(size(plan.intensity), output_shape)
    complex_staging = request.volume === ComplexF64 && plan.backend isa MetalBackend ?
                      Matrix{ComplexF32}(undef, output_shape) : nothing
    needs_intensity = !isnothing(request.min_projection) ||
                      (!isnothing(request.volume) && request.volume <: Real)

    _reset_frequency!(plan, wavefront)
    for z in 1:request.slices
        _next_slice!(plan, z; intensity=needs_intensity)
        if !isnothing(volume)
            if request.volume <: Complex
                @views _copy_complex_slice!(volume[:, :, z],
                                            plan.propagated[rows, cols],
                                            complex_staging)
            else
                @views _copy_intensity!(volume[:, :, z],
                                        plan.intensity[rows, cols];
                                        clamp_output)
            end
        end
        if !isnothing(projection_work)
            if z == 1
                @views projection_work .= plan.intensity[rows, cols]
            else
                @views projection_work .= min.(projection_work,
                                               plan.intensity[rows, cols])
            end
        end
    end
    if request.min_projection === N0f8
        _copy_intensity!(projection, projection_work; clamp_output=true)
    end
    return ReconstructionResult(volume, projection)
end

"""
    reconstruct(plan, wavefront, request)

Generate the requested volume and MinIP in one propagation pass. Access the
outputs as `result.volume` and `result.min_projection`; an unrequested output is
`nothing`. Set `check_memory=false` only when the conservative preflight should
not stop allocation.
"""
function reconstruct(plan::ReconstructionPlan, wavefront::Wavefront,
                     request::ReconstructionRequest;
                     clamp_output::Bool=false,
                     check_memory::Bool=plan.check_memory)
    return _reconstruct_result(plan, wavefront, request;
                               clamp_output, check_memory)
end

function reconstruct(plan::ReconstructionPlan, wavefront::Wavefront;
                     clamp_output::Bool=false,
                     check_memory::Bool=plan.check_memory)
    isnothing(plan.request) && throw(ArgumentError(
        "This plan has no ReconstructionRequest. Pass a request to `reconstruct` " *
        "or construct the plan with `request=...`.",
    ))
    return reconstruct(plan, wavefront, plan.request; clamp_output, check_memory)
end

"""
    reconstruct_padded(plan, wavefront[, request]; mode=:mean)

Pad `wavefront` to the plan shape and reconstruct only the original central
field of view. The propagation uses the padded plane, but no padded 3-D output
volume is allocated.
"""
function reconstruct_padded(plan::ReconstructionPlan, wavefront::Wavefront,
                            request::Union{Nothing,ReconstructionRequest}=plan.request;
                            mode::Symbol=:mean,
                            clamp_output::Bool=false,
                            check_memory::Bool=plan.check_memory)
    isnothing(request) && throw(ArgumentError(
        "Pass a ReconstructionRequest or construct the plan with `request=...`.",
    ))
    all(size(wavefront) .<= size(plan.intensity)) || throw(DimensionMismatch(
        "Wavefront $(size(wavefront)) does not fit inside plan $(size(plan.intensity)).",
    ))
    diagnostic = memory_diagnostic(plan, request;
                                   output_shape=size(wavefront),
                                   include_plan=false,
                                   safety_factor=plan.memory_safety_factor)
    _enforce_memory(diagnostic, check_memory)
    source = to_backend(plan.backend, wavefront.data)
    eltype(source) === ComplexF32 || (source = ComplexF32.(source))
    padded = Wavefront(pad2d(source, size(plan.intensity); mode))
    return _reconstruct_result(plan, padded, request;
                               output_shape=size(wavefront),
                               clamp_output,
                               check_memory=false)
end

"""Reconstruct intensity into a preallocated `(height, width, slices)` array."""
function reconstruct!(volume::AbstractArray{<:Real,3}, plan::ReconstructionPlan,
                      wavefront::Wavefront; clamp_output::Bool=false)
    slices = _validate_slices(size(volume, 3))
    size(volume)[1:2] == size(plan.front_transfer) ||
        throw(DimensionMismatch("Volume plane size must be $(size(plan.front_transfer)); got $(size(volume)[1:2])."))
    _reset_frequency!(plan, wavefront)
    for z in 1:slices
        _next_slice!(plan, z)
        @views _copy_intensity!(volume[:, :, z], plan.intensity; clamp_output)
    end
    return volume
end

"""Allocate and return a reconstructed intensity volume using a reusable plan."""
function reconstruct(plan::ReconstructionPlan, wavefront::Wavefront, slices::Integer;
                     output::Type=Float32, clamp_output::Bool=false,
                     check_memory::Bool=plan.check_memory)
    request = ReconstructionRequest(slices; volume=output, min_projection=nothing)
    return reconstruct(plan, wavefront, request;
                       clamp_output, check_memory).volume
end

function reconstruct(b::AbstractBackend, wavefront::Wavefront,
                     front::PropagationKernel, step::PropagationKernel,
                     slices::Integer; kwargs...)
    return reconstruct(ReconstructionPlan(b, front, step), wavefront, slices; kwargs...)
end

"""Reconstruct complex amplitude into a preallocated volume."""
function reconstruct_complex!(volume::AbstractArray{<:Complex,3}, plan::ReconstructionPlan,
                              wavefront::Wavefront)
    slices = _validate_slices(size(volume, 3))
    size(volume)[1:2] == size(plan.front_transfer) || throw(DimensionMismatch("Invalid complex volume plane size."))
    _reset_frequency!(plan, wavefront)
    for z in 1:slices
        _next_slice!(plan, z; intensity=false)
        @views volume[:, :, z] .= plan.propagated
    end
    return volume
end

"""Allocate and return the complex-amplitude reconstruction volume."""
function reconstruct_complex(plan::ReconstructionPlan, wavefront::Wavefront,
                             slices::Integer; output::Type=ComplexF64,
                             check_memory::Bool=plan.check_memory)
    request = ReconstructionRequest(slices; volume=output, min_projection=nothing)
    return reconstruct(plan, wavefront, request; check_memory).volume
end

"""Compute the minimum-intensity projection along reconstruction depth."""
function xyprojection!(projection::AbstractMatrix{<:AbstractFloat}, plan::ReconstructionPlan,
                       wavefront::Wavefront, slices::Integer)
    n = _validate_slices(slices)
    size(projection) == size(plan.intensity) || throw(DimensionMismatch("Projection must have size $(size(plan.intensity))."))
    _reset_frequency!(plan, wavefront)
    for z in 1:n
        _next_slice!(plan, z)
        if z == 1
            projection .= plan.intensity
        else
            projection .= min.(projection, plan.intensity)
        end
    end
    return projection
end

"""Allocate and return the minimum-intensity projection over reconstruction depth."""
function xyprojection(plan::ReconstructionPlan, wavefront::Wavefront,
                      slices::Integer; output::Type=Float32,
                      check_memory::Bool=plan.check_memory)
    request = ReconstructionRequest(slices; volume=nothing, min_projection=output)
    return reconstruct(plan, wavefront, request; check_memory).min_projection
end

"""Fill preallocated intensity `volume` and minimum-intensity `projection`."""
function reconstruct_and_projection!(volume::AbstractArray{<:Real,3},
                                     projection::AbstractMatrix{<:AbstractFloat},
                                     plan::ReconstructionPlan, wavefront::Wavefront;
                                     clamp_output::Bool=false)
    slices = _validate_slices(size(volume, 3))
    size(volume)[1:2] == size(projection) == size(plan.intensity) ||
        throw(DimensionMismatch("Volume, projection, and plan plane sizes must match."))
    _reset_frequency!(plan, wavefront)
    for z in 1:slices
        _next_slice!(plan, z)
        @views _copy_intensity!(volume[:, :, z], plan.intensity; clamp_output)
        if z == 1
            projection .= plan.intensity
        else
            projection .= min.(projection, plan.intensity)
        end
    end
    return volume, projection
end

"""Allocate and return both an intensity volume and its depth projection."""
function reconstruct_and_projection(plan::ReconstructionPlan, wavefront::Wavefront,
                                    slices::Integer; output::Type=Float32,
                                    projection_output::Type=Float32,
                                    clamp_output::Bool=false,
                                    check_memory::Bool=plan.check_memory)
    request = ReconstructionRequest(slices;
                                    volume=output,
                                    min_projection=projection_output)
    result = reconstruct(plan, wavefront, request; clamp_output, check_memory)
    return result.volume, result.min_projection
end

function reconstruct_and_projection_padded(b::AbstractBackend, wavefront::Wavefront,
                                           front::PropagationKernel,
                                           step::PropagationKernel,
                                           slices::Integer; output::Type=Float32,
                                           projection_output::Type=Float32,
                                           clamp_output::Bool=false,
                                           check_memory::Bool=true)
    expected = (2 * size(wavefront, 1), 2 * size(wavefront, 2))
    size(front) == size(step) == expected ||
        throw(DimensionMismatch("Padded transfers must have size $expected; got $(size(front)) and $(size(step))."))
    request = ReconstructionRequest(slices;
                                    volume=output,
                                    min_projection=projection_output)
    plan = ReconstructionPlan(b, front, step;
                              request,
                              check_memory,
                              output_shape=size(wavefront))
    result = reconstruct_padded(plan, wavefront, request;
                                clamp_output,
                                check_memory)
    return result.volume, result.min_projection
end

"""Propagate `input` into the preallocated `output` wavefront."""
function asm_propagate!(output::Wavefront, input::Wavefront,
                        grid::PropagationGrid, distance::Real,
                        wavelength::Real)
    size(output) == size(input) == size(grid) || throw(DimensionMismatch("Wavefronts and propagation grid must have the same size."))
    b = backendof(input)
    tf = propagation_kernel(b, distance, wavelength, grid)
    source = ComplexF32.(to_backend(b, input.data))
    frequency = similar(source)
    fft_plan = plan_fft(source)
    ifft_plan = plan_ifft(source)
    mul!(frequency, fft_plan, source)
    frequency .*= tf.data
    mul!(output.data, ifft_plan, frequency)
    return output
end

"""Allocate a wavefront and propagate `input` by one angular-spectrum step."""
function asm_propagate(input::Wavefront, grid::PropagationGrid,
                       distance::Real, wavelength::Real)
    output = Wavefront(similar(input.data))
    return asm_propagate!(output, input, grid, distance, wavelength)
end

function _legacy(old::Symbol, new::Symbol)
    Base.depwarn("`$old` is a v0.2 CUDA-specific name; use backend-neutral `$new` in v1.", old)
    return nothing
end

# ---------------------------------------------------------------------------
# v0.2 compatibility wrappers

function cu_transfer_sqrt_arr(datlen::Int, wavelength::Real, pixel_pitch::Real)
    _legacy(:cu_transfer_sqrt_arr, :propagation_grid)
    return propagation_grid(backend(:cuda), datlen, wavelength, pixel_pitch)
end

function cu_transfer(distance::Real, datlen::Int, wavelength::Real,
                     grid::PropagationGrid)
    _legacy(:cu_transfer, :propagation_kernel)
    size(grid) == (datlen, datlen) || throw(DimensionMismatch("datlen does not match grid."))
    return propagation_kernel(backend(:cuda), distance, wavelength, grid)
end

function cu_gabor_wavefront(hologram::AbstractMatrix)
    _legacy(:cu_gabor_wavefront, :gabor_wavefront)
    return gabor_wavefront(backend(:cuda), hologram)
end

function cu_phase_retrieval_holo(hologram1::AbstractMatrix, hologram2::AbstractMatrix,
                                 forward::PropagationKernel,
                                 backward::PropagationKernel,
                                 iterations::Int, datlen::Int)
    _legacy(:cu_phase_retrieval_holo, :phase_retrieval)
    size(hologram1) == (datlen, datlen) || throw(DimensionMismatch("datlen does not match holograms."))
    b = backend(:cuda)
    return phase_retrieval(b, hologram1, hologram2, forward, backward; iterations)
end

cu_2d_pad(input) = (_legacy(:cu_2d_pad, :pad2d); pad2d(input))

function cu_get_reconst_vol(wavefront::Wavefront, front::PropagationKernel,
                            step::PropagationKernel,
                            slices::Int, output::Type=N0f8)
    _legacy(:cu_get_reconst_vol, :reconstruct)
    plan = ReconstructionPlan(backendof(wavefront), front, step)
    return reconstruct(plan, wavefront, slices; output, clamp_output=true)
end

function cu_get_reconst_complex_vol(wavefront::Wavefront,
                                    front::PropagationKernel,
                                    step::PropagationKernel, slices::Int)
    _legacy(:cu_get_reconst_complex_vol, :reconstruct_complex)
    return reconstruct_complex(ReconstructionPlan(backendof(wavefront), front, step),
                               wavefront, slices; output=ComplexF32)
end

function cu_get_reconst_xyprojection(wavefront::Wavefront,
                                     front::PropagationKernel,
                                     step::PropagationKernel, slices::Int)
    _legacy(:cu_get_reconst_xyprojection, :xyprojection)
    return xyprojection(ReconstructionPlan(backendof(wavefront), front, step), wavefront, slices)
end

function cu_get_reconst_vol_and_xyprojection(wavefront::Wavefront,
                                             front::PropagationKernel,
                                             step::PropagationKernel, slices::Int,
                                             output::Type=N0f8)
    _legacy(:cu_get_reconst_vol_and_xyprojection, :reconstruct_and_projection)
    plan = ReconstructionPlan(backendof(wavefront), front, step)
    return reconstruct_and_projection(plan, wavefront, slices; output, clamp_output=true)
end

function cu_get_reconst_vol_and_xyprojection_padded(wavefront::Wavefront,
                                                    front::PropagationKernel,
                                                    step::PropagationKernel,
                                                    slices::Int, output::Type=N0f8)
    _legacy(:cu_get_reconst_vol_and_xyprojection_padded, :reconstruct_and_projection_padded)
    return reconstruct_and_projection_padded(backendof(wavefront), wavefront, front, step,
                                             slices; output, clamp_output=true)
end

function cu_asm_prop!(output::Wavefront, input::Wavefront,
                      grid::PropagationGrid, distance::Real,
                      datlen::Int, wavelength::Real)
    _legacy(:cu_asm_prop!, :asm_propagate!)
    size(input) == (datlen, datlen) || throw(DimensionMismatch("datlen does not match wavefront."))
    asm_propagate!(output, input, grid, distance, wavelength)
    return nothing
end
