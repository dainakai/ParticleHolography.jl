using AbstractFFTs
using FFTW
using FixedPointNumbers: N0f8
using LinearAlgebra: mul!
using Statistics: mean

export transfer_sqrt, transfer, gabor_wavefront, pad2d
export PhaseRetrievalPlan, phase_retrieval!, phase_retrieval
export ReconstructionPlan, reconstruct!, reconstruct, reconstruct_complex!, reconstruct_complex
export xyprojection!, xyprojection, reconstruct_and_projection!, reconstruct_and_projection
export reconstruct_and_projection_padded, asm_propagate!, asm_propagate
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
    transfer_sqrt(backend, shape, wavelength, pixel_pitch)

Construct the square-root term of the angular-spectrum transfer function in
FFT-native order. `shape` may be a square side length or `(height, width)`.
All length parameters must use the same unit (for example, micrometres).
"""
function transfer_sqrt(b::AbstractBackend, shape::Tuple{Int,Int}, wavelength::Real, pixel_pitch::Real)
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
    return TransferSqrtPart(to_backend(b, host))
end

transfer_sqrt(b::AbstractBackend, datlen::Int, wavelength::Real, pixel_pitch::Real) =
    transfer_sqrt(b, _validate_shape(datlen), wavelength, pixel_pitch)

transfer_sqrt(shape, wavelength::Real, pixel_pitch::Real; backend::AbstractBackend=CPUBackend()) =
    transfer_sqrt(backend, shape, wavelength, pixel_pitch)

"""
    transfer(backend, distance, wavelength, sqrt_part)

Construct an angular-spectrum transfer for `distance`. The transfer is stored
in the order consumed directly by `plan_fft`; no shift is needed in hot loops.
"""
function transfer(b::AbstractBackend, distance::Real, wavelength::Real, sqrt_part::TransferSqrtPart)
    isfinite(distance) || throw(ArgumentError("distance must be finite. Got $distance."))
    isfinite(wavelength) && wavelength > 0 || throw(ArgumentError("wavelength must be finite and positive. Got $wavelength."))
    phase_scale = 2π * Float64(distance) / Float64(wavelength)
    sqrt_host = to_host(sqrt_part.data)
    minimum(sqrt_host) >= 0 || throw(DomainError(minimum(sqrt_host), "transfer square-root term contains negative values."))
    host = ComplexF32.(exp.(complex.(0.0, phase_scale .* sqrt.(Float64.(sqrt_host)))))
    return Transfer(to_backend(b, host))
end

transfer(distance::Real, wavelength::Real, sqrt_part::TransferSqrtPart;
         backend::AbstractBackend=backendof(sqrt_part)) =
    transfer(backend, distance, wavelength, sqrt_part)

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

gabor_wavefront(hologram::AbstractMatrix; backend::AbstractBackend=backendof(hologram)) =
    gabor_wavefront(backend, hologram)

"""Pad a 2-D array to twice its dimensions, using its mean as the border value."""
function pad2d(input::AbstractMatrix)
    height, width = size(input)
    output = similar(input, eltype(input), (2 * height, 2 * width))
    fill!(output, convert(eltype(input), mean(to_host(input))))
    rows = fld(height, 2) + 1:fld(height, 2) + height
    cols = fld(width, 2) + 1:fld(width, 2) + width
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
end

function PhaseRetrievalPlan(b::AbstractBackend, forward::Transfer, backward::Transfer)
    size(forward) == size(backward) || throw(DimensionMismatch("Forward and backward transfers must have the same size. Got $(size(forward)) and $(size(backward))."))
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
                              sqrt1, sqrt2, phase1, phase2, fft_plan, ifft_plan)
end

PhaseRetrievalPlan(forward::Transfer, backward::Transfer;
                   backend::AbstractBackend=backendof(forward)) =
    PhaseRetrievalPlan(backend, forward, backward)

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
                         hologram2::AbstractMatrix, forward::Transfer,
                         backward::Transfer; iterations::Integer=10)
    plan = PhaseRetrievalPlan(b, forward, backward)
    return phase_retrieval(plan, hologram1, hologram2; iterations)
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
end

function ReconstructionPlan(b::AbstractBackend, front::Transfer, step::Transfer)
    size(front) == size(step) || throw(DimensionMismatch("Front and slice transfers must have the same size. Got $(size(front)) and $(size(step))."))
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
                              fft_plan, ifft_plan)
end

ReconstructionPlan(front::Transfer, step::Transfer;
                   backend::AbstractBackend=backendof(front)) =
    ReconstructionPlan(backend, front, step)

function _reset_frequency!(plan::ReconstructionPlan, wavefront::Wavefront)
    size(wavefront) == size(plan.front_transfer) ||
        throw(DimensionMismatch("Wavefront and transfers must have size $(size(plan.front_transfer)); got $(size(wavefront))."))
    _activate!(plan.backend)
    data = ComplexF32.(to_backend(plan.backend, wavefront.data))
    mul!(plan.frequency, plan.fft_plan, data)
    plan.frequency .*= plan.front_transfer
    return nothing
end

function _next_slice!(plan::ReconstructionPlan, slice::Int)
    slice > 1 && (plan.frequency .*= plan.slice_transfer)
    mul!(plan.propagated, plan.ifft_plan, plan.frequency)
    plan.intensity .= Float32.(abs2.(plan.propagated))
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
                     output::Type=Float32, clamp_output::Bool=false)
    n = _validate_slices(slices)
    volume = similar(plan.intensity, output, (size(plan.intensity)..., n))
    return reconstruct!(volume, plan, wavefront; clamp_output)
end

function reconstruct(b::AbstractBackend, wavefront::Wavefront, front::Transfer,
                     step::Transfer, slices::Integer; kwargs...)
    return reconstruct(ReconstructionPlan(b, front, step), wavefront, slices; kwargs...)
end

"""Reconstruct complex amplitude into a preallocated volume."""
function reconstruct_complex!(volume::AbstractArray{<:Complex,3}, plan::ReconstructionPlan,
                              wavefront::Wavefront)
    slices = _validate_slices(size(volume, 3))
    size(volume)[1:2] == size(plan.front_transfer) || throw(DimensionMismatch("Invalid complex volume plane size."))
    _reset_frequency!(plan, wavefront)
    for z in 1:slices
        _next_slice!(plan, z)
        @views volume[:, :, z] .= plan.propagated
    end
    return volume
end

"""Allocate and return the complex-amplitude reconstruction volume."""
function reconstruct_complex(plan::ReconstructionPlan, wavefront::Wavefront, slices::Integer)
    n = _validate_slices(slices)
    volume = similar(plan.propagated, ComplexF32, (size(plan.propagated)..., n))
    return reconstruct_complex!(volume, plan, wavefront)
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
function xyprojection(plan::ReconstructionPlan, wavefront::Wavefront, slices::Integer)
    projection = similar(plan.intensity)
    return xyprojection!(projection, plan, wavefront, slices)
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
                                    clamp_output::Bool=false)
    n = _validate_slices(slices)
    volume = similar(plan.intensity, output, (size(plan.intensity)..., n))
    projection = similar(plan.intensity)
    return reconstruct_and_projection!(volume, projection, plan, wavefront; clamp_output)
end

function _center_crop(input::AbstractArray, shape::Tuple{Int,Int})
    height, width = shape
    firstrow = fld(size(input, 1) - height, 2) + 1
    firstcol = fld(size(input, 2) - width, 2) + 1
    ranges = (firstrow:firstrow + height - 1, firstcol:firstcol + width - 1)
    if ndims(input) == 2
        output = similar(input, eltype(input), shape)
        @views output .= input[ranges...]
    else
        output = similar(input, eltype(input), (height, width, size(input, 3)))
        @views output .= input[ranges..., :]
    end
    return output
end

function reconstruct_and_projection_padded(b::AbstractBackend, wavefront::Wavefront,
                                           front::Transfer, step::Transfer,
                                           slices::Integer; output::Type=Float32,
                                           clamp_output::Bool=false)
    expected = (2 * size(wavefront, 1), 2 * size(wavefront, 2))
    size(front) == size(step) == expected ||
        throw(DimensionMismatch("Padded transfers must have size $expected; got $(size(front)) and $(size(step))."))
    padded = Wavefront(pad2d(ComplexF32.(to_backend(b, wavefront.data))))
    plan = ReconstructionPlan(b, front, step)
    volume, projection = reconstruct_and_projection(plan, padded, slices; output, clamp_output)
    return _center_crop(volume, size(wavefront)), _center_crop(projection, size(wavefront))
end

"""Propagate `input` into the preallocated `output` wavefront."""
function asm_propagate!(output::Wavefront, input::Wavefront,
                        sqrt_part::TransferSqrtPart, distance::Real,
                        wavelength::Real)
    size(output) == size(input) == size(sqrt_part) || throw(DimensionMismatch("Wavefronts and transfer term must have the same size."))
    b = backendof(input)
    tf = transfer(b, distance, wavelength, sqrt_part)
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
function asm_propagate(input::Wavefront, sqrt_part::TransferSqrtPart,
                       distance::Real, wavelength::Real)
    output = Wavefront(similar(input.data))
    return asm_propagate!(output, input, sqrt_part, distance, wavelength)
end

function _legacy(old::Symbol, new::Symbol)
    Base.depwarn("`$old` is a v0.2 CUDA-specific name; use backend-neutral `$new` in v1.", old)
    return nothing
end

# ---------------------------------------------------------------------------
# v0.2 compatibility wrappers

function cu_transfer_sqrt_arr(datlen::Int, wavelength::Real, pixel_pitch::Real)
    _legacy(:cu_transfer_sqrt_arr, :transfer_sqrt)
    return transfer_sqrt(backend(:cuda), datlen, wavelength, pixel_pitch)
end

function cu_transfer(distance::Real, datlen::Int, wavelength::Real, sqrt_part::TransferSqrtPart)
    _legacy(:cu_transfer, :transfer)
    size(sqrt_part) == (datlen, datlen) || throw(DimensionMismatch("datlen does not match sqrt_part."))
    return transfer(backend(:cuda), distance, wavelength, sqrt_part)
end

function cu_gabor_wavefront(hologram::AbstractMatrix)
    _legacy(:cu_gabor_wavefront, :gabor_wavefront)
    return gabor_wavefront(backend(:cuda), hologram)
end

function cu_phase_retrieval_holo(hologram1::AbstractMatrix, hologram2::AbstractMatrix,
                                 forward::Transfer, backward::Transfer,
                                 iterations::Int, datlen::Int)
    _legacy(:cu_phase_retrieval_holo, :phase_retrieval)
    size(hologram1) == (datlen, datlen) || throw(DimensionMismatch("datlen does not match holograms."))
    b = backend(:cuda)
    return phase_retrieval(b, hologram1, hologram2, forward, backward; iterations)
end

cu_2d_pad(input) = (_legacy(:cu_2d_pad, :pad2d); pad2d(input))

function cu_get_reconst_vol(wavefront::Wavefront, front::Transfer, step::Transfer,
                            slices::Int, output::Type=N0f8)
    _legacy(:cu_get_reconst_vol, :reconstruct)
    plan = ReconstructionPlan(backendof(wavefront), front, step)
    return reconstruct(plan, wavefront, slices; output, clamp_output=true)
end

function cu_get_reconst_complex_vol(wavefront::Wavefront, front::Transfer,
                                    step::Transfer, slices::Int)
    _legacy(:cu_get_reconst_complex_vol, :reconstruct_complex)
    return reconstruct_complex(ReconstructionPlan(backendof(wavefront), front, step), wavefront, slices)
end

function cu_get_reconst_xyprojection(wavefront::Wavefront, front::Transfer,
                                     step::Transfer, slices::Int)
    _legacy(:cu_get_reconst_xyprojection, :xyprojection)
    return xyprojection(ReconstructionPlan(backendof(wavefront), front, step), wavefront, slices)
end

function cu_get_reconst_vol_and_xyprojection(wavefront::Wavefront, front::Transfer,
                                             step::Transfer, slices::Int,
                                             output::Type=N0f8)
    _legacy(:cu_get_reconst_vol_and_xyprojection, :reconstruct_and_projection)
    plan = ReconstructionPlan(backendof(wavefront), front, step)
    return reconstruct_and_projection(plan, wavefront, slices; output, clamp_output=true)
end

function cu_get_reconst_vol_and_xyprojection_padded(wavefront::Wavefront,
                                                    front::Transfer, step::Transfer,
                                                    slices::Int, output::Type=N0f8)
    _legacy(:cu_get_reconst_vol_and_xyprojection_padded, :reconstruct_and_projection_padded)
    return reconstruct_and_projection_padded(backendof(wavefront), wavefront, front, step,
                                             slices; output, clamp_output=true)
end

function cu_asm_prop!(output::Wavefront, input::Wavefront,
                      sqrt_part::TransferSqrtPart, distance::Real,
                      datlen::Int, wavelength::Real)
    _legacy(:cu_asm_prop!, :asm_propagate!)
    size(input) == (datlen, datlen) || throw(DimensionMismatch("datlen does not match wavefront."))
    asm_propagate!(output, input, sqrt_part, distance, wavelength)
    return nothing
end
