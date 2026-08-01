using AbstractFFTs
using FFTW
using LinearAlgebra: mul!

export rectangle_filter, super_gaussian_filter, apply_low_pass_filter, apply_low_pass_filter!
export cu_rectangle_filter, cu_super_gaussian_filter, cu_apply_low_pass_filter, cu_apply_low_pass_filter!

function _validate_filter_parameters(propagation_distance::Real, wavelength::Real,
                                     image_length::Int, pixel_pitch::Real)
    propagation_distance > 0 || throw(ArgumentError("propagation_distance must be positive."))
    wavelength > 0 || throw(ArgumentError("wavelength must be positive."))
    image_length > 0 || throw(ArgumentError("image_length must be positive."))
    pixel_pitch > 0 || throw(ArgumentError("pixel_pitch must be positive."))
    return nothing
end

"""Construct the rectangular angular-spectrum low-pass filter from Fugal (2009)."""
function rectangle_filter(b::AbstractBackend, propagation_distance::Real,
                          wavelength::Real, image_length::Int, pixel_pitch::Real)
    _validate_filter_parameters(propagation_distance, wavelength, image_length, pixel_pitch)
    cutoff = image_length^2 * Float64(pixel_pitch)^2 /
             (Float64(wavelength) * sqrt(4 * Float64(propagation_distance)^2 +
                                         image_length^2 * Float64(pixel_pitch)^2))
    host = Matrix{Float32}(undef, image_length, image_length)
    for j in 1:image_length, i in 1:image_length
        fy = _fft_frequency_index(i, image_length)
        fx = _fft_frequency_index(j, image_length)
        host[i, j] = fx^2 + fy^2 < cutoff^2 ? 1.0f0 : 0.0f0
    end
    return LowPassFilter(to_backend(b, host))
end

rectangle_filter(propagation_distance::Real, wavelength::Real, image_length::Int,
                 pixel_pitch::Real; backend::AbstractBackend=CPUBackend()) =
    rectangle_filter(backend, propagation_distance, wavelength, image_length, pixel_pitch)

"""Construct the sixth-order super-Gaussian low-pass filter from Fugal (2009)."""
function super_gaussian_filter(b::AbstractBackend, propagation_distance::Real,
                               wavelength::Real, image_length::Int, pixel_pitch::Real)
    _validate_filter_parameters(propagation_distance, wavelength, image_length, pixel_pitch)
    n = image_length
    dx = Float64(pixel_pitch)
    cutoff = n^2 * dx^2 /
             (Float64(wavelength) * sqrt(4 * Float64(propagation_distance)^2 + n^2 * dx^2))
    sigma = cutoff / (n * dx) / (2 * log(2.0))^(1 / 6)
    host = Matrix{Float32}(undef, n, n)
    for j in 1:n, i in 1:n
        fy = _fft_frequency_index(i, n) / (n * dx)
        fx = _fft_frequency_index(j, n) / (n * dx)
        host[i, j] = Float32(exp(-0.5 * ((fx / sigma)^2 + (fy / sigma)^2)^3))
    end
    return LowPassFilter(to_backend(b, host))
end

super_gaussian_filter(propagation_distance::Real, wavelength::Real,
                      image_length::Int, pixel_pitch::Real;
                      backend::AbstractBackend=CPUBackend()) =
    super_gaussian_filter(backend, propagation_distance, wavelength, image_length, pixel_pitch)

"""Apply an FFT-native low-pass filter in place to a wavefront."""
function apply_low_pass_filter!(wavefront::Wavefront, filter::LowPassFilter)
    size(wavefront) == size(filter) || throw(DimensionMismatch("Wavefront and filter must have the same size."))
    b = backendof(wavefront)
    _activate!(b)
    filter_data = Float32.(to_backend(b, filter.data))
    frequency = similar(wavefront.data)
    fft_plan = plan_fft(wavefront.data)
    ifft_plan = plan_ifft(wavefront.data)
    mul!(frequency, fft_plan, wavefront.data)
    frequency .*= filter_data
    mul!(wavefront.data, ifft_plan, frequency)
    return wavefront
end

function apply_low_pass_filter(wavefront::Wavefront, filter::LowPassFilter)
    output = copy(wavefront)
    return apply_low_pass_filter!(output, filter)
end

function cu_rectangle_filter(propagation_distance::Real, wavelength::Real,
                             image_length::Int, pixel_pitch::Real)
    _legacy(:cu_rectangle_filter, :rectangle_filter)
    return rectangle_filter(backend(:cuda), propagation_distance, wavelength,
                            image_length, pixel_pitch)
end

function cu_super_gaussian_filter(propagation_distance::Real, wavelength::Real,
                                  image_length::Int, pixel_pitch::Real)
    _legacy(:cu_super_gaussian_filter, :super_gaussian_filter)
    return super_gaussian_filter(backend(:cuda), propagation_distance, wavelength,
                                 image_length, pixel_pitch)
end

function cu_apply_low_pass_filter!(wavefront::Wavefront, filter::LowPassFilter)
    _legacy(:cu_apply_low_pass_filter!, :apply_low_pass_filter!)
    apply_low_pass_filter!(wavefront, filter)
    return nothing
end

function cu_apply_low_pass_filter(wavefront::Wavefront, filter::LowPassFilter)
    _legacy(:cu_apply_low_pass_filter, :apply_low_pass_filter)
    return apply_low_pass_filter(wavefront, filter)
end
