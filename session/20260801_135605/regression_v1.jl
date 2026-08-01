using CUDA
using ParticleHolography
using Printf
using Serialization
using Statistics

CUDA.functional() || error("CUDA is required for the v0.2.4/v1 regression")
CUDA.allowscalar(false)
old = deserialize("/tmp/particleholo_v024_regression.bin")
n = size(old.hologram1, 1)
wavelength = 0.6328
pixel_pitch = 10.0

function calculate(selected)
    sqrt_part = transfer_sqrt(selected, n, wavelength, pixel_pitch)
    forward = transfer(selected, 120.0, wavelength, sqrt_part)
    backward = transfer(selected, -120.0, wavelength, sqrt_part)
    front = transfer(selected, -800.0, wavelength, sqrt_part)
    step = transfer(selected, -20.0, wavelength, sqrt_part)
    wavefront = phase_retrieval(
        PhaseRetrievalPlan(selected, forward, backward),
        old.hologram1, old.hologram2; iterations=3,
    )
    plan = ReconstructionPlan(selected, front, step)
    volume = reconstruct(plan, wavefront, 5; output=Float32, clamp_output=true)
    projection = xyprojection(plan, wavefront, 5)
    complex_volume = reconstruct_complex(plan, wavefront, 5)
    synchronize_backend(selected)
    return (
        sqrt_part=to_host(sqrt_part).data,
        forward=to_host(forward).data,
        wavefront=to_host(wavefront).data,
        volume=to_host(volume),
        projection=to_host(projection),
        complex_volume=to_host(complex_volume),
    )
end

function report(label, candidate, reference)
    difference = abs.(candidate .- reference)
    max_abs = maximum(difference)
    rmse = sqrt(mean(abs2, candidate .- reference))
    scale = max(maximum(abs, reference), eps(Float32))
    @printf("%-31s max_abs=%10.4e  rmse=%10.4e  max_rel=%10.4e\n",
            label, max_abs, rmse, max_abs / scale)
end

cpu = calculate(backend(:cpu))
cuda = calculate(backend(:cuda))
for (name, reference) in pairs(old)
    name in (:hologram1, :hologram2) && continue
    cpu_value = getproperty(cpu, name)
    cuda_value = getproperty(cuda, name)
    if name in (:sqrt_part, :forward)
        cpu_value = circshift(cpu_value, (n ÷ 2, n ÷ 2))
        cuda_value = circshift(cuda_value, (n ÷ 2, n ÷ 2))
    end
    report("v0.2.4 vs v1 CPU / $name", cpu_value, reference)
    report("v0.2.4 vs v1 CUDA / $name", cuda_value, reference)
    report("v1 CPU vs CUDA / $name", getproperty(cpu, name), getproperty(cuda, name))
end
