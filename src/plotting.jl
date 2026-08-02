export particleplot, particleplot!, trajectoryplot, trajectoryplot!

function _plots_extension()
    extension = Base.get_extension(@__MODULE__, :ParticleHolographyPlotsExt)
    isnothing(extension) && throw(ArgumentError("Plotting requires Plots.jl. Install it and run `using Plots` before calling a plotting function."))
    return extension
end

particleplot(args...; kwargs...) = _plots_extension().particleplot(args...; kwargs...)
particleplot!(args...; kwargs...) = _plots_extension().particleplot!(args...; kwargs...)
trajectoryplot(args...; kwargs...) = _plots_extension().trajectoryplot(args...; kwargs...)
trajectoryplot!(args...; kwargs...) = _plots_extension().trajectoryplot!(args...; kwargs...)
