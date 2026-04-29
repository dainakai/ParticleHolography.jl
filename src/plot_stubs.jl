export particleplot, particleplot!, trajectoryplot, trajectoryplot!

"""
    particleplot(data; kwargs...)

Create a particle scatter plot from a particle dictionary. Load `Plots` before
calling this API.
"""
function particleplot end

"""
    particleplot!(data; kwargs...)

Add a particle scatter plot to the current Plots.jl plot. Load `Plots` before
calling this API.
"""
function particleplot! end

"""
    trajectoryplot(paths, fulldict; kwargs...)

Create a trajectory plot from particle paths and a full particle dictionary.
Load `Plots` before calling this API.
"""
function trajectoryplot end

"""
    trajectoryplot!(paths, fulldict; kwargs...)

Add a trajectory plot to the current Plots.jl plot. Load `Plots` before calling
this API.
"""
function trajectoryplot! end
