using ParticleHolography
using Documenter
using DocumenterCitations

DocMeta.setdocmeta!(ParticleHolography, :DocTestSetup, :(using ParticleHolography); recursive=true)
bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(;
    modules=[ParticleHolography],
    authors="Dai Nakai <dainakai1031@gmail.com> and contributors",
    sitename="ParticleHolography.jl",
    format=Documenter.HTML(;
        canonical="https://dainakai.github.io/ParticleHolography.jl",
        edit_link="main",
        assets=["assets/styles.css", "assets/favicon.ico"],
    ),
    pages=[
        "Home" => "index.md",
        "What's inline holography?" => "whats_inline_holography.md",
        "Tutorials" => Any[
            "Gabor holography" => "tutorials/gabor.md",
            "Phase retrieval holography" => "tutorials/pr.md",
            "Particle handling" => "tutorials/particle.md",
        ],
        "Usage" => Any[
            "Preprocessings" => "usage/preprocessings.md",
            "Volume reconstruction" => "usage/reconstruction.md",
            # "Particle detection" => "usage/particle_detection.md",
            # "Particle tracking" => "usage/particle_tracking.md",
            # "Types" => "usage/types.md",
        ],
        "Reference" => "reference.md",
    ],
    plugins=[bib],
)

deploydocs(;
    repo="github.com/dainakai/ParticleHolography.jl",
    devbranch="main",
)
