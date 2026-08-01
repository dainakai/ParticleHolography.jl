using ParticleHolography
using Documenter
using DocumenterCitations

include("generate_assets.jl")

function build_documentation()
    generate_workflow_figure()
    DocMeta.setdocmeta!(ParticleHolography, :DocTestSetup,
                        :(using ParticleHolography); recursive=true)
    bibliography = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

    return makedocs(;
        modules=[ParticleHolography],
        authors="Dai Nakai <dainakai1031@gmail.com> and contributors",
        sitename="ParticleHolography.jl",
        doctest=true,
        checkdocs=:exports,
        warnonly=[:missing_docs],
        format=Documenter.HTML(;
            canonical="https://dainakai.github.io/ParticleHolography.jl",
            edit_link="main",
            assets=["assets/styles.css", "assets/favicon.ico"],
        ),
        pages=[
            "Home" => "index.md",
            "Start here" => [
                "10-minute CPU quickstart" => "getting-started/quickstart.md",
                "Choose CPU, Metal, or CUDA" => "getting-started/backends.md",
                "Parameters and units" => "getting-started/parameters.md",
            ],
            "Understand holography" => [
                "From hologram to particles" => "concepts/workflow.md",
                "Inline holography theory" => "whats_inline_holography.md",
            ],
            "Guides" => [
                "Gabor and phase retrieval" => "guides/reconstruction.md",
                "Detection and tracking" => "guides/detection-tracking.md",
                "Preprocessing and files" => "guides/preprocessing.md",
                "phdemo real-data tutorial" => "guides/phdemo.md",
                "Performance and memory" => "guides/performance.md",
            ],
            "Help" => [
                "Troubleshooting" => "troubleshooting.md",
                "Migrate from v0.2" => "migration-v1.md",
            ],
            "API reference" => "reference.md",
        ],
        plugins=[bibliography],
    )
end
