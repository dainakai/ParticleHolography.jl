using ParticleHolography
using Documenter

DocMeta.setdocmeta!(ParticleHolography, :DocTestSetup, :(using ParticleHolography); recursive=true)

makedocs(;
    modules=[ParticleHolography],
    authors="Dai Nakai <dainakai1031@gmail.com> and contributors",
    sitename="ParticleHolography.jl",
    format=Documenter.HTML(;
        canonical="https://dainakai.github.io/ParticleHolography.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dainakai/ParticleHolography.jl",
    devbranch="main",
)
