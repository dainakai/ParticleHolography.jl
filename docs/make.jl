include("build.jl")
build_documentation()

# Publication is intentionally confined to the CI entrypoint. Running
# docs/build_local.jl never calls deploydocs.
deploydocs(;
    repo="github.com/dainakai/ParticleHolography.jl",
    devbranch="main",
)
