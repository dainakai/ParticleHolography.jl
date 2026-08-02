using Random
using LinearAlgebra

@testset "calibration" begin
    identity_coefficients = Float64[0, 1, 0, 0, 0, 0,
                                    0, 0, 1, 0, 0, 0]
    image = reshape(Float32.(1:64), 8, 8)
    @test quadratic_distortion_correction(image, identity_coefficients) == image
    @test_throws ArgumentError quadratic_distortion_correction(image, ones(11))

    random_image = rand(MersenneTwister(8), Float32, 16, 16)
    vectors = piv_map(CPUBackend(), random_image, random_image;
                      grid_size=8, interrogation_size=8, search_size=16)
    backend(:cpu)
    default_vectors = piv_map(random_image, random_image;
                              grid_size=8, interrogation_size=8,
                              search_size=16)
    @test size(vectors) == (1, 1, 2)
    @test vectors ≈ zeros(Float32, 1, 1, 2) atol=0.15
    @test default_vectors == vectors
    @test_throws ArgumentError piv_map(CPUBackend(), random_image, random_image;
                                       grid_size=8, interrogation_size=9,
                                       search_size=16)

    # Exercise the calibration system independently of image registration and
    # verify that the package's compact factorization solves the normal
    # equations, rather than only checking return shapes.
    jacobian = ParticleHolography.getYacobian(32, 8)
    @test size(jacobian) == (18, 12)
    normal_matrix = transpose(jacobian) * jacobian
    decomposition = ParticleHolography.modified_Cholesky_decomposition(normal_matrix)
    residual = randn(MersenneTwister(11), size(jacobian, 1))
    correction = ParticleHolography.simultanious_equation_solver(
        decomposition, jacobian, residual,
    )
    @test normal_matrix * correction ≈ -transpose(jacobian) * residual atol=1e-8
    @test_throws DimensionMismatch ParticleHolography.modified_Cholesky_decomposition(
        zeros(2, 3),
    )
    @test_throws LinearAlgebra.PosDefException ParticleHolography.modified_Cholesky_decomposition(
        zeros(2, 2),
    )

    identity_coefficients = Float64[0, 1, 0, 0, 0, 0,
                                    0, 0, 1, 0, 0, 0]
    zero_vectors = zeros(Float32, 3, 3, 2)
    @test ParticleHolography.getErrorVec(
        zero_vectors, identity_coefficients, 8, 32,
    ) == zeros(18)
    @test_throws DimensionMismatch ParticleHolography.getErrorVec(
        zeros(Float32, 2, 2, 2), identity_coefficients, 8, 32,
    )

    calibration_image = rand(MersenneTwister(12), Float32, 32, 32)
    coefficients = get_distortion_coefficients(
        calibration_image, calibration_image;
        grid_size=8, interrogation_size=8, search_size=8,
    )
    @test length(coefficients) == 12
    @test all(isfinite, coefficients)
end
