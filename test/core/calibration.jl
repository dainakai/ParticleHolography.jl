using Random

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
end
