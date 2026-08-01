using Random
using UUIDs

@testset "detection and IO" begin
    binary = falses(7, 7)
    binary[2, 2] = true
    binary[3, 3] = true
    binary[6, 6] = true
    labels = connected_component_labeling(binary)
    @test count_labels(labels) == 2
    @test find_valid_labels(labels) == [1, 2]
    @test sort(get_bounding_rectangles(labels)) == [(2, 2, 3, 3), (6, 6, 6, 6)]

    volume = falses(8, 8, 3)
    volume[3:6, 3:6, :] .= true
    dilated = dilate(volume)
    @test sum(dilated) == 36 * 3
    @test !any(dilated[[1, end], :, :])

    boxes1 = particle_bounding_boxes_3d(volume; rng=MersenneTwister(12))
    boxes2 = particle_bounding_boxes_3d(volume; rng=MersenneTwister(12))
    @test boxes1 == boxes2
    @test length(boxes1) == 1
    @test first(values(boxes1)) == [3, 3, 1, 6, 6, 3]

    intensity = ones(Float32, size(volume))
    intensity[3:6, 3:6, :] .= 0.25f0
    coordinates = particle_coordinates(boxes1, intensity;
                                       profile_smoothing_kernel=nothing)
    id = first(keys(boxes1))
    @test coordinates[id][1:2] ≈ Float32[4.5, 4.5]
    with_diameter = particle_coor_diams(boxes1, intensity;
                                       profile_smoothing_kernel=nothing)
    @test length(with_diameter[id]) == 4
    @test all(isfinite, with_diameter[id])
    @test_throws ArgumentError particle_coordinates(boxes1, volume)
    @test_throws ArgumentError particle_coordinates(boxes1,
                                                     complex.(intensity))

    images = [UInt8[10 20; 30 40], UInt8[10 25; 30 45],
              UInt8[10 20; 30 40]]
    background = make_background_mode(images)
    @test background ≈ [10 20; 30 40] ./ 255

    contours = find_external_contours(Float32[0 0 0; 0 1 0; 0 0 0])
    @test length(contours) == 1
    canvas = zeros(Int, 3, 3)
    draw_contours!(canvas, 1, contours)
    @test canvas[2, 2] == 1

    padded = pad_with_mean(Float32[1 2 3; 4 5 6], 6)
    @test size(padded) == (6, 6)
    @test padded[3:4, 2:4] == Float32[1 2 3; 4 5 6]
    @test padded[1, 1] == 3.5f0
    @test_throws ArgumentError pad_with_mean(ones(2, 3), 3)

    mktempdir() do directory
        path = joinpath(directory, "particles.json")
        dictsave(path, coordinates)
        @test dictload(path) == coordinates
    end
end
