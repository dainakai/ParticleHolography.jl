using Graphs
using MetaGraphsNext
using UUIDs

@testset "tracking" begin
    first_id, second_id, branch_id = uuid4(), uuid4(), uuid4()
    frame1 = Dict(first_id => Float32[1, 2, 3])
    frame2 = Dict(second_id => Float32[1.01, 2.01, 3])
    graph = labonte(frame1, frame2; max_distance=2, radius=2,
                    final_radius=0.2, decay=0.5, exclusion_limit=4)
    @test nv(graph) == 2
    @test !isempty(enum_edge(graph))
    @test frame1[first_id] == Float32[1, 2, 3]

    empty_graph = labonte(Dict{UUID,Vector{Float32}}(), frame2)
    @test nv(empty_graph) == 1
    @test ne(empty_graph) == 0
    @test_throws ArgumentError labonte(frame1, Dict(first_id => Float32[2, 3, 4]))

    branching = MetaGraph(DiGraph(), label_type=UUID,
                          vertex_data_type=NTuple{3,Float64})
    branching[first_id] = (0.0, 0.0, 0.0)
    branching[second_id] = (1.0, 0.0, 0.0)
    branching[branch_id] = (1.0, 1.0, 0.0)
    add_edge!(branching, first_id, second_id)
    add_edge!(branching, first_id, branch_id)
    paths = Vector{UUID}[[first_id]]
    append_path!(paths, branching)
    @test Set(last.(paths)) == Set((second_id, branch_id))

    full = gen_fulldict([frame1, frame2])
    @test full[first_id] == Float32[1, 1, 2, 3]
    @test full[second_id] == Float32[2, 1.01, 2.01, 3]
end
