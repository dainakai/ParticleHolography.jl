using Graphs
using MetaGraphsNext
using UUIDs

export labonte, Labonte, enum_edge, append_path!, gen_fulldict, node_distance

"""Euclidean particle distance with a configurable optical-axis weight."""
function node_distance(node1, node2, dim3weight::Real=0.01)
    length(node1) >= 3 && length(node2) >= 3 || throw(ArgumentError("Nodes must contain at least three coordinates."))
    dim3weight >= 0 || throw(ArgumentError("dim3weight must be non-negative."))
    return hypot(node1[1] - node2[1], node1[2] - node2[2],
                 dim3weight * (node1[3] - node2[3]))
end

function _som_delta(source, target, max_distance, learning_rate, radius, dim3weight)
    delta = Dict(id => zeros(Float64, 3) for id in keys(target))
    winners = Set{UUID}()
    isempty(target) && return delta, winners

    for (_, source_point) in source
        winner = first(keys(target))
        minimum_distance = Inf
        neighbours = UUID[]
        for (candidate_id, candidate_point) in target
            distance = node_distance(source_point, candidate_point, dim3weight)
            if distance < minimum_distance
                minimum_distance = distance
                winner = candidate_id
            end
            distance <= max_distance && push!(neighbours, candidate_id)
        end

        for id in neighbours
            winner_distance = node_distance(target[winner], target[id], dim3weight)
            local_rate = winner_distance <= radius ? learning_rate :
                         learning_rate * exp(-(winner_distance - radius)^2 / (2radius^2))
            delta[id] .+= local_rate .* (source_point .- target[winner])
        end
        push!(winners, winner)
    end
    return delta, winners
end

function SOM!(weight1, weight2, max_distance, learning_rate, radius, dim3weight)
    delta2, winners2 = _som_delta(weight1, weight2, max_distance, learning_rate,
                                  radius, dim3weight)
    delta1, winners1 = _som_delta(weight2, weight1, max_distance, learning_rate,
                                  radius, dim3weight)
    for id in keys(weight1)
        weight1[id] .+= delta1[id]
    end
    for id in keys(weight2)
        weight2[id] .+= delta2[id]
    end
    return winners1, winners2
end

function SOM_iteration!(weight1, weight2, max_distance, learning_rate,
                        radius, final_radius, decay, exclusion_limit,
                        dim3weight=0.01)
    keys1 = Set(keys(weight1))
    keys2 = Set(keys(weight2))
    excluded1 = Dict(id => 0 for id in keys1)
    excluded2 = Dict(id => 0 for id in keys2)

    while !isempty(weight1) && !isempty(weight2)
        winners1, winners2 = SOM!(weight1, weight2, max_distance,
                                  learning_rate, radius, dim3weight)
        for id in keys1
            excluded1[id] = id in winners1 ? 0 : excluded1[id] + 1
        end
        for id in keys2
            excluded2[id] = id in winners2 ? 0 : excluded2[id] + 1
        end
        radius <= final_radius && break
        radius *= decay
        learning_rate /= decay

        remove1 = [id for id in keys1 if excluded1[id] >= exclusion_limit]
        remove2 = [id for id in keys2 if excluded2[id] >= exclusion_limit]
        for id in remove1
            delete!(weight1, id)
            delete!(keys1, id)
            delete!(excluded1, id)
        end
        for id in remove2
            delete!(weight2, id)
            delete!(keys2, id)
            delete!(excluded2, id)
        end
    end
    return nothing
end

function two_frame_metagraph(weight1, weight2, frame1, frame2, final_radius,
                             dim3weight)
    graph = MetaGraph(DiGraph(), label_type=UUID,
                      vertex_data_type=NTuple{3,Float64})
    for (id, value) in frame1
        graph[id] = Tuple(Float64.(value[1:3]))
    end
    for (id, value) in frame2
        graph[id] = Tuple(Float64.(value[1:3]))
    end
    for (id1, point1) in weight1, (id2, point2) in weight2
        node_distance(point1, point2, dim3weight) <= final_radius &&
            add_edge!(graph, id1, id2)
    end
    return graph
end

"""
    labonte(frame1, frame2; kwargs...)

Apply the improved Labonté particle-correspondence algorithm to two frames.
Input dictionaries are never mutated.
"""
function labonte(frame1::AbstractDict{UUID,<:AbstractVector},
                 frame2::AbstractDict{UUID,<:AbstractVector};
                 max_distance::Real=50.0, learning_rate::Real=0.005,
                 radius::Real=50.0, final_radius::Real=0.1,
                 decay::Real=0.9, exclusion_limit::Integer=10,
                 dim3weight::Real=0.01)
    max_distance > 0 || throw(ArgumentError("max_distance must be positive."))
    learning_rate > 0 || throw(ArgumentError("learning_rate must be positive."))
    radius >= final_radius > 0 || throw(ArgumentError("Require radius ≥ final_radius > 0."))
    0 < decay < 1 || throw(ArgumentError("decay must be between zero and one."))
    exclusion_limit > 0 || throw(ArgumentError("exclusion_limit must be positive."))
    isempty(intersect(keys(frame1), keys(frame2))) || throw(ArgumentError("Particle UUIDs must be unique across frames."))

    weight1 = Dict(id => Float64.(value[1:3]) for (id, value) in frame1)
    weight2 = Dict(id => Float64.(value[1:3]) for (id, value) in frame2)
    if !isempty(weight1) && !isempty(weight2)
        SOM_iteration!(weight1, weight2, max_distance, learning_rate, radius,
                       final_radius, decay, exclusion_limit, dim3weight)
    end
    return two_frame_metagraph(weight1, weight2, frame1, frame2,
                               final_radius, dim3weight)
end

"""v0.2 keyword-compatible spelling of [`labonte`](@ref)."""
function Labonte(frame1, frame2; Dmax=50.0, α=0.005, R=50.0, Rend=0.1,
                 β=0.9, N=10, dim3weight=0.01)
    return labonte(frame1, frame2; max_distance=Dmax, learning_rate=α,
                   radius=R, final_radius=Rend, decay=β,
                   exclusion_limit=N, dim3weight)
end

function enum_edge(graph::MetaGraph, labeltype=UUID)
    paths = Vector{labeltype}[]
    for edge in edges(graph)
        push!(paths, labeltype[label_for(graph, edge.src),
                               label_for(graph, edge.dst)])
    end
    return paths
end

"""Append one correspondence graph to an existing set of paths, preserving branches."""
function append_path!(paths::Vector{<:AbstractVector}, graph::MetaGraph)
    for label in labels(graph)
        outgoing = collect(outneighbor_labels(graph, label))
        matching = findall(path -> !isempty(path) && last(path) == label, paths)
        if isempty(matching) && (isempty(inneighbor_labels(graph, label)) || !isempty(outgoing))
            push!(paths, [label])
            matching = [lastindex(paths)]
        end
        isempty(outgoing) && continue

        originals = [copy(paths[index]) for index in matching]
        for (index, original) in zip(matching, originals)
            paths[index] = [original; first(outgoing)]
            for destination in Iterators.drop(outgoing, 1)
                push!(paths, [original; destination])
            end
        end
    end
    return paths
end

function _merge_frame!(full::Dict{UUID,Vector{Float32}}, frame, frame_index::Int)
    for (id, values) in frame
        haskey(full, id) && throw(ArgumentError("Duplicate particle UUID $id across frames."))
        full[id] = Float32[frame_index; values...]
    end
    return full
end

function gen_fulldict(filepaths::AbstractVector{<:AbstractString})
    full = Dict{UUID,Vector{Float32}}()
    for (index, path) in enumerate(filepaths)
        _merge_frame!(full, dictload(path), index)
    end
    return full
end

function gen_fulldict(frames::AbstractVector{<:AbstractDict})
    full = Dict{UUID,Vector{Float32}}()
    for (index, frame) in enumerate(frames)
        _merge_frame!(full, frame, index)
    end
    return full
end
