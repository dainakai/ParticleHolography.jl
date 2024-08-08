using JSON
using MetaGraphsNext
using Graphs
using UUIDs

export Labonte, enum_edge, append_path!, gen_fulldict

function node_distance(node1, node2)
    return sqrt((node1[1] - node2[1])^2 + (node1[2] - node2[2])^2 + (node1[3] - node2[3])^2)
end

function SOM!(weight1, weight2, Dmax, α, R)
    Δw = Dict(keys(weight2) .=> [[0.0, 0.0, 0.0]])
    cand_weight2 = []
    for point in weight1
        min_r = 10000000.0 # set too large value to find the minimum
        cand_key = collect(weight2)[1][1] # indicated as c letter in the paper

        point_neighbor = []
        for candidate in weight2
            dist = node_distance(point[2], candidate[2])
            if dist < min_r
                min_r = dist
                cand_key = candidate[1]
            end

            if dist <= Dmax
                push!(point_neighbor, candidate[1])
            end
        end

        # see Eq.(5) and (6) in "A new neural network for particle-tracking velocimetry" by G. Labonte.
        α_list = Dict()
        for key in point_neighbor
            if node_distance(weight2[cand_key], weight2[key]) <= R
                α_list[key] = α
            else
                α_list[key] = α*exp(-(node_distance(weight2[cand_key], weight2[key])-R)^2/(2*R^2))
            end
        end
        
        for key in point_neighbor
            Δw[key] += α_list[key] .* (point[2] .- weight2[cand_key])
        end

        push!(cand_weight2, cand_key)
    end

    ΔW = Dict(keys(weight1) .=> [[0.0, 0.0, 0.0]])
    cand_weight1 = []
    for point in weight2
        min_r = 10000000.0
        cand_key = collect(weight1)[1][1]

        point_neighbor = []
        for candidate in weight1
            dist = node_distance(point[2], candidate[2])
            if dist < min_r
                min_r = dist
                cand_key = candidate[1]
            end

            if dist <= Dmax
                push!(point_neighbor, candidate[1])
            end
        end

        α_list = Dict()
        for key in point_neighbor
            if node_distance(weight1[cand_key], weight1[key]) <= R
                α_list[key] = α
            else
                α_list[key] = α*exp(-(node_distance(weight1[cand_key], weight1[key])-R)^2/(2*R^2))
            end
        end
        
        for key in point_neighbor
            ΔW[key] += α_list[key] .* (point[2] .- weight1[cand_key])
        end

        push!(cand_weight1, cand_key)
    end

    for key in keys(weight1)
        weight1[key] .+= ΔW[key]
    end

    for key in keys(weight2)
        weight2[key] .+= Δw[key]
    end
    
    return Set{UUID}(cand_weight1), Set{UUID}(cand_weight2)
end

function SOM_iteration!(weight1, weight2, Dmax, α, R, Rend, β, N)
    n = 0
    weight1keys = Set(keys(weight1))
    weight2keys = Set(keys(weight2))
    w1_cont_excld_cnt = Dict(k => 0 for k in weight1keys)
    w2_cont_excld_cnt = Dict(k => 0 for k in weight2keys)

    while true
        n += 1
        cand_weight1, cand_weight2 = SOM!(weight1, weight2, Dmax, α, R)

        complement_weight1 = setdiff(weight1keys, cand_weight1)
        complement_weight2 = setdiff(weight2keys, cand_weight2)

        for key in complement_weight1
            w1_cont_excld_cnt[key] += 1
        end

        for key in complement_weight2
            w2_cont_excld_cnt[key] += 1
        end

        for cand_weight1_key in cand_weight1
            w1_cont_excld_cnt[cand_weight1_key] = 0
        end

        for cand_weight2_key in cand_weight2
            w2_cont_excld_cnt[cand_weight2_key] = 0
        end

        if R <= Rend
            break
        end

        R = R * β
        α = α / β

        keys1todelete = filter(x -> w1_cont_excld_cnt[x] >= N, keys(w1_cont_excld_cnt))
        keys2todelete = filter(x -> w2_cont_excld_cnt[x] >= N, keys(w2_cont_excld_cnt))

        for key in keys1todelete
            delete!(weight1, key)
            delete!(weight1keys, key)
        end

        for key in keys2todelete
            delete!(weight2, key)
            delete!(weight2keys, key)
        end
    end
    return nothing
end

function two_frame_metagraph(weight1, weight2, file1, file2, Rend)
    g = MetaGraph(
        DiGraph(),
        label_type = UUID,
        vertex_data_type = NTuple{2, Float64},
    )

    for (key, value) in file1
        g[key] = Tuple(Float64.(value[1:2]))
    end

    for (key, value) in file2
        g[key] = Tuple(Float64.(value[1:2]))
    end

    for point1 in weight1
        for point2 in weight2
            dist = node_distance(point1[2], point2[2])
            if dist <= Rend
                add_edge!(g, point1[1], point2[1])
            end
        end
    end

    return g
end

"""
    Labonte(dict1, dict2; Dmax=50.0, α=0.005, R=50.0, Rend=0.1, β=0.9, N=10)

Implementation of the improved Labonté algorithm [ohmi, labonte](@cite). Takes dictionaries of detected particle coordinates at two time points as input and returns a graph representing the correspondence between particles at these two time points. The graph contains UUID keys of all particles from both time points as nodes, with directed edges representing particle correspondences from particles in `dict1` to particles in `dict2`. All correspondences can be retrieved using the [`enum_edge`](@ref enum_edge) function.

# Arguments
- `dict1::Dict{UUID, Vector{Float32}}`: Dictionary of detected particle coordinates at the first time point.
- `dict2::Dict{UUID, Vector{Float32}}`: Dictionary of detected particle coordinates at the second time point.

# Optional keyword arguments
- `Dmax::Float32=50.0`: Maximum allowable movement distance.
- `α::Float32=0.005`: Learning rate.
- `R::Float32=50.0`: Initial value of the inspection neighborhood radius.
- `Rend::Float32=0.1`: Final value of the inspection neighborhood radius. The algorithm terminates when the inspection radius becomes smaller than this value through iterations.
- `β::Float32=0.9`: Decay rate of the inspection neighborhood radius.
- `N::Int=10`: Nodes that are not selected as winners for N consecutive times are removed. Refer to the original paper for details.

# Returns
- `MetaGraph`: Graph representing the correspondence between detected particles at two time points.
"""
function Labonte(dict1, dict2; Dmax=50.0, α=0.005, R=50.0, Rend=0.1, β=0.9, N=10)
    weight1 = copy(dict1)
    weight2 = copy(dict2)

    SOM_iteration!(weight1, weight2, Dmax, α, R, Rend, β, N)
    g = two_frame_metagraph(weight1, weight2, dict1, dict2, Rend)
    return g
end

"""
    enum_edge(g::MetaGraph)

Enumerates all edges in the given `MetaGraph` `g` and returns them as an array of arrays of UUIDs. Each inner array contains the labels of the source and destination nodes of an edge.

# Arguments
- `g::MetaGraph`: The `MetaGraph` to enumerate edges from.

# Returns
- `Vector{UUID}[]`: An array of arrays of UUIDs, where each inner array contains the labels of the source and destination nodes of an edge.
"""
function enum_edge(g::MetaGraph)
    paths = Vector{UUID}[]
    for edge in edges(g)
        push!(paths, [label_for(g,edge.src), label_for(g,edge.dst)])
    end
    return paths
end

"""
    append_path!(paths, g::MetaGraph)

Adds correspondences from the MetaGraph `g` to the trajectory array `paths` initialized by the [`enum_edge`](@ref enum_edge) function. Specifically, if the last element of a `Vector{UUID}` array `path` in `paths` matches the starting point of an edge in `g`, the endpoint of that edge is appended to `paths`. If the graph is not injective, `path` is duplicated.

# Arguments
- `paths::Vector{UUID}[]`: Array of trajectories.
- `g::MetaGraph`: The `MetaGraph` from which correspondences are to be added.

# Returns
- `nothing`
"""
function append_path!(paths, g::MetaGraph)
    for label in labels(g)
        if isempty(filter(path -> last(path) == label, paths))
            push!(paths, [label])
        end
        for (idx, onl) in enumerate(outneighbor_labels(g, label))
            if idx == 1
                for path in filter(path -> last(path) == label, paths)
                    push!(path, onl)
                end
            else
                for path in filter(path -> last(path) == label, paths)
                    newpath = copy(path)
                    push!(newpath, onl)
                    push!(paths, newpath)
                end
            end
        end
    end
end

"""
    gen_fulldict(filepaths::Vector{String})

Generates a dictionary containing the UUIDs of particles for all conceivable frames as keys and their coordinates as values. The input is a vector of file paths containing JSON files with particle coordinates.

# Arguments
- `filepaths::Vector{String}`: Vector of file paths containing JSON files with particle coordinates.

# Returns
- `Dict{UUID, Vector{Float32}}`: Dictionary containing the UUIDs of particles for all conceivable frames as keys and their coordinates as values.
"""
function gen_fulldict(filepaths::Vector{String})
    fulldict = Dict()
    for (idx, file) in enumerate(filepaths)
        subdict = dictload(file)
        newdict = Dict()
        for key in keys(subdict)
            newdict[key] = [idx, subdict[key]...]
        end
        merge!(fulldict, newdict)
    end
    return fulldict
end

"""
    gen_fulldict(dicts::Vector{Dict{UUID, Vector{Float32}}})

Generates a dictionary containing the UUIDs of particles for all conceivable frames as keys and their coordinates as values. The input is a vector of dictionaries with UUID keys and values as Vector{Float32}, which includes the coordinates (and diameters) of the particles.

# Arguments
- `dicts::Vector{Dict{UUID, Vector{Float32}}}`: Vector of dictionaries with UUID keys and values as Vector{Float32}.

# Returns
- `Dict{UUID, Vector{Float32}}`: Dictionary containing the UUIDs of particles for all conceivable frames as keys and their coordinates as values.
"""
function gen_fulldict(dicts::Vector{Dict{UUID, Vector{Float32}}})
    fulldict = Dict()
    for (idx, subdict) in enumerate(dicts)
        newdict = Dict()
        for key in keys(subdict)
            newdict[key] = [idx, subdict[key]...]
        end
        merge!(fulldict, newdict)
    end
    return fulldict
end