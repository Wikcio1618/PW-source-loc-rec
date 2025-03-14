include("StructModule.jl")
using .StructModule

using Graphs
using Statistics
using LinearAlgebra
using Base.Threads


# Each and every one of 3 localization methods returns pairs of (node_idx, score) sorted by score

function pearson_loc(g::SimpleGraph, obs_data::Dict{Int,Int})::Vector{Tuple{Int64,Float64}}
    V = nv(g)
    O = length(obs_data)
    obs_idxs = collect(keys(obs_data))
    obs_times = collect(values(obs_data))

    # 2d array - each row is distances from every observer
    LENS = Array{Int,2}(undef, O, V)
    # for every observer
    for (i, obs_idx) in enumerate(obs_idxs)
        # TODO: use paths_from_node algortithm for speedup?
        LENS[i, :] = dijkstra_shortest_paths(g, obs_idx).dists
    end
    # for every node
    covs = Dict{Int,Float64}(s => Statistics.cor(LENS[:, s], obs_times) for s in 1:V)
    pairs = sort(collect(zip(keys(covs), values(covs))), by=x -> x[2], rev=true)

    return pairs
end


function LPTVA_loc(g::SimpleGraph, obs_data::Dict{Int,Int}, beta)::Vector{Tuple{Int64,Float64}}
    pairs = sort(collect(zip(keys(obs_data), values(obs_data))), by=x -> x[2])
    observers = [p[1] for p in pairs]
    times = [p[2] for p in pairs]

    K = length(observers)
    if K < 2
        throw(ArgumentError("At least two observers are required."))
    end

    t_prim = [times[i+1] - times[1] for i in 1:K-1]

    vertices_list = collect(vertices(g))
    scores_arr = Vector{Tuple{Int,Float64}}(undef, length(vertices_list))

    @threads for i in eachindex(vertices_list)
        v = vertices_list[i]
        score = calculate_phi_score(g, v, t_prim, observers, beta)
        scores_arr[i] = (v, score)
    end

    return sort(scores_arr, by=x -> x[2], rev=true)
end


"""
Following Holyst '18
"""
function GMLA_loc(g::SimpleGraph, obs_data::Dict{Int,Int}, beta, K0=missing)::Vector{Tuple{Int64,Float64}}
    if ismissing(K0)
        K0 = round(Int, sqrt(nv(g)))
        if K0 > length(obs_data)
            K0 = length(obs_data)
        end
    end
    if K0 < 2
        throw(ArgumentError("At least two observers are required."))
    end

    pairs = sort(collect(zip(keys(obs_data), values(obs_data))), by=x -> x[2])
    observers = [p[1] for p in pairs][1:K0]
    times = [p[2] for p in pairs][1:K0]
    t_prim = [times[i+1] - times[1] for i in 1:K0-1]

    curr_node = observers[1]
    max_score = -Inf
    scores = Dict{Int,Float64}()
    while true
        Tv_max = -Inf
        nei_max = 0
        for nei ∈ neighbors(g, curr_node)
            if !haskey(scores, nei)
                score = calculate_phi_score(g, nei, t_prim, observers, beta)
                if score > Tv_max
                    Tv_max = score
                    nei_max = nei
                end
                scores[nei] = score
            end
        end

        if Tv_max < max_score
            break
        else
            max_score = Tv_max
            curr_node = nei_max
        end
    end

    pairs = sort(collect(zip(keys(scores), values(scores))), by=x -> x[2], rev=true)
    return pairs
end

"""
From Machura '22
"""
function calculate_phi_score(graph::SimpleGraph, s::Int, t_prim::Vector{Int}, observers::Vector{Int}, beta)::Float64
    K = length(observers)
    sig_2 = (1 - beta) / beta / beta
    tree = SimpleGraph(bfs_tree(graph, s))

    obs_paths::Dict{Int,Vector{Int}} = paths_to_target(tree, observers[1], observers[2:K])

    L_s = zeros(K - 1, K - 1)
    for i in 1:K-1
        len_i = length(obs_paths[observers[i+1]]) - 1
        for j in i:K-1
            len_j = length(obs_paths[observers[j+1]]) - 1
            union_ij = length(union(obs_paths[observers[i+1]], obs_paths[observers[j+1]])) - 1
            # basic set theory (|A| + |B| - |A ⊎ B|)
            L_s[i, j] = L_s[j, i] = sig_2 * (len_i + len_j - union_ij)
        end
    end
    det_L = det(L_s)
    L_s_inv = cholesky(L_s).L \ I

    mu = 1 / beta
    source_paths::Dict{Int,Vector{Int}} = paths_to_target(tree, s, observers)
    ref_path_length = length(source_paths[observers[1]]) - 1
    mu_s = [mu * (length(source_paths[observers[i]]) - 1 - ref_path_length) for i in 2:K]

    temp_vec = t_prim - mu_s
    phi = -(transpose(temp_vec) * L_s_inv * temp_vec) - log(det_L)
    return phi
end

"""
Given node `target` and TREE graph `g` return dictionary mapping index of each target node in the graph to the vector of indexes defining a path from `nodes` to the `target`
"""
function paths_to_target(tree::SimpleGraph, target::Int, nodes::Vector{Int})::Dict{Int,Vector{Int}}
    paths = Dict(i => Vector{Int}() for i in nodes)

    for i in nodes
        visited = Set{Int}(i)
        path_rec(tree, target, visited, [i], paths)
    end
    return paths
end

function path_rec(tree::SimpleGraph, target::Int, visited::Set{Int}, curr_path::Vector{Int}, paths::Dict{Int,Vector{Int}})
    curr = curr_path[end]
    # found target
    if curr == target
        paths[curr_path[1]] = curr_path
        return
    end
    # visit all the neighbors
    for nei in neighbors(tree, curr)
        if nei ∉ visited
            push!(visited, nei)
            path_rec(tree, target, visited, [curr_path; nei], paths)
        end
    end
end