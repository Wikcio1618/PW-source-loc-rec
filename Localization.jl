using Graphs
using Statistics
using LinearAlgebra
using Base.Threads
using DataStructures

# Each and every one of 3 localization methods returns pairs of (node_idx, score) sorted by score

function pearson_loc(g::SimpleGraph, obs_data::Dict{Int,Int})::Vector{Tuple{Int64,Float64}}
    V = nv(g)
    obs_idxs = collect(keys(obs_data))
    obs_times = collect(values(obs_data))
    O = length(obs_idxs)

    # Compute distances from each observer once
    D = Matrix{Float64}(undef, O, V)
    for (i, obs) in enumerate(obs_idxs)
        dists = get_path_lengths(g, obs, vertices(g))  # returns Dict{Int,Int} or similar
        for v in 1:V
            D[i, v] = get(dists, v, Inf)  # assign Inf if unreachable
        end
    end

    # Correlation between each column of D and obs_times
    pairs = Vector{Tuple{Int64,Float64}}()
    for s in 1:V
        col = D[:, s]
        if any(isinf.(col))
            push!(pairs, (s, -Inf))
        else
            push!(pairs, (s, Statistics.cor(obs_times, col)))
        end
    end

    sort!(pairs, by=x -> x[2], rev=true)
    return pairs
end

"""
**Needed by Brute Pearson reconstruction algorithm**\n
Calculate pearson score for a single suspect `node` given graph `g` and observer data `obs_data`.\n
This func is not used in `pearson_loc` bcs it is more optimal to calculate paths by traversing the graph O times rather than V times
"""
function single_node_pearson_score(g::SimpleGraph, node::Int, obs_data::Dict{Int,Int})::Float64
    path_lengths = get_path_lengths(g, node, Set(keys(obs_data)))
    dists = collect(values(path_lengths))
    times = [obs_data[obs] for obs in collect(keys(path_lengths))]
    score = Statistics.cor(dists, times)
end

function LPTVA_loc(g::SimpleGraph, obs_data::Dict{Int,Int}, beta)::Vector{Tuple{Int64,Float64}}
    pairs = sort(collect(zip(keys(obs_data), values(obs_data))), by=x -> x[2])
    observers = [p[1] for p in pairs]
    times = [p[2] for p in pairs]

    K = length(observers)
    if K < 3
        throw(ArgumentError("At least two observers are required."))
    end

    t_prim = [times[i+1] - times[1] for i in 1:K-1]

    vertices_list = collect(vertices(g))
    scores_arr = Vector{Tuple{Int,Float64}}(undef, length(vertices_list))

    @threads for i in 1:nv(g)
        score = calculate_phi_score(g, i, t_prim, observers, beta)
        scores_arr[i] = (i, score)
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
    if K0 < 3
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
    L_s_inv = inv(L_s)

    mu = 1 / beta
    source_paths::Dict{Int,Vector{Int}} = paths_to_target(tree, s, observers)
    ref_path_length = length(source_paths[observers[1]]) - 1
    mu_s = [mu * (length(source_paths[observers[i]]) - 1 - ref_path_length) for i in 2:K]

    temp_vec = t_prim .- mu_s
    phi = -(transpose(temp_vec) * L_s_inv * temp_vec) - logdet(L_s)
    return phi
end

"""
Given node `src` and graph `g` returns a dictionary mapping index of each node from `targets` to the length of the path connecting them
"""
function get_path_lengths(g::SimpleGraph, src::Int, targets::Set{Int})::Dict{Int,Int}
    path_lengths = Dict{Int,Int}()

    queue = [src]
    visited = Set([src])
    curr_length = 0
    while length(path_lengths) < length(targets) && length(visited) - length(queue) < nv(g)
        next_queue = Int[]
        for curr_node in queue
            if curr_node ∈ targets
                path_lengths[curr_node] = curr_length
            end
            for nei in neighbors(g, curr_node)
                if nei ∉ visited
                    push!(visited, nei)
                    push!(next_queue, nei)
                end
            end
        end
        queue = next_queue
        curr_length += 1
    end

    return path_lengths
end

"""
Given node `target` and TREE graph `g` return dictionary mapping index of each target node in the graph to the vector of indexes defining a path from `nodes` to the `target`
"""
function paths_to_target(tree::SimpleGraph, target::Int, nodes::Vector{Int})::Dict{Int,Vector{Int}}
    paths = Dict{Int,Vector{Int}}()
    for src in nodes
        paths[src] = bfs_path(tree, src, target)
    end
    return paths
end

function bfs_path(tree::SimpleGraph, src::Int, target::Int)::Vector{Int}
    if src == target
        return [target]
    end

    queue = [(src, [src])]
    visited = Set{Int}([src])

    while !isempty(queue)
        curr, path = popfirst!(queue)
        for nei in neighbors(tree, curr)
            if nei ∉ visited
                new_path = [path; nei]
                if nei == target
                    return new_path
                end
                push!(queue, (nei, new_path))
                push!(visited, nei)
            end
        end
    end
    return []
end

const loc_type_dict = Dict(
    :pearson => pearson_loc,
    :lptva => LPTVA_loc,
    :gmla => GMLA_loc
)