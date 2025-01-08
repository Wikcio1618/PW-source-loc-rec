module Localization
export pearson_loc, LPTVA_loc, GMLA_loc

using ..ObserverGraph
using ..Propagation

using Graphs
using Statistics
using LinearAlgebra

function pearson_loc(og::ObsGraph, obs_data::Dict{Int,Int})::Int
    N = nv(og.graph)
    O = length(obs_data)
    obs_idxs = collect(keys(obs_data))
    obs_times = collect(values(obs_data))

    # 2d array - each row is distances from every observer
    LENS = Array{Int,2}(undef, O, N)

    # for every observer
    for (i, obs_idx) in enumerate(obs_idxs)
        LENS[i, :] = dijkstra_shortest_paths(og.graph, obs_idx).dists
    end

    # for every node
    covs::Vector{Float64} = [Statistics.cor(LENS[:, i], obs_times) for i in 1:N]
    return argmax(covs)
end



function LPTVA_loc(og::ObsGraph, obs_data::Dict{Int,Int}, beta)
    observers = collect(keys(obs_data))
    times = collect(values(obs_data))
    K = length(observers)  # Number of observers
    if K < 2
        throw(ArgumentError("At least two observers are required."))
    end

    t_prim = [times[i+1] - times[1] for i in 1:K-1]

    # Initialize the best score and corresponding source
    best_score = -Inf
    best_source = -1

    # Iterate over each node in the graph as a candidate source
    for s in vertices(og.graph)
        F_s = calculate_phi_score(og.graph, s, t_prim, observers, beta)
        if F_s > best_score
            best_score = F_s
            best_source = s
        end
    end

    return best_source
end


"""
Following Holyst '18
"""
function GMLA_loc(og::ObsGraph, obs_data::Dict{Int,Int}, beta, K0=missing)::Int
    if ismissing(K0)
        K0 = round(Int, sqrt(nv(og.graph)))
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

    best_source = observers[1]
    best_score = calculate_phi_score(og.graph, observers[1], t_prim, observers, beta)
    S = Set(observers[1])

    while true
        println("Considering $best_source")
        T_v = Dict(nei => calculate_phi_score(og.graph, nei, t_prim, observers, beta) for nei ∈ neighbors(og.graph, best_source) if nei ∉ S)

        if isempty(T_v)
            break
        end
        union!(S, collect(keys(T_v)))

        curr_best_source = argmax(T_v)
        curr_best_score = T_v[curr_best_source]
        if curr_best_score < best_score
            break
        else
            best_score = curr_best_score
            best_source = curr_best_source
        end
    end

    return best_source
end

"""
From Machura '22
"""
function calculate_phi_score(g::SimpleGraph, s::Int, t_prim::Vector{Int}, observers::Vector{Int}, beta)::Float64
    tree = SimpleGraph(Graphs.bfs_tree(g, s))
    K = length(observers)
    mu = 1 / beta
    sig_2 = (1 - beta) / beta / beta

    ref_path_length = length(a_star(tree, s, observers[1]))
    mu_s = [mu * (length(a_star(tree, s, observers[i+1])) - ref_path_length) for i in 1:K-1]

    L_s = zeros(K - 1, K - 1)
    for i in 1:K-1
        for j in 1:K-1
            edges_i = Set(a_star(tree, observers[i+1], observers[1]))
            edges_j = Set(a_star(tree, observers[j+1], observers[1]))
            L_s[i, j] = sig_2 * length(intersect(edges_i, edges_j))
        end
    end

    det_L = det(L_s)
    if det_L == 0
        L_s += 1e-6 * I
        det_L = det(L_s)
    end
    L_s_inv = inv(L_s)  # Invert the covariance matrix

    temp_vec = t_prim - mu_s
    phi = -(transpose(temp_vec) * L_s_inv * temp_vec) - log(det_L)
    return phi
end

end