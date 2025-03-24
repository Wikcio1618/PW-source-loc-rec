using Graphs
using DataStructures
using LinearAlgebra
using SparseArrays

"""
Reconstructs the graph based on similarity score specified by `score_type`. 
`add_thresh` top scores are used to add unexisting links and, similarily, `hide_thresh` lowest scores are used to hide exisiting links
"""
function reconstruct_thresh!(g::SimpleGraph, hide_thresh::Int, add_thresh::Int, score_type=:cn)
    @assert (hide_thresh + add_thresh <= ne(g))

    min_heap = PriorityQueue(scores)
    max_heap = PriorityQueue(Base.Order.Reverse, scores)
    @assert (hide_thresh <= length(min_heap) && add_thresh <= length(max_heap))

    for _ in 1:hide_thresh
        (u, v) = dequeue!(min_heap)
        rem_edge!(g, u, v)
    end
    for _ in 1:add_thresh
        (u, v) = dequeue!(max_heap)
        add_edge!(g, u, v)
    end
end

function get_RA_scores(g::SimpleGraph)::Dict{Tuple{Int,Int},Float64}
    scores = Dict{Tuple{Int,Int},Float64}()

    for u in vertices(g)
        neis = neighbors(g, u)
        for i in 1:length(neis)-1
            for j in i+1:length(neis)
                x, y = neis[i], neis[j]
                pair = (min(x, y), max(x, y))
                if !haskey(scores, pair)
                    ra_score = sum(x -> length(x) == 0 ? 0 : 1 / degree(g, x), intersect(neighbors(g, x), neighbors(g, y)))
                    scores[pair] = ra_score
                end
            end
        end
    end
    return scores
end

function get_CN_scores(g::SimpleGraph)::Dict{Tuple{Int,Int},Float64}
    scores = Dict{Tuple{Int,Int},Float64}()

    for u in vertices(g)
        neis = neighbors(g, u)
        for i in 1:length(neis)-1
            for j in i+1:length(neis)
                x, y = neis[i], neis[j]
                pair = (min(x, y), max(x, y))
                if !haskey(scores, pair)
                    cn_score = Float64(length(intersect(neighbors(g, x), neighbors(g, y))))
                    scores[pair] = cn_score
                end
            end
        end
    end
    return scores
end

function get_RWR_scores(g::SimpleGraph; alpha=0.5)
    eps = 1e-6
    scores = Dict{Tuple{Int,Int},Float64}()
    V = nv(g)
    adj_mat = adjacency_matrix(g, Bool)
    M_T = transpose(adj_mat ./ sum(adj_mat, dims=2))
    p_vec::Vector{Vector{Float64}} = []
    for v in vertices(g)
        R = zeros(V)
        R[v] = (1 - alpha)
        p_curr::Vector{Float64} = zeros(V)
        while true
            p_t = alpha * (M_T * p_curr) + R
            if sum((p_t .- p_curr) .^ 2) < eps
                p_curr .= p_t
                break
            end
            p_curr .= p_t
        end
        push!(p_vec, p_curr)
    end
    for x in vertices(g), y in vertices(g)
        if x != y
            pair = (min(x, y), max(x, y))
            scores[pair] = p_vec[x][y] + p_vec[y][x]
        end
    end

    return scores
end

function get_SRW_scores(g::SimpleGraph; lim=3)
    scores = Dict{Tuple{Int,Int},Float64}()
    V = nv(g)
    adj_mat = adjacency_matrix(g, Bool)
    M_T = transpose(adj_mat ./ sum(adj_mat, dims=2))
    superposed_p_vec::Vector{Vector{Float64}} = []
    for v in vertices(g)
        p_curr::Vector{Float64} = zeros(V)
        p_curr[v] = 1
        superposition = zeros(V)
        for _ in 1:lim
            p_curr = M_T * p_curr
            superposition .+= p_curr
        end
        push!(superposed_p_vec, superposition)
    end

    for x in vertices(g), y in vertices(g)
        if x != y
            pair = (min(x, y), max(x, y))
            scores[pair] = (degree(g, x) * superposed_p_vec[x][y] + degree(g, y) * superposed_p_vec[y][x])
        end
    end

    return scores
end

const score_type_dict = Dict(
    :cn => get_CN_scores,
    :ra => get_RA_scores,
    :rwr => get_RWR_scores,
    :srw => get_SRW_scores
)
