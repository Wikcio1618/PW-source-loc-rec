using Graphs
using DataStructures
using LinearAlgebra
using SparseArrays

"""
Modifies the `heap` and modifies the graph depending on `inplace`
"""
function reconstruct_top_k!(g::SimpleGraph, heap::PriorityQueue, k::Int; type=:add, inplace=false)::SimpleGraph
    @assert type in [:add, :hide]
    @assert (k < length(heap))
    if type == :add
        @assert k < nv(g)^2 - ne(g) # number of possible new edges is greater than k (while loop is safe)
    else
        @assert k < ne(g)
    end
    g_mod = inplace ? g : copy(g)
    mod_func = type == :add ? add_edge! : rem_edge!
    while (k > 0)
        (u, v) = dequeue!(heap)
        if mod_func(g_mod, u, v)
            k += -1
        end
    end
    return g_mod
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
                    ra_score = sum(i -> degree(g, i), intersect(neighbors(g, x), neighbors(g, y)))
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
    max_iter = 1000
    V = nv(g)

    adj_mat = adjacency_matrix(g, Float64)
    degs = degree(g)
    degs[degs.==0] .= 1.0
    M_T = adj_mat ./ degs'
    scores = Dict{Tuple{Int,Int},Float64}()

    for v in 1:V
        R = zeros(V)
        R[v] = 1 - alpha
        p_curr = zeros(V)
        for _ in 1:max_iter
            mul!(p_curr, M_T, p_curr)  # In-place multiplication: p_curr = M_T * p_curr
            p_curr .*= alpha
            p_curr .+= R

            if norm(p_curr - R) < eps
                break
            end
        end
        for u in (v+1):V
            scores[(v, u)] = p_curr[u] + p_curr[v]
        end
    end
    return scores
end

function get_SRW_scores(g::SimpleGraph; lim=3)
    V = nv(g)
    scores = Dict{Tuple{Int,Int},Float64}()

    adj_mat = adjacency_matrix(g, Float64)
    degs = degree(g)
    degs[degs.==0] .= 1.0  # Prevent division by 0
    M_T = adj_mat ./ degs'

    for v in 1:V
        p_curr = zeros(V)
        p_curr[v] = 1
        superposition = zeros(V)
        for _ in 1:lim
            mul!(superposition, M_T, p_curr, 1.0, 1.0)  # superposition += M_T * p_curr
            mul!(p_curr, M_T, p_curr)  # p_curr = M_T * p_curr
        end
        for u in (v+1):V
            scores[(v, u)] = degs[v] * superposition[u] + degs[u] * superposition[v]
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
