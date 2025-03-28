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

    g_mod = inplace ? g : copy(g)

    mod_func = type == :add ? add_edge! : rem_edge!
    for _ in 1:k
        (u, v) = dequeue!(heap)
        mod_func(g_mod, u, v)
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
    scores = Dict{Tuple{Int,Int},Float64}()
    V = nv(g)
    adj_mat = adjacency_matrix(g, Float64)
    _degs = degree(g)
    _degs[_degs.==0] .= 1.0 # prevent division by 0 for isolated nodes
    M_T = transpose(adj_mat ./ _degs)
    p_vecs = Array{Float64,2}(undef, (V, V)) # p-ility of reaching node j from i
    for v in vertices(g)
        R = zeros(V)
        R[v] = 1 - alpha
        p_curr::Vector{Float64} = zeros(V)
        for _ in 1:max_iter
            p_t = alpha * (M_T * p_curr) .+ R
            if norm(p_t .- p_curr) < eps
                p_curr .= p_t
                break
            end
            p_curr .= p_t
        end
        p_vecs[v, :] .= p_curr
    end
    for x in 1:V, y in (x+1):V
        scores[(x, y)] = p_vecs[x, y] + p_vecs[y, x]
    end

    return scores
end

function get_SRW_scores(g::SimpleGraph; lim=3)
    scores = Dict{Tuple{Int,Int},Float64}()
    V = nv(g)
    adj_mat = adjacency_matrix(g, Float64)
    degs = degree(g)
    degs[degs.==0] .= 1.0 # prevent division by 0 for isolated nodes
    M_T = transpose(adj_mat ./ degs)
    superposed_p_vecs = Array{Float64,2}(undef, (V, V)) # summed p-ilities of reaching node j from i in step 1,...,lim
    for v in vertices(g)
        p_curr::Vector{Float64} = zeros(V)
        p_curr[v] = 1
        superposition = zeros(V)
        for _ in 1:lim
            p_curr = M_T * p_curr
            superposition .+= p_curr
        end
        superposed_p_vecs[v, :] .= superposition
    end
    for x in vertices(g), y in vertices(g)
        if x != y
            pair = (min(x, y), max(x, y))
            scores[pair] = (degs[x] * superposed_p_vecs[x, y] + degs[y] * superposed_p_vecs[y, x])
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
