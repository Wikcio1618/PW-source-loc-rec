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
                    ra_score = sum(i -> 1 / degree(g, i), intersect(neighbors(g, x), neighbors(g, y)))
                    scores[pair] = ra_score
                end
            end
        end
    end
    return scores
end

function get_RWR_scores(g::SimpleGraph; alpha=0.5)
    eps = 1e-5
    max_iter = 100
    V = nv(g)

    adj_mat = adjacency_matrix(g, Float64)
    degs = degree(g)
    degs[degs.==0] .= 1.0
    M_T = adj_mat ./ degs'

    all_p = Matrix{Float64}(undef, V, V)
    p_curr = zeros(V)
    p_t = similar(p_curr)
    R = zeros(V)
    for v in 1:V
        fill!(p_curr, 0.0)
        fill!(R, 0.0)
        R[v] = 1 - alpha
        p_curr = zeros(V)
        for _ in 1:max_iter
            mul!(p_t, M_T, p_curr)             # p_t = M_T * p_curr
            @. p_t = alpha * p_t + R           # fused broadcast, avoids allocation
            if norm(p_t .- p_curr) < eps
                copy!(p_curr, p_t)
                break
            end
            copy!(p_curr, p_t)
        end
        all_p[v, :] .= p_curr
    end
    scores = Dict{Tuple{Int,Int},Float64}()
    for x in 1:V, y in (x+1):V
        scores[(x, y)] = all_p[x, y] + all_p[y, x]
    end

    return scores
end

function get_SRW_scores(g::SimpleGraph; lim=3)
    V = nv(g)
    scores = Dict{Tuple{Int,Int},Float64}()

    adj_mat = adjacency_matrix(g, Float64)
    degs = degree(g)
    degs[degs.==0] .= 1.0
    M_T = adj_mat ./ degs'

    # Store superposed probabilities: each row i contains superposition from node i
    all_superpositions = zeros(V, V)
    p_curr = zeros(V)
    p_t = similar(p_curr)
    for v in 1:V
        fill!(p_curr, 0.0)
        p_curr[v] = 1.0
        superposition = zeros(V)
        for _ in 1:lim
            mul!(p_t, M_T, p_curr)
            superposition .+= p_t
            copy!(p_curr, p_t)
        end
        all_superpositions[v, :] .= superposition
    end

    for x in 1:V, y in (x+1):V
        scores[(x, y)] = degs[x] * all_superpositions[x, y] + degs[y] * all_superpositions[y, x]
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
