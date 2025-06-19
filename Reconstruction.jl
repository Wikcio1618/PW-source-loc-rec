using Graphs
using DataStructures
using LinearAlgebra
using SparseArrays

include("Localization.jl")

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

""" 
Calculates scores of each non-observed link by calculating how much its presence increases pearson correlation.\n
New correlation is calculated for each of `S0` randomly selected `tester nodes`.\n
Final score for a given link is taken as maximum pearson correlation increase over all S0 nodes.
"""
function get_BRUTE_PEARSON_scores(g::SimpleGraph, S0=missing; obs_data::Dict{Int,Int})::Dict{Tuple{Int,Int},Float64}
    V = nv(g)
    @assert ismissing(S0) || S0 <= V
    if ismissing(S0)
        S0 = sqrt(V)
    end
    scores = Dict{Tuple{Int,Int},Float64}()

    # Create dictionary mapping each tester_node to its base pearson correlation
    base_tester_scores = Dict{Int,Float64}()
    while length(base_tester_scores) < S0
        rnd = rand(1:V)
        if !haskey(base_tester_scores, rnd)
            base_tester_scores[rnd] = single_node_pearson_score(g, rnd, obs_data)
        end
    end

    # calculate score for each non-observed link
    for i in 1:V, j in i+1:V
        if has_edge(g, i, j)
            continue
        end
        add_edge!(g, i, j)
        max_increase = -Inf
        for (tester, base_score) in base_tester_scores
            increase = abs(single_node_pearson_score(g, tester, obs_data)) - abs(base_score)
            max_increase = increase > max_increase ? increase : max_increase
        end
        scores[(i, j)] = max_increase
    end

    return scores
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

function get_FP_scores(g::SimpleGraph; alpha=0.5, eps=1e-6, max_iter=10000)
    V = nv(g)
    A = adjacency_matrix(g, Float64)

    row_sums = sum(A, dims=2)  # sum over rows (i.e., for Dl)
    col_sums = sum(A, dims=1)  # sum over columns (i.e., for Dr)
    row_sums[row_sums.==0.0] .= 1.0
    col_sums[col_sums.==0.0] .= 1.0

    Dl = Diagonal(1.0 ./ map(sqrt, row_sums[:]))
    Dr = Diagonal(1.0 ./ map(sqrt, col_sums[:]))
    M = Dl * A * Dr  # Bi-normalized adjacency

    all_p = Matrix{Float64}(undef, V, V)
    for x in 1:V
        y = zeros(V)
        y[x] = 1.0
        f = copy(y)

        for _ in 1:max_iter
            f_new = alpha * (M * f) + (1 - alpha) * y
            if norm(f_new - f) < eps
                break
            end
            f = f_new
        end
        all_p[x, :] .= f
    end

    scores = Dict{Tuple{Int,Int},Float64}()
    for x in 1:V, y in (x+1):V
        if !has_edge(g, x, y)
            scores[(x, y)] = all_p[x, y] + all_p[y, x]
        end
    end

    return scores
end

function get_SRW_scores(g::SimpleGraph; lim=3)
    V = nv(g)

    adj_mat = adjacency_matrix(g, Float64)
    degs = degree(g)
    degs[degs.==0] .= 1.0
    M_T = adj_mat ./ degs'

    # Store superposed probabilities: each row i contains superposition from node i
    all_superpositions = Matrix{Float64}(undef, V, V)
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

    scores = Dict{Tuple{Int,Int},Float64}()
    for x in 1:V, y in (x+1):V
        scores[(x, y)] = has_edge(g, x, y) ? 0.0 : degs[x] * all_superpositions[x, y] + degs[y] * all_superpositions[y, x]
    end

    return scores
end

function get_CN_scores(g::SimpleGraph)::Dict{Tuple{Int,Int},Float64}
    V = nv(g)
    scores = Dict{Tuple{Int,Int},Float64}()

    for u in 1:V, v in (u+1):V
        if !has_edge(g, u, v)
            cn_score = length(intersect(neighbors(g, u), neighbors(g, v)))
            scores[(u, v)] = cn_score
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

function get_MERW_scores(g::SimpleGraph; eps=1e-6, max_iter=1000)
    A = adjacency_matrix(g)
    V = nv(g)

    λ_list, ψ_mat = eigen(Matrix(A))
    λ = real(λ_list[1])
    ψ = real(ψ_mat[:, 1])
    ψ ./= norm(ψ)

    M = sparse(zeros(V, V))
    for i in 1:V, j in 1:V
        if A[i, j] != 0
            M[i, j] = A[i, j] / λ * ψ[j] / ψ[i]
        end
    end

    all_p = zeros(V, V)
    for v in 1:V
        p = zeros(V)
        p[v] = 1.0
        for _ in 1:max_iter
            p_new = M * p
            if norm(p_new .- p) < eps
                break
            end
            p = p_new
        end
        all_p[v, :] .= p
    end

    scores = Dict{Tuple{Int,Int},Float64}()
    for x in 1:V, y in (x+1):V
        scores[(x, y)] = all_p[x, y] + all_p[y, x]
    end

    return scores
end

function get_RW_scores(g::SimpleGraph; eps=1e-14, max_iter=10000000)
    V = nv(g)
    adj_mat = adjacency_matrix(g, Float64)
    degs = degree(g)
    degs[degs.==0] .= 1.0
    M_T = adj_mat ./ degs'

    all_p = Matrix{Float64}(undef, V, V)
    errs = Vector(undef, V)
    p_curr = zeros(V)
    p_next = similar(p_curr)

    for x in 1:V
        conv = []
        fill!(p_curr, 0.0)
        p_curr[x] = 1.0
        for _ in 1:max_iter
            mul!(p_next, M_T, p_curr)
            err = norm(p_next .- p_curr)
            append!(conv, err)
            if err < eps
                break
            end
            copy!(p_curr, p_next)
        end
        errs[x] = conv
        all_p[x, :] .= p_curr
    end
    # return errs

    scores = Dict{Tuple{Int,Int},Float64}()
    for x in 1:V, y in (x+1):V
        if !has_edge(g, x, y)
            if abs(all_p[x, y] - all_p[y, x]) > 0.01
                println(x, y)
            end
            scores[(x, y)] = all_p[x, y] + all_p[y, x]
        end
    end

    return g
end

const score_type_dict = Dict(
    :cn => get_CN_scores,
    :ra => get_RA_scores,
    :rw => get_RW_scores,
    :rwr => get_RWR_scores,
    :fp => get_FP_scores,
    :srw => get_SRW_scores,
    :merw => get_MERW_scores,
    :bp => get_BRUTE_PEARSON_scores
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
