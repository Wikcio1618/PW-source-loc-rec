using Graphs
using DataStructures


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
                v, w = neis[i], neis[j]
                pair = (min(v, w), max(v, w))
                if !haskey(scores, pair)
                    ra_score = sum(x -> length(x) == 0 ? 0 : 1 / degree(g, x), intersect(neighbors(g, v), neighbors(g, w)))
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
                v, w = neis[i], neis[j]
                pair = (min(v, w), max(v, w))
                if !haskey(scores, pair)
                    cn_score = Float64(length(intersect(neighbors(g, v), neighbors(g, w))))
                    scores[pair] = cn_score
                end
            end
        end
    end
    return scores
end

const score_type_dict = Dict(
    :cn => get_CN_scores,
    :ra => get_RA_scores
)
