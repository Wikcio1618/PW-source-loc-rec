using Graphs
using DataStructures

score_type_dict = Dict(
    :cn => get_CN_scores
)

"""
    Reconstructs the graph based on similarity score specified by `score_type`. `add_thresh` top scores are used to add unexisting links and, similarily, `hide_thresh` lowest scores are used to hide exisiting links
"""
function reconstruct_thresh!(g::SimpleGraph, hide_thresh::Int, add_thresh::Int, score_type=:cn)
    @assert (hide_thresh < length(min_heap) && add_thresh <= length(max_heap))
    @assert (hide_thresh + add_thresh <= nv(g))
    
    min_heap, max_heap = score_type_dict[score_type](g)
    for _ in 1:hide_thresh
        u, v = popfirst!(min_heap)
        rem_edge!(g, u, v)
    end
    for _ in 1:add_thresh
        u, v = popfirst!(max_heap)
        add_edge!(g, u, v)
    end
end

function get_CN_scores(g::SimpleGraph)::Tuple{PriorityQueue,PriorityQueue}
    min_heap = PriorityQueue{Tuple{Int,Int},Float64}(Base.Order.Forward)
    sizehint!(min_heap, nv(g)^2)
    max_heap = PriorityQueue{Tuple{Int,Int},Float64}(Base.Order.Reverse)
    sizehint!(max_heap, nv(g)^2)

    for u in vertices(g)
        neighbors = neighbors(g, u)
        for i in 1:length(neighbors)-1
            for j in i+1:length(neighbors)
                v, w = neighbors[i], neighbors[j]
                pair = (min(v, w), max(v, w))
                if !haskey(min_heap, pair)
                    cn_score = length(intersect(neighbors(g, v), neighbors(g, w)))

                    min_heap[pair] = cn_score
                    max_heap[pair] = cn_score
                end
            end
        end
    end

    return (min_heap, max_heap)
end

