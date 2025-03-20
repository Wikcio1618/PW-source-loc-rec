using Graphs
using DataStructures


"""
    Reconstructs the graph based on similarity score specified by `score_type`. `add_thresh` top scores are used to add unexisting links and, similarily, `hide_thresh` lowest scores are used to hide exisiting links
"""
function reconstruct_thresh!(g::SimpleGraph, hide_thresh::Int, add_thresh::Int, score_type=:cn)
    @assert (hide_thresh + add_thresh <= ne(g))

    min_heap, max_heap = score_type_dict[score_type](g)
    @assert (hide_thresh < length(min_heap) && add_thresh <= length(max_heap))

    for _ in 1:hide_thresh
        (u, v) = dequeue!(min_heap)
        rem_edge!(g, u, v)
    end
    for _ in 1:add_thresh
        (u, v) = dequeue!(max_heap)
        add_edge!(g, u, v)
    end
end

function get_CN_scores(g::SimpleGraph)
    min_heap = PriorityQueue()
    sizehint!(min_heap, nv(g)^2)
    max_heap = PriorityQueue()
    sizehint!(max_heap, nv(g)^2)

    for u in vertices(g)
        neis = neighbors(g, u)
        for i in 1:length(neis)-1
            for j in i+1:length(neis)
                v, w = neis[i], neis[j]
                pair = (min(v, w), max(v, w))
                if !haskey(min_heap, pair)
                    cn_score = Float64(length(intersect(neighbors(g, v), neighbors(g, w))))

                    min_heap[pair] = cn_score
                    max_heap[pair] = -cn_score
                end
            end
        end
    end

    return (min_heap, max_heap)
end

score_type_dict = Dict(
    :cn => get_CN_scores
)