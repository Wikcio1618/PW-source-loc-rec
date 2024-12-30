module Localization
export pearson_loc

using ..ObserverGraph
using ..Propagation

using Graphs
using Statistics

function pearson_loc(og::ObsGraph, obs_data::Dict{Int,Int})::Int
    N = nv(og.graph)
    O = length(obs_data)
    obs_idxs = collect(keys(obs_data))
    obs_times = collect(values(obs_data))
    

    # 2d array - each row is distances from every observer
    LENS = Array{Int, 2}(undef, O, N)

    # for every observer
    for (i, obs_idx) in enumerate(obs_idxs)
        LENS[i, :] = dijkstra_shortest_paths(og.graph, obs_idx).dists
    end

    # for every node
    covs::Vector{Float64} = [Statistics.cor(LENS[:, i], obs_times) for i in 1:N]
    println(covs)
    return argmax(covs)
end

end