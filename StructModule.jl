module StructModule
export LocData, ObsGraph

using Graphs

struct LocData
    obs_data::Dict{Int,Int}
    source::Int
    t_start::Int
end

struct ObsGraph
    graph::Graph
    observer_set::Set
end
end