module StructModule
export LocData, ObsGraph

using Graphs

struct LocData
    obs_data::Dict{Int,Int}
    source::Int
    t_start::Int
end

end