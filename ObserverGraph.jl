include("StructModule.jl")

using Graphs
using GraphPlot
using Random

function make_ER_obs_graph(;V::Int, p::Float64, r::Float64)::ObsGraph
    @assert 0.0 <= p <= 1.0
    @assert 0.0 <= r <= 1.0

    g = Graphs.erdos_renyi(V, p)
    return ObsGraph(g, Set(), Set())
end

function make_BA_obs_graph(;V::Int, n0::Int, k::Int, r::Float64)::ObsGraph
    @assert 0.0 <= r <= 1.0
    
    g = Graphs.barabasi_albert(V, n0, k, complete=true)
    return ObsGraph(g, Set(), Set())
end