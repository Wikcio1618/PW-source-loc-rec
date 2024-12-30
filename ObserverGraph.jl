module ObserverGraph
export make_ER_obs_graph, make_BA_obs_graph, ObsGraph

using Graphs
using GraphPlot
using Random

struct ObsGraph
    graph::Graph
    observer_set::Set
    infected_set::Set
end

function make_ER_obs_graph(N::Integer, p::Float16, frac_observers)::ObsGraph
    @assert 0.0 <= p <= 1.0
    @assert 0.0 <= frac_observers <= 1.0

    g = Graphs.erdos_renyi(N, p)
    observer_set = Set(Random.randperm(N)[1:round(Int, frac_observers * N)])
    return ObsGraph(g, observer_set, Set())
end

function make_BA_obs_graph(N::Int, n0::Int, k::Int, frac_observers)::ObsGraph
    # @assert 0.0 <= p <= 1.0
    @assert 0.0 <= frac_observers <= 1.0

    g = Graphs.barabasi_albert(N, n0, k, complete=true)
    observer_set = Set(Random.randperm(N)[1:round(Int, frac_observers * N)])
    return ObsGraph(g, observer_set, Set())
end

end