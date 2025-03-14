include("StructModule.jl")
using .StructModule
include("GraphMaking.jl")

using Graphs
using Random
using NLsolve

function add_modify(g::SimpleGraph, dj::Float64)::SimpleGraph
    @assert 0.0 <= dj < 1.0
    new_g = copy(g)
    V = nv(new_g)
    M = round(Int, dj / (1 - dj) * ne(new_g))
    for _ in 1:M
        u, v = rand(1:V, 2)
        while has_edge(new_g, u, v) || u == v
            u, v = rand(1:V, 2)
        end
        add_edge!(new_g, u, v)
    end
    return g
end

function hide_modify(g::SimpleGraph, dj::Float64)::SimpleGraph
    @assert 0.0 <= dj < 1.0

    new_g = copy(g)
    M = dj * ne(new_g)
    edges_list = collect(edges(new_g))
    for e in sample(edges_list, M; replace=false)
        rem_edge!(new_g, src(e), dst(e))
    end
    return new_g
end

function fluctuation_modify(new_g::SimpleGraph, dj::Float64, graph_type::Symbol, graph_args::Dict)::SimpleGraph
    @assert 0.0 <= dj < 1.0
    @assert graph_type in [:er, :ba, :facebook, :california, :email]

    function theta_equations!(F, thetas, ki_mean)
        N = length(ki_mean)
        for i in 1:N
            F[i] = ki_mean[i] - sum(1 / (1 + exp(thetas[i] + thetas[j])) for j in 1:N)
        end
    end
    theta_init = zeros(length(ki_mean))

    V = graph_args[:V]
    new_g = copy(g)

    if graph_type == :er # Erdos-Renyi graph
        p = graph_args[:p]
        theta = log1p((1 - p) / p) / 2
        thetas = [theta for _ in 1:V]
    elseif graph_type == :ba # Barabasi-Albert graph
        ki_mean = zeros(V)
        M = 10
        for _ in 1:10
            ki_mean .+= degree(make_BA_graph(; graph_args...))
        end
        ki_mean ./= M
        thetas = nlsolve(theta -> theta_equations!(theta, theta, ki_mean), theta_init).zero
    else # one of real graphs (email, facebook, california)
        ki_mean = degree(graph_to_make_method_dict[graph_type]())
        thetas = nlsolve(theta -> theta_equations!(theta, theta, ki_mean), theta_init).zero
    end

    return new_g
end

graph_to_make_method_dict = Dict(
    :ba => make_BA_graph,
    :er => make_ER_graph,
    :facebook => get_facebook_graph
)