using Graphs
using Random

function modify_add(g::SimpleGraph, dj::Float64; inplace=false)::SimpleGraph
    @assert 0.0 <= dj < 1.0
    g_mod = inplace ? g : copy(g)

    V = nv(g_mod)
    M = round(Int, dj / (1 - dj) * ne(g_mod))

    for _ in 1:M
        u, v = rand(1:V, 2)
        while has_edge(g_mod, u, v) || u == v
            u, v = rand(1:V, 2)
        end
        add_edge!(g_mod, u, v)
    end

    return g_mod
end


function modify_hide(g::SimpleGraph, dj::Float64; inplace=false)::SimpleGraph
    @assert 0.0 <= dj < 1.0
    if !inplace
        g = copy(g)
    end
    M = dj * ne(g)
    edges_list = collect(edges(g))
    for e in sample(edges_list, M; replace=false)
        rem_edge!(g, src(e), dst(e))
    end
    # TODO - najwiekszy component
    # How to keep source and observers? RELABELLING!
    components = connected_components(g)  # Get all components as vertex sets
    largest_comp = argmax(length, components)  # Find the largest component

    return induced_subgraph(g, components[largest_comp])
end

function modify_fluctuate(g::SimpleGraph, dj::Float64)::SimpleGraph
    @assert 0.0 <= dj < 1.0
    @assert graph_type in [:er, :ba, :facebook, :california, :email]

    # 1. 2 linki losowa
    # 2. (A - B; C - D) -> (A - C; B - D)

    return g
end

const modify_type_dict = Dict(
    :hide => modify_hide,
    :add => modify_add
)