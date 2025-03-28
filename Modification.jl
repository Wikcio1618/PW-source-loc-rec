using Graphs
using Random

# each method returns a new graph and a new LocData since some observers can be dropped because of incomplete graph

function modify_add(g::SimpleGraph, loc_data=LocData(Dict(), 1, 1), dj::Float64=0.0)::Tuple{SimpleGraph,LocData}
    @assert 0.0 <= dj < 1.0
    g_mod = copy(g)

    V = nv(g_mod)
    M = round(Int, dj / (1 - dj) * ne(g_mod))

    for _ in 1:M
        u, v = rand(1:V, 2)
        while u == v || has_edge(g_mod, u, v)
            u, v = rand(1:V, 2)
        end
        add_edge!(g_mod, u, v)
    end

    return (g_mod, copy(loc_data))
end


function modify_hide(g::SimpleGraph, loc_data=LocData(Dict(), 1, 1), dj::Float64=0.0)::Tuple{SimpleGraph,LocData}
    @assert 0.0 <= dj < 1.0

    g_mod = copy(g)
    M = round(Int, dj * ne(g_mod))
    edges_list = collect(edges(g_mod))
    for e in shuffle(edges_list)[1:M]
        rem_edge!(g_mod, src(e), dst(e))
    end

    components = connected_components(g_mod)
    largest_component = argmax(map(length, components))
    sg, vmap = induced_subgraph(g_mod, components[largest_component])

    obs_data_mod = Dict(i => loc_data.obs_data[old_idx]
                        for (i, old_idx) in enumerate(vmap)
                        if haskey(loc_data.obs_data, old_idx))

    source_mod = findfirst(x -> x == loc_data.source, vmap)
    if source_mod === nothing # source not found in the largest component (localization will be for 100% unsuccessful)
        source_mod = loc_data.source
    end

    # println("$dj, $(length(components))")

    return (sg, LocData(obs_data_mod, source_mod, loc_data.t_start))
end

function modify_fluctuate(g::SimpleGraph, loc_data::LocData, dj::Float64)::Tuple{SimpleGraph,LocData}
    @assert 0.0 <= dj < 1.0
    @assert graph_type in [:er, :ba, :facebook, :california, :email]

    # 1. 2 linki losowa
    # 2. (A - B; C - D) -> (A - C; B - D)

    return g
end

function calc_jaccard(g1::SimpleGraph, g2::SimpleGraph, vmap::Vector{Int})::Float64
    edges1 = Set([(min(e.src, e.dst), max(e.src, e.dst)) for e in edges(g1)])
    edges2 = Set([(min(vmap[e.src], vmap[e.dst]), max(vmap[e.src], vmap[e.dst])) for e in edges(g2)])

    intersection_size = length(intersect(edges1, edges2))
    union_size = length(union(edges1, edges2))

    return union_size == 0 ? 1.0 : 1 - intersection_size / union_size
end

const modify_type_dict = Dict(
    :hide => modify_hide,
    :add => modify_add
)