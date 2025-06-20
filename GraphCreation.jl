using Graphs
using GraphPlot
using Random


function make_ER_graph(; V::Int, p::Float64)::SimpleGraph
    g = erdos_renyi(V, p)
    components = connected_components(g)
    for i in eachindex(components)[2:end]
        u = rand(components[i-1])
        v = rand(components[i])
        add_edge!(g, u, v)
    end
    return g
end

function make_BA_graph(; V::Int, n0::Int=missing, k::Int)::SimpleGraph
    if ismissing(n0)
        n0 = k
    end
    g = Graphs.barabasi_albert(V, n0, k, complete=true)
    return g
end

function get_FB_graph()::SimpleGraph
    return read_edges_file("networks/facebook_edges.csv")
end

function get_YST_graph()::SimpleGraph
    return read_edges_file("networks/yst_edges.csv")
end

function get_USA_graph()::SimpleGraph
    return read_net_file("networks/usa.net")
end

function get_CAL_graph()::SimpleGraph
    return read_edges_file("networks/california.csv")
end

function get_INF_graph()::SimpleGraph
    return read_edges_file("networks/inf_edges.txt")
end

function get_EMAIL_graph()::SimpleGraph
    return read_edges_file("networks/email.txt")
end

function get_CEL_graph()::SimpleGraph
    return read_edges_file("networks/cel_edges.txt")
end

function read_net_file(path)::SimpleGraph
    open(path, "r") do io
        lines = readlines(io)
        edge_start = findfirst(l -> occursin("*edges", lowercase(l)), lines)
        @assert edge_start !== nothing "No *edges section found in the file"

        g = SimpleGraph()
        for i in edge_start+1:length(lines)
            tokens = split(lines[i])
            if length(tokens) < 2
                continue
            end
            u, v = parse(Int, tokens[1]), parse(Int, tokens[2])
            if u != v
                while max(u, v) > nv(g)
                    add_vertex!(g)
                end
                add_edge!(g, u, v)
            end
        end

        preprocess_graph!(g)
        return g
    end
end

function read_edges_file(path::String)::SimpleGraph
    open(path, "r") do io
        lines = readlines(io)
        edges = []
        unq_dict = Dict{Int,Int}()
        unq = 1

        for line in lines
            line_vals = [parse(Int, num) for num in split(line, [',', ' '])]
            u = line_vals[1]
            v = line_vals[2]
            if !haskey(unq_dict, u)
                unq_dict[u] = unq
                unq += 1
            end
            if !haskey(unq_dict, v)
                unq_dict[v] = unq
                unq += 1
            end
            if u != v
                push!(edges, (u, v))
            end
        end
        g = SimpleGraph(unq-1)
        for (u, v) in edges
            add_edge!(g, unq_dict[u], unq_dict[v])
        end

        preprocess_graph!(g)
        return g
    end
end

function read_gml_file(file_path)::SimpleGraph
    node_map = Dict{Int,Int}()
    edges = []
    node_id_set = Set{Int}()

    open(file_path, "r") do io
        source = nothing
        for line in eachline(io)
            words = split(strip(line))
            if isempty(words)
                continue
            elseif words[1] == "id"
                node_id = parse(Int, words[2])
                push!(node_id_set, node_id)
            elseif words[1] == "source"
                source = parse(Int, words[2])
            elseif words[1] == "target" && !isnothing(source)
                target = parse(Int, words[2])
                if source != target
                    push!(edges, (source, target))
                end
                source = nothing
            end
        end
    end

    sorted_ids = sort(collect(node_id_set))
    for (new_idx, old_idx) in enumerate(sorted_ids)
        node_map[old_idx] = new_idx
    end

    num_nodes = length(node_map)
    g = SimpleGraph(num_nodes)

    for (src, dst) in edges
        add_edge!(g, node_map[src], node_map[dst])
    end

    preprocess_graph!(g)
    return g
end

"""
Remove isolated nodes, make it undirected, remove self loops, duplicated links
"""
function preprocess_graph!(g::SimpleGraph)
    components = connected_components(g)
    largest_component = argmax(map(length, components))
    g, _ = induced_subgraph(g, components[largest_component])
end


const graph_type_dict = Dict(
    :ba => make_BA_graph,
    :er => make_ER_graph,
    :usa => get_USA_graph,
    :fb => get_FB_graph,
    :inf => get_INF_graph,
    :yst => get_YST_graph,
    :email => get_EMAIL_graph,
    :cal => get_CAL_graph,
    :cel => get_CEL_graph
)