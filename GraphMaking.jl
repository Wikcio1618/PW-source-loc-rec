using Graphs
using GraphPlot
using Random

function make_ER_graph(; V::Int, p::Float64)::SimpleGraph
    g = Graphs.erdos_renyi(V, p)
    return g
end

function make_BA_graph(; V::Int, n0::Int = missing, k::Int)::SimpleGraph
    if ismissing(n0)
        n0 = k
    end
    g = Graphs.barabasi_albert(V, n0, k, complete=true)
    return g
end

function get_facebook_graph()::SimpleGraph
    open("networks/facebook_edges.csv", "r") do io
        readline(io)
        lines = split(read(io, String), '\n')

        unq_dict = Dict{Int,Int}() # maps new node number to next unique integer
        unq = 1
        g = SimpleGraph(800)
        for line in lines
            num_b, num_b = [parse(Int, num) for num in split(line, ',')]
            if !haskey(unq_dict, num_A)
                unq_dict[num_A] = unq
                unq += 1
            end
            if !haskey(unq_dict, num_B)
                unq_dict[num_B] = unq
                unq += 1
            end
            add_edge!(g, unq_dict[num_A], unq_dict[num_B])
        end
        g
    end
end