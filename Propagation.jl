include("StructModule.jl")

using Graphs
using Random

function propagate_SI!(g::SimpleGraph, r::Float64, beta::Float64, t_max::Int=10^11)::LocData
    @assert 0.0 < beta < 1.0
    @assert 0.0 < r <= 1.0
    V::Int = nv(g)
    t_start::Int = Random.rand(1:100) # arbitrary large number for possible start
    obs_dict = Dict{Int, Int}()
    
    observer_set = Set(Random.randperm(V)[1:round(Int, r * V)])

    source = Random.rand(1:V)

    infected_set = Set{Int}(source)

    if source in observer_set
        obs_dict[source] = t_start
    end

    t::Int = t_start+1
    while (length(infected_set) < V)
        temp_set = Set()
        for i in infected_set
            for nei in Graphs.neighbors(g, i)
                if (!(nei in infected_set) && Random.rand() < beta) # infection probability AND not already infected 
                    push!(temp_set, nei)
                    # add observer data (observer_idx, timestamp)
                    if nei in observer_set
                        obs_dict[nei] = t
                    end
                end
            end
        end
        for i in temp_set
            push!(infected_set, i)
        end

        t += 1
        if t >= t_max
            break
        end
    end

    return LocData(obs_dict, source, t_start)
end
