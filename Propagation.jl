module Propagation
export LocData, propagate_SI!, propagate_SI_parallel!

using ..ObserverGraph

using Graphs
using Random
using Base.Threads

struct LocData
    obs_data::Dict{Int,Int}
    source::Int
    t_start::Int
end

function propagate_SI!(og::ObsGraph, beta, t_max::Int=10^11)::LocData
    @assert 0.0 <= beta <= 1.0
    N::Int = nv(og.graph)
    t_start::Int = Random.rand(1:100) # arbitrary large number for possible start
    obs_dict = Dict()

    source = Random.rand(1:N)
    empty!(og.infected_set)
    push!(og.infected_set, source)
    if source in og.observer_set
        obs_dict[source] = t_start
    end

    t::Int = t_start + 1
    while (length(og.infected_set) < N)
        temp_set = Set()
        for i in og.infected_set
            for nei in Graphs.neighbors(og.graph, i)
                if (!(nei in og.infected_set) && Random.rand() < beta) # infection probability AND not already infected 
                    # infect node
                    push!(temp_set, nei)
                    # add observer data (observer_idx, timestamp)
                    if nei in og.observer_set
                        obs_dict[nei] = t
                    end
                end
            end
        end
        for i in temp_set
            push!(og.infected_set, i)
        end

        t += 1
        if t >= t_max
            break
        end
    end

    return LocData(obs_dict, source, t_start)
end

function propagate_SI_parallel!(og::ObsGraph, beta, t_max::Int=10^11)::LocData
    @assert 0.0 <= beta <= 1.0
    N::Int = nv(og.graph)
    t_start::Int = Random.rand(1:100)  # arbitrary large number for possible start
    obs_dict = Dict()

    source = Random.rand(1:N)
    empty!(og.infected_set)
    push!(og.infected_set, source)
    if source in og.observer_set
        obs_dict[source] = t_start
    end

    t::Int = t_start + 1

    # Create locks for synchronization
    infected_set_lock = ReentrantLock()
    temp_set_lock = ReentrantLock()
    obs_dict_lock = ReentrantLock()

    while length(og.infected_set) < N
        temp_set = Set()

        # Convert the infected_set into an array for iteration
        infected_array = collect(og.infected_set)

        # Parallelize the processing of infected nodes
        @threads for i in infected_array
            for nei in Graphs.neighbors(og.graph, i)
                if !(nei in og.infected_set) && Random.rand() < beta
                    # Infection occurs, add to temp_set
                    lock(temp_set_lock) do
                        push!(temp_set, nei)
                    end
                    # Add observer data if needed
                    lock(obs_dict_lock) do
                        if nei in og.observer_set
                            obs_dict[nei] = t
                        end
                    end
                end
            end
        end

        # Add the newly infected nodes to the infected_set
        lock(infected_set_lock) do
            union!(og.infected_set, temp_set)
        end

        t += 1
        if t >= t_max
            break
        end
    end

    return LocData(obs_dict, source, t_start)
end
end