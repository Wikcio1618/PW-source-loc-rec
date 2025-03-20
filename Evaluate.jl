include("Localization.jl")
include("Propagation.jl")
include("GraphCreation.jl")
include("StructModule.jl")
include("Modification.jl")
include("Reconstruction.jl")
import .StructModule: LocData

using Graphs

graph_type_dict = Dict(
    :ba => make_BA_graph,
    :er => make_ER_graph
)

loc_type_dict = Dict(
    :pearson => pearson_loc,
    :lptva => LPTVA_loc,
    :gmla => GMLA_loc
)
modify_type_dict = Dict(
    :hide => modify_hide,
    :add => modify_add
)

function evaluate_reconstruct_to_file(
    path::String,
    graph_type::Symbol,
    loc_type::Symbol,
    beta::Float64,
    r::Float64,
    N::Int,
    graph_args::Dict,
    modify_type::Symbol,
    dj::Float64,
    reconstruct_type::Symbol,
    hide_thresh::Int,
    add_thresh::Int
)
    @assert haskey(graph_type_dict, graph_type)
    @assert haskey(loc_type_dict, loc_type)
    @assert haskey(modify_type_dict, modify_type)
    @assert reconstruct_type in [:cn]

    open(path, "w") do io
        println(io, "N=$N,graph=$graph_type,method=$loc_type,r=$r,beta=$beta,graph_args=$graph_args,modify_type=$modify_type,dj=$dj,reconstruct_type=$reconstruct_type,hide_thresh=$hide_thresh,add_thresh=$add_thresh")
        println(io, "rank,precision,dj")
        for _ in 1:max(1, (round(Int, N / 25)))
            g = graph_type_dict[graph_type](; graph_args...)
            for _ in 1:5
                loc_data::LocData = propagate_SI!(g, r, beta)
                # for each modification
                for _ in 1:5
                    if !ismissing(modify_type)
                        new_g = modify_type_dict[modify_type](g, dj)
                        reconstruct_thresh!(new_g, hide_thresh, add_thresh, reconstruct_type)
                        if loc_type == :pearson
                            loc_result = loc_type_dict[loc_type](new_g, loc_data.obs_data)
                        else
                            loc_result = loc_type_dict[loc_type](new_g, loc_data.obs_data, beta)
                        end
                        rank = calc_rank(loc_data, loc_result, graph_args[:V])
                        prec = calc_prec(loc_data, loc_result)
                        dj = calc_jaccard(g, new_g)
                        println(io, "$rank,$prec,$dj")
                    end
                end
            end
        end
    end
end

function evaluate_modify_to_file(
    path::String,
    graph_type::Symbol,
    loc_type::Symbol,
    beta::Float64,
    r::Float64,
    N::Int,
    graph_args::Dict,
    modify_type::Symbol,
    dj::Float64,
)
    @assert haskey(graph_type_dict, graph_type)
    @assert haskey(loc_type_dict, loc_type)
    @assert (
        if !ismissing(modify_type)
            haskey(modify_type_dict, modify_type)
        end
    )

    open(path, "w") do io
        println(io, "N=$N,graph=$graph_type,method=$loc_type,r=$r,beta=$beta,graph_args=$graph_args,modify_type=$modify_type,dj=$dj")
        println(io, "rank,precision")
        for _ in 1:max(1, (round(Int, N / 25)))
            g = graph_type_dict[graph_type](; graph_args...)
            for _ in 1:5
                loc_data::LocData = propagate_SI!(g, r, beta)
                # for each modification
                for _ in 1:5
                    if !ismissing(modify_type)
                        modify_type_dict[modify_type](g, dj; inplace=true)
                        if loc_type == :pearson
                            loc_result = loc_type_dict[loc_type](g, loc_data.obs_data)
                        else
                            loc_result = loc_type_dict[loc_type](g, loc_data.obs_data, beta)
                        end
                        rank = calc_rank(loc_data, loc_result, graph_args[:V])
                        prec = calc_prec(loc_data, loc_result)
                        println(io, "$rank,$prec")
                    end
                end
            end
        end
    end
end

function evaluate_original_to_file(
    path::String,
    graph_type::Symbol,
    loc_type::Symbol,
    beta::Float64,
    r::Float64,
    N::Int,
    graph_args::Dict
)
    @assert haskey(graph_type_dict, graph_type)
    @assert haskey(loc_type_dict, loc_type)

    open(path, "w") do io
        println(io, "N=$N,graph=$graph_type,method=$loc_type,r=$r,beta=$beta,graph_args=$graph_args")
        println(io, "rank,precision")
        for _ in 1:max(1, (round(Int, N / 10)))
            g = graph_type_dict[graph_type](; graph_args...)
            for _ in 1:10
                loc_data::LocData = propagate_SI!(g, r, beta)
                if loc_type == :pearson
                    loc_result = loc_type_dict[loc_type](g, loc_data.obs_data)
                else
                    loc_result = loc_type_dict[loc_type](g, loc_data.obs_data, beta)
                end
                rank = calc_rank(loc_data, loc_result, graph_args[:V])
                prec = calc_prec(loc_data, loc_result)
                println(io, "$rank,$prec")
            end
        end
    end
end

function calc_rank(loc_data::LocData, loc_result::Vector{Tuple{Int,Float64}}, V::Int)::Int
    true_source = loc_data.source
    for (rank, (idx, score)) in enumerate(loc_result)
        if idx == true_source
            return rank
        end
    end
    return V
end

function calc_prec(loc_data::LocData, loc_result::Vector{Tuple{Int,Float64}})::Float64
    true_source = loc_data.source
    best_score = loc_result[1][2]
    predicted_exequo = false

    num_exequo = 0
    for (idx, score) in loc_result
        if idx == true_source && score == best_score
            predicted_exequo = true
        elseif score < best_score
            break
        end
        num_exequo += 1
    end
    return predicted_exequo ? (1 / num_exequo) : 0
end

function calc_jaccard(g1::SimpleGraph, g2::SimpleGraph)::Float64
    edges1 = Set([(min(e.src, e.dst), max(e.src, e.dst)) for e in edges(g1)])
    edges2 = Set([(min(e.src, e.dst), max(e.src, e.dst)) for e in edges(g2)])

    intersection_size = length(intersect(edges1, edges2))
    union_size = length(union(edges1, edges2))

    return union_size == 0 ? 1.0 : 1 - intersection_size / union_size
end