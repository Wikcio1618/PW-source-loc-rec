include("Localization.jl")
include("Propagation.jl")
include("ObserverGraph.jl")
include("StructModule.jl")
import .StructModule: LocData

graph_type_dict = Dict(
    :ba => make_BA_obs_graph,
    :er => make_ER_obs_graph
)

method_type_dict = Dict(
    :pearson => pearson_loc,
    :lptva => LPTVA_loc,
    :gmla => GMLA_loc
)

function precision_to_file(
    path::String,
    graph_type::Symbol,
    loc_type::Symbol,
    beta::Float64,
    N::Int,
    graph_args::Dict
)
    r = graph_args[:r]
    open(path, "w") do io
        println(io, "N=$N,graph=$graph_type,method=$loc_type,r=$r,beta=$beta,graph_args=$graph_args")
        println(io, "rank,precision")
        for _ in 1:(round(Int, N / 10))
            og = graph_type_dict[graph_type](;graph_args...)
            for _ in 1:10
                loc_data::LocData = propagate_SI!(og, r, beta)
                # loc_result::Vector{Tuple{Int,Float64}}
                if loc_type == :pearson
                    loc_result = method_type_dict[loc_type](og, loc_data.obs_data)
                else
                    loc_result = method_type_dict[loc_type](og, loc_data.obs_data, beta)
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