using Revise
using Base.Threads

include("Localization.jl")
include("Propagation.jl")
include("GraphCreation.jl")
include("Modification.jl")
includet("Reconstruction.jl")

using Graphs
using Random
using DataStructures


function evaluate_reconstruct_to_file(
    graph_type::Symbol,
    loc_type::Symbol,
    beta::Float64,
    r::Float64,
    N::Int,
    modify_type::Symbol,
    dj::Float64,
    reconstruct_type::Symbol,
    k_vec
)
    @assert haskey(graph_type_dict, graph_type)
    @assert haskey(loc_type_dict, loc_type)
    @assert haskey(modify_type_dict, modify_type)
    @assert haskey(score_type_dict, reconstruct_type)

    path = "data/rec_$(String(reconstruct_type))_dj$(round(Int, dj*100))_$(String(graph_type))_$(loc_type)_r$(round(Int,r*100))_beta$(round(Int, beta*100)).csv"

    open(path, "w") do io
        println(io, "N=$N,graph=$graph_type,method=$loc_type,r=$r,beta=$beta,modify_type=$modify_type,dj=$dj,reconstruct_type=$reconstruct_type")
        println(io, "k,prec,std_err")
        g = graph_type_dict[graph_type]()
        precs = Array{Float64,2}(undef, (length(k_vec), N))
        for t in 1:N
            if t % round(Int, N / 10) == 0
                println("$reconstruct_type, dj=$dj: Starting $t iteration")
            end
            loc_data::LocData = propagate_SI!(g, r, beta)
            sg, new_loc_data = modify_type_dict[modify_type](g, loc_data, dj)
            heap = PriorityQueue(
                modify_type == :hide ? Base.Order.Reverse : Base.Order.Forward,
                score_type_dict[reconstruct_type](sg)
            )
            for i in eachindex(k_vec)
                reconstruct_top_k!(
                    sg,
                    heap,
                    i == 1 ? k_vec[i] : k_vec[i] - k_vec[i-1];
                    type=modify_type == :hide ? :add : :hide,
                    inplace=true
                )
                loc_result =
                    loc_type == :pearson ?
                    loc_type_dict[loc_type](sg, new_loc_data.obs_data) :
                    loc_type_dict[loc_type](sg, new_loc_data.obs_data, beta)

                precs[i, t] = calc_prec(new_loc_data, loc_result)
            end
        end
        for (i, k) in enumerate(k_vec)
            prec = mean(precs[i, :])
            err = std(precs[i, :]) / sqrt(N)
            println(io, "$k,$prec,$err")
        end
    end
end


function evaluate_original_to_file(
    graph_type::Symbol,
    loc_type::Symbol,
    beta::Float64,
    r::Float64,
    N::Int;
    graph_args=Dict()
)
    @assert haskey(graph_type_dict, graph_type)
    @assert haskey(loc_type_dict, loc_type)

    path = "data/loc_$(String(graph_type))_$(loc_type)_r$(round(Int,r*100))_beta$(round(Int, beta*100)).csv"
    g = graph_type_dict[graph_type](graph_args...)
    output = Vector{String}(undef, N)

    @threads for t in 1:N
        if t % (div(N , 10)) == 0
            println("Starting $t iteration")
        end
        g_copy = deepcopy(g)
        loc_data = propagate_SI!(g_copy, r, beta)
        loc_result = loc_type == :pearson ?
                     loc_type_dict[loc_type](g_copy, loc_data.obs_data) :
                     loc_type_dict[loc_type](g_copy, loc_data.obs_data, beta)

        rank = calc_rank(loc_data, loc_result, nv(g_copy))
        prec = calc_prec(loc_data, loc_result)
        output[t] = "$rank,$prec"
    end

    open(path, "w") do io
        println(io, "N=$N,graph=$graph_type,method=$loc_type,r=$r,beta=$beta")
        println(io, "rank,precision")
        for row in output
            println(io, row)
        end
    end
end


function evaluate_modify_to_file(
    graph_type::Symbol,
    loc_type::Symbol,
    beta::Float64,
    r::Float64,
    N::Int,
    modify_type::Symbol,
    dj::Float64
)
    @assert haskey(graph_type_dict, graph_type)
    @assert haskey(loc_type_dict, loc_type)
    @assert haskey(modify_type_dict, modify_type)
    path = "data/mod_$(String(modify_type))_dj$(round(Int, dj*100))_$(String(graph_type))_$(loc_type)_r$(round(Int,r*100))_beta$(round(Int, beta*100)).csv"
    open(path, "w") do io
        println(io, "N=$N,modify=$modify_type,graph=$graph_type,dj=$dj,method=$loc_type,r=$r,beta=$beta")
        println(io, "rank,precision")
        for t in 1:N
            if t % 100 == 0
                println("Starting $t iteration")
            end
            g = graph_type_dict[graph_type]()
            loc_data::LocData = propagate_SI!(g, r, beta)
            sg, new_loc_data = modify_type_dict[modify_type](g, loc_data, dj)
            loc_result = loc_type == :pearson ?
                         loc_type_dict[loc_type](sg, new_loc_data.obs_data) :
                         loc_type_dict[loc_type](sg, new_loc_data.obs_data, beta)

            rank = calc_rank(new_loc_data, loc_result, nv(sg))
            prec = calc_prec(new_loc_data, loc_result)
            println(io, "$rank,$prec")
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


# LINK PREDICTION
function calc_prec_link_pred(graph_type::Symbol, pred_type::Symbol; num_folds=5)
    @assert haskey(graph_type_dict, graph_type)
    @assert haskey(score_type_dict, pred_type)

    g = graph_type_dict[graph_type]()
    edges_list = collect(edges(g)) # E
    shuffle!(edges_list)
    fold_size = div(length(edges_list), num_folds)
    println(fold_size)

    res = Vector(undef, num_folds)

    for i in 1:num_folds
        prec_list = []
        eval_pairs = Set(edge2pair(edge) for edge in edges_list[1+fold_size*(i-1):fold_size*i]) # E_V
        train_pairs = Set(edge2pair(edge) for edge in edges_list if edge2pair(edge) ∉ eval_pairs) # E_T
        train_graph = SimpleGraph(nv(g))
        for pair in collect(train_pairs)
            add_edge!(train_graph, pair...)
        end
        scores_heap = PriorityQueue(Base.Order.Reverse, score_type_dict[pred_type](train_graph))
        # return collect(values(score_type_dict[pred_type](train_graph)))
        # println(sort(collect(scores_heap), by=x->x[2], order=Base.Reverse)[1:200])
        TP = 0
        k = 0
        while k < 5*fold_size && !isempty(scores_heap) # |E_O| = |E_V|
            pair = dequeue!(scores_heap)
            if pair ∈ train_pairs
                continue
            end
            k += 1
            if pair ∈ eval_pairs
                TP += 1
            end
            append!(prec_list, TP / k)
        end
        res[i] = prec_list
        # append!(res, TP / fold_size)
    end

    return res
end

function calc_auc_link_pred(graph_type::Symbol, pred_type::Symbol; num_folds=5)::Float64
    @assert haskey(graph_type_dict, graph_type)
    @assert haskey(score_type_dict, pred_type)

    g = graph_type_dict[graph_type]()
    edges_list = collect(edges(g)) # E
    shuffle!(edges_list)
    fold_size = div(length(edges_list), num_folds)
    nonexistent_pairs = Set((u, v) for u in vertices(g), v in vertices(g) if (!has_edge(g, u, v) && u < v)) # U - E

    auc_list = Float64[]

    for i in 1:num_folds
        eval_edges = Set(edges_list[1+fold_size*(i-1):fold_size*i]) # E_V
        train_edges = [edge for edge in edges_list if edge ∉ eval_edges] # # E_T
        train_graph = SimpleGraph(nv(g))
        for edge in train_edges
            add_edge!(train_graph, edge)
        end
        scores = score_type_dict[pred_type](train_graph)

        # AUC = (n' + 0.5n'') / n - all samples, samples that result in higher score for correct prediction, sample that result in a draw
        auc_nominator = 0
        for _ in 1:10^6
            missing_score = get(scores, edge2pair(rand(eval_edges)), 0)
            nonexistent_score = get(scores, rand(nonexistent_pairs), 0)
            if missing_score > nonexistent_score
                auc_nominator += 1
            elseif missing_score == nonexistent_score
                auc_nominator += 0.5
            end
        end
        append!(auc_list, auc_nominator / 10^6)
    end

    return mean(auc_list)
end

function edge2pair(edge::Edge)
    return (
        min(src(edge), dst(edge)),
        max(src(edge), dst(edge))
    )
end
