include("Localization.jl")
include("Propagation.jl")
include("GraphCreation.jl")
include("StructModule.jl")
include("Modification.jl")
include("Reconstruction.jl")

using Graphs
using Random
using DataStructures

function graph_reconstruct_precision_to_file(
    path::String,
    graph::Union{Symbol,SimpleGraph},
    loc_type::Symbol,
    beta::Float64,
    r::Float64,
    N::Int,
    modify_type::Symbol,
    dj::Float64,
    reconstruct_type::Symbol,
    k_vec,
    graph_args::Dict=Dict()
)
    if typeof(graph) == Symbol
        @assert haskey(graph_type_dict, graph_type)
        graph = graph_type_dict[graph](; graph_args...)
    end
    @assert haskey(loc_type_dict, loc_type)
    @assert haskey(modify_type_dict, modify_type)
    @assert haskey(score_type_dict, reconstruct_type)

    open(path, "w") do io
        println(io, "N=$N,graph=$graph_type,method=$loc_type,r=$r,beta=$beta,graph_args=$graph_args,modify_type=$modify_type,dj=$dj,reconstruct_type=$reconstruct_type")
        println(io, "k,avg_precision,avg_dj")
        g = graph_type_dict[graph_type](; graph_args...)
        prec_cumulative = zeros(length(k_vec))
        # dj_cumulative = zeros(length(k_vec))
        for _ in 1:N
            loc_data::LocData = propagate_SI!(g, r, beta)
            sg, new_loc_data = modify_type_dict[modify_type](g, loc_data, dj)
            heap = PriorityQueue(score_type_dict[reconstruct_type](sg))
            for i in eachindex(k_vec)
                new_g = reconstruct_top_k!(
                    sg,
                    heap,
                    i == 1 ? k_vec[i] : k_vec[i] - k_vec[i-1];
                    type=modify_type == :hide ? :add : :hide
                )
                loc_result =
                    loc_type == :pearson ?
                    loc_type_dict[loc_type](new_g, new_loc_data.obs_data) :
                    loc_type_dict[loc_type](new_g, new_loc_data.obs_data, beta)

                prec_cumulative[i] += calc_prec(new_loc_data, loc_result)
                # dj_cumulative[i] += calc_jaccard(g, new_g)
            end
        end
        for (i, k) in enumerate(k_vec)
            println(io, "$k,$(prec_cumulative[i]/N)")
        end
    end
end

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
    @assert haskey(score_type_dict, reconstruct_type)

    open(path, "w") do io
        println(io, "N=$N,graph=$graph_type,method=$loc_type,r=$r,beta=$beta,graph_args=$graph_args,modify_type=$modify_type,dj=$dj,reconstruct_type=$reconstruct_type,hide_thresh=$hide_thresh,add_thresh=$add_thresh")
        println(io, "rank,precision,dj")
        for _ in 1:max(1, (round(Int, N / 1)))
            g = graph_type_dict[graph_type](; graph_args...)
            for _ in 1:1
                loc_data::LocData = propagate_SI!(g, r, beta)
                # for each modification
                for _ in 1:1
                    new_g = modify_type_dict[modify_type](g, dj; inplace=true)
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

function calc_jaccard(g1::SimpleGraph, g2::SimpleGraph, vmap::Vector{Int})::Float64
    edges1 = Set([(min(e.src, e.dst), max(e.src, e.dst)) for e in edges(g1)])
    edges2 = Set([(min(vmap[e.src], vmap[e.dst]), max(vmap[e.src], vmap[e.dst])) for e in edges(g2)])

    intersection_size = length(intersect(edges1, edges2))
    union_size = length(union(edges1, edges2))

    return union_size == 0 ? 1.0 : 1 - intersection_size / union_size
end

# LINK PREDICTION
function calc_prec_link_pred(graph_type::Symbol, pred_type::Symbol; graph_args::Dict=Dict())
    @assert haskey(graph_type_dict, graph_type)
    @assert haskey(score_type_dict, pred_type)

    g = graph_type_dict[graph_type](; graph_args...)
    edges = collect(edges(g))
    total_edges = length(edges)
    fold_size = div(total_edges, 5)

    precisions = Float64[]

    # Shuffle edges for random fold assignment
    shuffled_edges = shuffle(edges)

    for fold in 1:5
        # Determine test and training edges
        test_start = (fold - 1) * fold_size + 1
        test_end = min(fold * fold_size, total_edges)
        test_edges = shuffled_edges[test_start:test_end]
        train_edges = setdiff(shuffled_edges, test_edges)

        # Create training graph
        train_g = SimpleGraph(nv(g))
        for (u, v) in train_edges
            add_edge!(train_g, u, v)
        end

        # Generate scores using the selected prediction method
        min_heap, max_heap = score_type_dict[pred_type](train_g)

        # Sort predictions by descending score
        predicted_links = collect(keys(max_heap))
        sort!(predicted_links, by=x -> max_heap[x], rev=true)

        # Compute precision: fraction of correctly predicted links in top-k
        test_set = Set(test_edges)
        top_k_predictions = predicted_links[1:min(k, length(predicted_links))]
        correct_predictions = count(x -> x in test_set, top_k_predictions)
        push!(precisions, correct_predictions / k)
    end

    return mean(precisions)
end

function calc_auc_5fold(graph_type::Symbol, pred_type::Symbol; k_folds=5)
    @assert haskey(graph_type_dict, graph_type)
    @assert haskey(score_type_dict, pred_type)

    g = graph_type_dict[graph_type]()
    edges_list = collect(edges(g))
    n_edges = length(edges_list)

    if n_edges < k_folds
        error("Not enough edges for k-fold cross-validation")
    end

    shuffled_edges = shuffle(edges_list)
    fold_size = div(n_edges, k_folds)
    aucs = []

    for i in 1:k_folds
        test_idx = ((i-1)*fold_size+1):min(i * fold_size, n_edges)
        test_edges = shuffled_edges[test_idx]
        train_edges = setdiff(shuffled_edges, test_edges)

        g_train = SimpleGraph(nv(g))
        for e in train_edges
            add_edge!(g_train, src(e), dst(e))
        end

        scores = score_type_dict[pred_type](g_train)

        # Select random negative samples (equal in size to probe set)
        all_possible_edges = Set(Edge(u, v) for u in vertices(g_train), v in vertices(g_train) if u < v)
        negative_samples = setdiff(all_possible_edges, Set(edges(g_train)))
        sampled_negatives = Random.shuffle(collect(negative_samples))[1:length(test_edges)]

        e2t = (e) -> (min(src(e), dst(e)), max(src(e), dst(e)))
        pos_scores = [scores[e2t(e)] for e in test_edges if haskey(scores, e2t(e))]
        neg_scores = [scores[e2t(e)] for e in sampled_negatives if haskey(scores, e2t(e))]

        println("Fold $i: |Pos| = $(length(pos_scores)), |Neg| = $(length(neg_scores))")

        n_pos = length(pos_scores)
        n_neg = length(neg_scores)
        if n_pos == 0 || n_neg == 0
            push!(aucs, 0.5)  # Neutral value if AUC is undefined
            continue
        end

        # Compute AUC using independent random sampling
        comparisons = [(rand(pos_scores), rand(neg_scores)) for _ in 1:(n_pos*n_neg)]
        U = sum(p > n ? 1 : (p == n ? 0.5 : 0) for (p, n) in comparisons)

        auc_value = U / (n_pos * n_neg)
        println("Fold $i AUC: $auc_value")
        push!(aucs, auc_value)
    end

    final_auc = mean(aucs)
    println("Final AUC: $final_auc")
    return final_auc
end


function calc_auc(graph_type::Symbol, pred_type::Symbol)
    @assert haskey(graph_type_dict, graph_type)
    @assert haskey(score_type_dict, pred_type)

    g = graph_type_dict[graph_type]()
    scores = score_type_dict[pred_type](g)

    positive_samples = Set(edges(g))
    all_node_pairs = Set(Edge(u, v) for u in vertices(g), v in vertices(g) if u < v)
    negative_samples = setdiff(all_node_pairs, positive_samples)

    e2t = (e) -> (min(src(e), dst(e)), max(src(e), dst(e)))
    pos_scores = [scores[e2t(e)] for e in collect(positive_samples) if haskey(scores, e2t(e))]
    neg_scores = [scores[e2t(e)] for e in collect(negative_samples) if haskey(scores, e2t(e))]

    n_pos = length(pos_scores)
    n_neg = length(neg_scores)
    if n_pos == 0 || n_neg == 0
        return 0.5  # Undefined AUC, return 0.5 as a neutral value
    end

    U = 0
    for pos_score in pos_scores
        for neg_score in neg_scores
            if pos_score > neg_score
                U += 1
            elseif pos_score == neg_score
                U += 0.5
            end
        end
    end

    auc = U / (n_pos * n_neg)
    return auc
end