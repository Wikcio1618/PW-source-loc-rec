include("Localization.jl")
include("Propagation.jl")
include("GraphCreation.jl")
include("Modification.jl")
include("Reconstruction.jl")

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
        precs = Array{Float64, 2}(undef, (length(k_vec), N))
        for t in 1:N
            if t % round(Int, N / 10) == 0
                println("$reconstruct_type: Starting $t iteration")
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
    N::Int
)
    @assert haskey(graph_type_dict, graph_type)
    @assert haskey(loc_type_dict, loc_type)
    path = "data/loc_$(String(graph_type))_$(loc_type)_r$(round(Int,r*100))_beta$(round(Int, beta*100)).csv"
    open(path, "w") do io
        println(io, "N=$N,graph=$graph_type,method=$loc_type,r=$r,beta=$beta")
        println(io, "rank,precision")
        g = graph_type_dict[graph_type]()
        for t in 1:N
            if t % round(Int, N / 10) == 0
                println("$loc_type: Starting $t iteration")
            end
            loc_data::LocData = propagate_SI!(g, r, beta)
            loc_result = loc_type == :pearson ?
                         loc_type_dict[loc_type](g, loc_data.obs_data) :
                         loc_type_dict[loc_type](g, loc_data.obs_data, beta)
            rank = calc_rank(loc_data, loc_result, nv(g))
            prec = calc_prec(loc_data, loc_result)
            println(io, "$rank,$prec")
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