include("Evaluation.jl")

# using Plots

# graph_type = :email
# pred_type = :ml
# res = calc_prec_link_pred(graph_type, pred_type; num_folds=5)
# print(res)
# println(mean(map(x->x[553 ], res)))
# plot(res)
# plot(res)

# Plot histogram
# histogram(score_values; bins=500, yscale=:log10)

using DataFrames
using CSV

graph_type_list = [:inf, :usa, :cel, :cal, :email, :fb]
pred_type_list = [:ra, :srw]

ppl_df = DataFrame(Graph=String[], Method=String[], PPL=Float64[])
auc_df = DataFrame(Graph=String[], Method=String[], AUC=Float64[])

# Compute metrics for each combination
for graph_type in graph_type_list
    for pred_type in pred_type_list
        ppl_value = calc_prec_link_pred(graph_type, pred_type)
        auc_value = calc_auc_link_pred(graph_type, pred_type)

        # Append results to DataFrames
        push!(ppl_df, (String(graph_type), String(pred_type), ppl_value))
        push!(auc_df, (String(graph_type), String(pred_type), auc_value))
    end
end

CSV.write("data/ppl_results.csv", ppl_df)
CSV.write("data/auc_results.csv", auc_df)






