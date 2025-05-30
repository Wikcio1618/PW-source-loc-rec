include("Evaluation.jl")

# using Statistics
using Plots
using Base.Threads

#using DataFrames
#using CSV

N = 10^4
graph_type = :email
println("Num of threads: $(nthreads())")

betas = [0.2]
methods = [:lptva]
R = [0.05, 0.1, 0.15, 0.2, 0.25]

for beta in betas
    for method in methods
        for r in R
            evaluate_original_to_file(graph_type, method, beta, r, N)
        end
    end
end


# graph_type_list = [:cel, :inf, :usa]
# pred_type_list = [:srw]

# graph_type = :inf
# pred_type = :merw
# res = calc_prec_link_pred(graph_type, pred_type; num_folds=5)
# # print(res)
# println(mean(map(x->x[553 ], res)))
# # plot(res)
# plot(res)

# Plot histogram
# histogram(score_values; bins=500, yscale=:log10)

# ppl_df = DataFrame(Graph=String[], Method=String[], PPL=Float64[])
# auc_df = DataFrame(Graph=String[], Method=String[], AUC=Float64[])

# # Compute metrics for each combination
# for graph_type in graph_type_list
#     for pred_type in pred_type_list
#         ppl_value = calc_prec_link_pred(graph_type, pred_type)
#         # auc_value = calc_auc_link_p red(graph_type, pred_type)

#         # Append results to DataFrames
#         push!(ppl_df, (String(graph_type), String(pred_type), ppl_value))
#         # push!(auc_df, (String(graph_type), String(pred_type), auc_value))
#     end
# end

# CSV.write("data/ppl_results.csv", ppl_df)
# CSV.write("data/auc_results.csv", auc_df)






