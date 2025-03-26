include("Evaluation.jl")

using Statistics
using Plots

N = 10
graph_type = :fb
beta = 0.8
r = 0.2
loc_type = :pearson
modify_type = :hide
reconstruct_type = :ra
dj = 0.2
k_vec = 0:10:200



# for hide_thresh in H
#     for add_thresh in A
#         println("doing $hide_thresh, $add_thresh")
#         path = "data/rec_$(String(graph_type))_$(hide_thresh)_$(add_thresh)_$(String(loc_type))_$(String(modify_type))_$(String(reconstruct_type)).csv"
#         evaluate_reconstruct_to_file(path, graph_type, loc_type, beta, r, N, graph_args, modify_type, dj, reconstruct_type, hide_thresh, add_thresh)
#     end
# end
path = "data/rec_$(String(graph_type))_k$(k_vec[begin])_$(k_vec[end])_$(String(loc_type))_$(String(modify_type))_$(String(reconstruct_type)).csv"
graph_reconstruct_precision_to_file(path, graph_type, loc_type, beta, r, N, modify_type, dj, reconstruct_type, k_vec)
