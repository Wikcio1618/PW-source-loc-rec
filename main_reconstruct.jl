include("MainModule.jl")
using .MainModule

using Statistics
using Plots

N = 2000
graph_type = :ba
graph_args = Dict(
    :V => 1000,
    :n0 => 4,
    :k => 4
)
beta = 0.8
r = 0.2
loc_type = :pearson
modify_type = :add
reconstruct_type = :cn
dj = 0.2
A = [0]
H = 0:100:400



# for hide_thresh in H
#     for add_thresh in A
#         println("doing $hide_thresh, $add_thresh")
#         path = "data/rec_$(String(graph_type))_$(hide_thresh)_$(add_thresh)_$(String(loc_type))_$(String(modify_type))_$(String(reconstruct_type)).csv"
#         evaluate_reconstruct_to_file(path, graph_type, loc_type, beta, r, N, graph_args, modify_type, dj, reconstruct_type, hide_thresh, add_thresh)
#     end
# end

calc_auc(:inf, :cn)
