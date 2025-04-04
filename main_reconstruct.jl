include("Evaluation.jl")

using Statistics
using Plots
using Base.Threads

println(nthreads())

N = 10^3
graph_type = :fb
beta = 0.95
r = 0.3
loc_type = :pearson
modify_type = :hide
reconstruct_type_list = [:ra, :rwr, :srw]
dj_list = [0.1, 0.2, 0.3, 0.4]
k_vec = 0:100:2000

@threads for dj in dj_list
    for rec_type in reconstruct_type_list
        evaluate_reconstruct_to_file(graph_type, loc_type, beta, r, N, modify_type, dj, rec_type, k_vec)
    end
end