include("Evaluation.jl")

using Statistics
using Plots
using Base.Threads

println(nthreads())

N = 10^2
graph_type = :fb
beta = 0.95
r = 0.3
loc_type = :gmla
modify_type = :hide
reconstruct_type_list = [:ra]
dj = 0.2
k_vec = 0:100:2000

for rec_type in reconstruct_type_list
    evaluate_reconstruct_to_file(graph_type, loc_type, beta, r, N, modify_type, dj, rec_type, k_vec)
end