include("Evaluation.jl")

using Statistics
using Plots
using Base.Threads
using Graphs

println(nthreads())

N = 5*10^2
graph_type = :email
E = ne(graph_type_dict[graph_type]())
beta = 0.95
r = 0.3
loc_type = :pearson
modify_type = :hide
reconstruct_type_list = [:ra, :srw]
dj_list = [0.1, 0.2, 0.3, 0.4]
# k_vec = 0:250:1000

@threads for dj in dj_list
    k_vec = [round(Int, n * dj * E) for n in [0, 0.5, 1, 2]]
    for rec_type in reconstruct_type_list
        evaluate_reconstruct_to_file(graph_type, loc_type, beta, r, N, modify_type, dj, rec_type, k_vec)
    end
end