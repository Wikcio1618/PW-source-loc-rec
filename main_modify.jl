include("Evaluation.jl")

using Base.Threads

println(nthreads())

N = 10^3
beta = 0.95
r = 0.3
loc_type = :gmla
graph_type = :email
modify_type = :hide

dj_list = 0:0.1:0.8
@threads for dj in dj_list
    evaluate_modify_to_file(graph_type, loc_type, beta, r, N, modify_type, dj)
end