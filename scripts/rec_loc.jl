using Base.Threads
using Graphs
include("../GraphCreation.jl")
include("../Evaluation.jl")

println("Number of threads: $(nthreads())")

N = 10^2
graph_type = :ba
graph_args = Dict()
graph_args = Dict(
    :V => 200,
    :n0 => 4,
    :k => 4
)
E = ne(graph_type_dict[graph_type](; graph_args...))
beta = 0.95
r = 0.1
modify_type = :hide

method = :pearson
rec_type = :bp
# dj_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
dj_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


start = time()

@threads for dj in dj_list
evaluate_reconstruct_to_file(
    graph_type,
    method,
    beta,
    r,
    N,
    modify_type,
    dj,
    rec_type,
    dj == 0.0 ? [0] : [round(Int, n * dj * E) for n in [0, 1, 2]],
    ;graph_args = graph_args
)
end
println("program run for $(time()-start)")