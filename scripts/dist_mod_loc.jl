using Distributed
using Graphs

N = 10^3
graph_type = :email
graph_args = Dict()
# graph_args = Dict(
#     :V => 100,
#     :n0 => 4,
#     :k => 4
# )
beta = 0.95
r = 0.1
modify_type = :hide

methods = [:pearson]
dj_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

addprocs(length(dj_list) * length(methods) * length(reconstruct_type_list))
@everywhere include("../Evaluation.jl")

start = time()
pmap(
    params -> begin
        dj, method = params
        evaluate_modify_to_file(
            graph_type,
            method,
            beta,
            r,
            N,
            modify_type,
            dj,
            ; graph_args=graph_args
        )
    end,
    Iterators.product(dj_list, methods),
)
println("program run for $(time()-start)")