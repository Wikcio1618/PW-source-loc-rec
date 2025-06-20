using Distributed

N = 10^3
graph_type = :ba
graph_args = Dict(
    :V => 100,
    :n0 => 4,
    :k => 4
)

betas = [0.5, 0.8]
methods = [:pearson]
R = [0.05, 0.1, 0.15, 0.2, 0.25]

addprocs(length(R) * length(betas))
@everywhere include("../Evaluation.jl")
start = time()
for method in methods
    pmap(
        param -> evaluate_original_to_file(graph_type, method, param[2], param[1], N; graph_args=graph_args),
        Iterators.product(R, betas)
    )
end
println("program run for $(time()-start)")