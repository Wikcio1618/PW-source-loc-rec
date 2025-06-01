using Distributed
using Graphs
include("../GraphCreation.jl")

N = 10^3
graph_type = :email
E = ne(graph_type_dict[graph_type]())
beta = 0.95
r = 0.1
modify_type = :hide

methods = [:pearson, :gmla, :lptva]
reconstruct_type_list = [:ra, :srw]
dj_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


addprocs(length(dj_list) * length(methods) * length(reconstruct_type_list))
@everywhere include("../Evaluation.jl")

start = time()
pmap(
	params -> begin
		dj, rec_type, method = params
		evaluate_reconstruct_to_file(
			graph_type,
			method,
			beta,
			r,
			N,
			modify_type,
			dj,
			rec_type,
			[round(Int, n * dj * E) for n in [0, 0.5, 1, 2]])
	end,
	Iterators.product(dj_list, reconstruct_type_list, methods),
)

println("program run for $(time()-start)")
