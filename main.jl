using Revise
includet("MainModule.jl")
using .MainModule

path = "data/test.csv"
N = 100
beta = 0.8
r = 0.2
graph_type = :ba
loc_type = :gmla 

graph_args = Dict(
    :V => 1000,
    :n0 => 4,
    :k => 4,
    :r => r
)

precision_to_file(path, graph_type, loc_type, beta, N, graph_args)
sum = open(path, "r") do io
    readline(io)
    readline(io)
    sum = 0
    for _ in 1:N
        line = readline(io)
        sum += parse(Float64, line[(findfirst(',', line)+1):end])
    end
    sum
end

sum / N

