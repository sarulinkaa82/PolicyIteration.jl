using DiscreteValueIteration
using PolicyIteration
using BenchmarkTools
using POMDPs
using POMDPTools
using POMDPModels

include("../examples/testing_domains.jl")

domain_size, grid_matrix = generate_test_domain("C:/repos/jukia_solvers/PolicyIteration.jl/examples/dataset-assignment2/data/maze-15-E.txt")
domain_size, grid_matrix = generate_random_domain((19,19), "tunnel")


mdp = CustomDomain(size = domain_size, grid = grid_matrix)

PI_solver = PolicyIterationSolver(include_Q = true)
@btime PI_policy = PolicyIteration.solve(PI_solver, mdp)

VI_solver = ValueIterationSolver(include_Q = true)
@btime VI_policy = DiscreteValueIteration.solve(VI_solver, mdp)

PI_times = [9.16, 7.14, 6.84, 6.86, 6.81]
VI_times = [1.74, 1.68, 1.69]

PI_mean = mean(PI_times)
VI_mean = mean(VI_times)
PI_std = std(PI_times)
VI_std = std(VI_times)