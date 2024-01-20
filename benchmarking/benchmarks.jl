using DiscreteValueIteration
using BenchmarkTools
using PolicyIteration
using POMDPs
using POMDPTools
using POMDPModels

include("../test/MausamKolobov.jl")
mdp = MausamKolobov()
PI_solver = PolicyIterationSolver(include_Q = true, verbose = true)
PI_policy = PolicyIteration.solve(PI_solver, mdp)

include("../examples/testing_domains.jl")
domain_size, grid_matrix = generate_test_domain("C:/repos/jukia_solvers/PolicyIteration.jl/examples/dataset-assignment2/data/maze-7-A1.txt")
mdp = CustomDomain(size = domain_size, grid = grid_matrix)

domain_size, grid_matrix = generate_random_domain((15,15), "tunnel")



@btime PI_policy = PolicyIteration.solve(PI_solver, mdp)

VI_solver = ValueIterationSolver(include_Q = true)
@btime VI_policy = DiscreteValueIteration.solve(VI_solver, mdp)

PI_times = []
VI_times = []

PI_mean = mean(PI_times)
VI_mean = mean(VI_times)
PI_std = std(PI_times)
VI_std = std(VI_times)