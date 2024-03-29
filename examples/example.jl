using Revise
# using ColorSchemes
using PolicyIteration
using POMDPs
using POMDPTools
using POMDPModels
# using DiscreteValueIteration
# using BenchmarkTools

include("../test/MausamKolobov.jl")

VIsolver = ValueIterationSolver()
PIsolver = PolicyIterationSolver()

mauskol = MausamKolobov()
PIpolicy = PolicyIteration.solve(PIsolver, mauskol)
VIpolicy = DiscreteValueIteration.solve(VIsolver, mauskol)

switch = mod(2, 2) + 1

include("testing_domains.jl")
# sizee, mat = generate_random_domain((7, 7), "gap")

sizee, mat = generate_test_domain("C:/repos/jukia_solvers/PolicyIteration.jl/examples/dataset-assignment2/data/maze-7-A1.txt")
mdp = CustomDomain(size = sizee, grid = mat)
PIsolver = PolicyIterationSolver(include_Q = true)
PIpolicy = PolicyIteration.solve(PIsolver, mdp)

sss = states(mdp)
println(sss)
stat = sss[18]
id = stateindex(mdp, stat)

# VIsolver = ValueIterationSolver(include_Q = true)
# VIpolicy = DiscreteValueIteration.solve(VIsolver, mdp)

# VIpolicy.qmat
# PIpolicy.qmat

# VIpolicy.util
# PIpolicy.util
# PIpolicy.policy == VIpolicy.policy

# accuracy = 0
# for i in 1:length(PIpolicy.policy)
#     if PIpolicy.policy[i] != VIpolicy.policy[i]
#         accuracy += 1
#         println(i)
#     end
# end

# println(accuracy, " / ", length(PIpolicy.policy))

sizee = 5
reward_grid = Dict{GWPos, Float64}();
reward_grid[GWPos(1, 1)] = 1
reward_grid[GWPos(sizee, 1)] = -1
# reward_grid[GWPos(5, 4)] = -1

mdp = SimpleGridWorld(
    size = (sizee,sizee),
    rewards = reward_grid
    )

function create_random_grid()
    grid_size = 100;
    reward_grid = Dict{GWPos, Float64}();

    for i in 1:grid_size
        for j in 1:grid_size
            pos = GWPos(i, j)
            reward = rand(-10.0:10.0)
            reward_grid[pos] = reward
        end
    end
    
    mdp = SimpleGridWorld(
        size = (100,100),
        rewards = reward_grid,
    );

    return mdp
end