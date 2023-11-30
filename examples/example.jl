using Revise

using PolicyIteration
using POMDPs
using POMDPTools
using POMDPModels
using DiscreteValueIteration

include("MausamKolobov.jl")

mauskol = MausamKolobov()
disc = discount(mauskol)


reward_grid = Dict{GWPos, Float64}();
reward_grid[GWPos(1, 1)] = 1
reward_grid[GWPos(6, 1)] = -1
# reward_grid[GWPos(5, 4)] = -1

mdp = SimpleGridWorld(
    size = (6,6),
    rewards = reward_grid
    )

PIsolver = PolicyIterationSolver(include_Q = true)
PIpolicy = PolicyIteration.solve(PIsolver, mauskol)

VIsolver = ValueIterationSolver(include_Q = true)
VIpolicy = DiscreteValueIteration.solve(VIsolver, mauskol)

VIpolicy.qmat

pol = [1, 3, 1, 2, 3, 3, 2, 3, 3, 1]

PIpolicy.policy == VIpolicy.policy

accuracy = 0
for i in 1:length(PIpolicy.policy)
    if PIpolicy.policy[i] != VIpolicy.policy[i]
        accuracy += 1
        println(i)
    end
end

println(accuracy, " / ", length(PIpolicy.policy))