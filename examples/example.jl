using Revise
using PolicyIteration
using POMDPs
using POMDPTools
using POMDPModels
using DiscreteValueIteration

include("MausamKolobov.jl")

mauskol = MausamKolobov()
disc = discount(mauskol)


sizee = 7
reward_grid = Dict{GWPos, Float64}();
reward_grid[GWPos(1, 1)] = 1
reward_grid[GWPos(sizee, 1)] = -1
# reward_grid[GWPos(5, 4)] = -1

mdp = SimpleGridWorld(
    size = (sizee,sizee),
    rewards = reward_grid
    )

PIsolver = PolicyIterationSolver(include_Q = true)
PIpolicy = PolicyIteration.solve(PIsolver, mauskol)

VIsolver = ValueIterationSolver(include_Q = true)
VIpolicy = DiscreteValueIteration.solve(VIsolver, mauskol)

VIpolicy.qmat
PIpolicy.qmat

VIpolicy.util
PIpolicy.util
PIpolicy.policy == VIpolicy.policy

accuracy = 0
for i in 1:length(PIpolicy.policy)
    if PIpolicy.policy[i] != VIpolicy.policy[i]
        accuracy += 1
        println(i)
    end
end

println(accuracy, " / ", length(PIpolicy.policy))