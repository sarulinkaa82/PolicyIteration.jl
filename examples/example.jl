using Revise
using PolicyIteration
using POMDPs
using POMDPTools
using POMDPModels
using DiscreteValueIteration


reward_grid = Dict{GWPos, Float64}();
reward_grid[GWPos(1, 1)] = 1
# reward_grid[GWPos(8, 1)] = -1
# reward_grid[GWPos(5, 4)] = -1

mdp = SimpleGridWorld(
    size = (3,3),
    rewards = reward_grid
    )

PIsolver = PolicyIterationSolver()
PIpolicy =  PolicyIteration.solve(PIsolver, mdp)

VIsolver = ValueIterationSolver()
VIpolicy = DiscreteValueIteration.solve(VIsolver, mdp)

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