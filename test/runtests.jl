using PolicyIteration
using Test
using POMDPs
using POMDPTools
using POMDPModels

@testset "PolicyIteration.jl" begin
    # Write your tests here.
    
    for i in 3:6
        reward_grid = Dict{GWPos, Float64}();
        reward_grid[GWPos(1, 1)] = 1
        reward_grid[GWPos(i, 1)] = -1

        mdp = SimpleGridWorld(
        size = (i,i),
        rewards = reward_grid
        )

        VIsolver = ValueIterationSolver()
        PIsolver = PolicyIterationSolver()
        PIpolicy = PolicyIteration.solve(PIsolver, mdp)
        VIpolicy = DiscreteValueIteration.solve(VIsolver, mdp)

        @test PIpolicy.policy == VIpolicy.policy
    end


end
