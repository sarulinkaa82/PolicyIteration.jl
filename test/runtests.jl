using Test
using PolicyIteration
using POMDPs
using POMDPTools
using POMDPModels
include("MausamKolobov.jl")


@testset "PolicyIteration.jl" begin
    # Write your tests here.
    
    VIsolver = ValueIterationSolver()
    PIsolver = PolicyIterationSolver()
    
    for i in 3:6
        reward_grid = Dict{GWPos, Float64}();
        reward_grid[GWPos(1, 1)] = 1
        reward_grid[GWPos(i, 1)] = -1

        mdp = SimpleGridWorld(
        size = (i,i),
        rewards = reward_grid
        )

        
        PIpolicy = PolicyIteration.solve(PIsolver, mdp)
        VIpolicy = DiscreteValueIteration.solve(VIsolver, mdp)


        @test PIpolicy.policy == VIpolicy.policy
    end

    mauskol = MausamKolobov()
    PIpolicy = PolicyIteration.solve(PIsolver, mauskol)
    VIpolicy = DiscreteValueIteration.solve(VIsolver, mauskol)

    @test PIpolicy.policy == VIpolicy.policy

end
