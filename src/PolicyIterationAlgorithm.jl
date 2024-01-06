mutable struct PolicyIterationSolver <: Solver
    max_iterations::Int64 # max number of iterations
    belres::Float64 # the Bellman Residual
    verbose::Bool 
    include_Q::Bool
    init_util::Vector{Float64}
end

# Default constructor
function PolicyIterationSolver(;max_iterations::Int64 = 100, 
    belres::Float64 = 1e-2,
    verbose::Bool = false,
    include_Q::Bool = false,
    init_util::Vector{Float64}=Vector{Float64}(undef, 0))    

    return PolicyIterationSolver(max_iterations, belres, verbose, include_Q, init_util)
end

function solve(solver::PolicyIterationSolver, mdp::MDP; kwargs...)

    belres = solver.belres
    discount_factor = discount(mdp)
    # discount_factor = 1.0
    states_ = states(mdp)
    ns = length(states(mdp))
    na = length(actions(mdp))
    
    # TODO: change - isnt initial value matrix kinda useless?
    #              - better make initial policy matrix handling

    # initializing the value_matrix, Q-matrix and policy
    if !isempty(solver.init_util) # already initialized v_func
        @assert length(solver.init_util) == ns && "Error, utility dimension mismatch!"
        value_matrix = solver.init_util
    else
        value_matrix = zeros(ns)
    end

    qmat = zeros(ns, na)
    policy_matrix = ones(Int64,ns)
    # create policy matrix
    for state in states_
        # println(state, " ", actions(mdp, state)[1])
        if isterminal(mdp, state)
            policy_matrix[stateindex(mdp, state)] = 1
        else
            def_action = actions(mdp, state)[1]
            policy_matrix[stateindex(mdp, state)] = actionindex(mdp, def_action)
        end
    end


    # println(policy_matrix)
    converged = false
    
    iters = 0
    while !converged # || iters < 30
        iters += 1
        # println(iters)

        # POLICY evaluation
        # value_matrix = zeros(ns)

        # @TODO: actually wouldnt even need to have value_matrix as an arg
        value_matrix = policy_evaluation(mdp, zeros(ns), policy_matrix, discount = discount_factor, belres = belres)
        
        # POLICY IMPROVEMENT
        policy_matrix, converged = policy_improvement(mdp, value_matrix, policy_matrix, qmat, discount = discount_factor)
        
        # println("iteration: ", iters)
        
    end

    println("Policy iteration iterations: ", iters)

    if solver.include_Q
        return PolicyIterationPolicy(mdp, qmat, value_matrix, policy_matrix)
    else
        return PolicyIterationPolicy(mdp, utility=value_matrix, policy=policy_matrix, include_Q=false)
    end


end



function policy_evaluation(mdp::MDP, value_matrix::Vector, policy::Vector; discount::Float64 = 1.0, belres::Float64 = 1e-3)
    state_vec = ordered_states(mdp)
    
    old_value_matrix = deepcopy(value_matrix)
    # instead of old vs new, amke a matrix of two arrays that just switch - also good for history
    
    # delta = 0
    i = 0
    while true
        i += 1
        delta = 0
        
        for state in state_vec # get value for each state
            state_i = stateindex(mdp, state)
            
            old_v = value_matrix[state_i]

            action_id = policy[state_i]
            action_vec = actions(mdp)
            action = action_vec[action_id]

            probability_distr = transition(mdp, state, action)

            new_v = 0

            # V(s) = ∑T(s,π(s),s') * (r(s,π(s),s') + γ * V_old(s'))
            # Mausam_Kolobov - page 43
            for (next_state, prob) in weighted_iterator(probability_distr)
                if prob == 0
                    continue
                end

                r = reward(mdp, state, action, next_state)
                next_state_i = stateindex(mdp, next_state)

                new_v += prob * (r + discount * old_value_matrix[next_state_i])
                

            end # prob distribution loop
            
            value_matrix[state_i] = new_v
            delta = max(delta, abs(old_v - new_v))
            
        end # state loop

        old_value_matrix = deepcopy(value_matrix)
        if delta < belres
            break
        end

    end
    # println("EVALUTAION DONE at iteration: ", i)
    
    return value_matrix
    
end


function policy_improvement(mdp::MDP, value_matrix::Vector, policy_matrix::Vector, q_mat::Matrix; discount::Float64 = 1.0)
    
    state_vec = states(mdp)
    converged = true
    old_value_matrix = value_matrix

    for state in state_vec # Q-func across all states
        
        state_i = stateindex(mdp, state)
        action_vec = actions(mdp, state)
        
        old_action = policy_matrix[state_i]
        best_action = nothing
        max_v = -Inf

        # Mausam_Kolobov - page 44
        # Q-func for each state_action pair
        #     remember max-val_action for 
        #                        policy improvement
        #                        new value function
        for action in action_vec 
            action_i = actionindex(mdp, action)
            action_val = 0

            probability_distr = transition(mdp, state, action)
            
            # Q(s,a) = ∑T(s,a,s')[C(s, a, s') * γV(s')]
            for (next_state, prob) = weighted_iterator(probability_distr)
                if prob == 0
                    continue
                end
                r = reward(mdp, state, action, next_state)
                next_state_i = stateindex(mdp, next_state)
                action_val += prob * (r + discount * old_value_matrix[next_state_i])
            end

            if action_val > max_v
                max_v = action_val
                best_action = action_i
            end
            
            q_mat[state_i, action_i] = action_val

        end # action loop
        
        value_matrix[state_i] = max_v

        if isnothing(best_action)
            best_action = 1
        end

        # update policy
        # check for value of q_mat and old_value -> if value better, update policy
        if q_mat[state_i, old_action] < value_matrix[state_i]
            policy_matrix[state_i] = best_action
        end


        # check for convergence
        if old_action != policy_matrix[state_i]
            converged = false
        end

    end # state loop

    return policy_matrix, converged
    
end


function solve(::PolicyIterationSolver, ::POMDP)
    throw("""
           PolicyIterationError: `solve(::PolicyIterationSolver, ::POMDP)` is not supported,
          `PolicyIterationSolver` supports MDP models only, look at QMDP.jl for a POMDP solver that assumes full observability.
           """)
end