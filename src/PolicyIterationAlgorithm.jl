# The solver type
"""
    PolicyIterationSolver <: Solver

The solver type. Contains the following parameters that can be passed as keyword arguments to the constructor

    - max_iterations::Int64, the maximum number of iterations value iteration runs for (default 100)
    - belres::Float64, the Bellman residual (default 1e-2)
    - include_Q::Bool, if set to true, the solver outputs the Q values in addition to the utility and the policy (default true)
    - init_util::Vector{Float64}, provides a custom initialization of the utility vector. (initializes utility to 0 by default)
"""

mutable struct PolicyIterationSolver <: Solver
    belres::Float64 # the Bellman Residual
    verbose::Bool 
    include_Q::Bool
end

# Default constructor
function PolicyIterationSolver(;
    belres::Float64 = 1e-2,
    verbose::Bool = false,
    include_Q::Bool = false)    

    return PolicyIterationSolver(belres, verbose, include_Q)
end

function solve(solver::PolicyIterationSolver, mdp::MDP; kwargs...)

    verbose = solver.verbose
    belres = solver.belres
    discount_factor = discount(mdp)
    # discount_factor = 1.0
    states_ = states(mdp)
    ns = length(states(mdp))
    na = length(actions(mdp))
    
    # TODO: make initial policy matrix handling

    # initializing the value_matrix, Q-matrix and policy
    
    value_matrix = zeros(ns)

    qmat = zeros(ns, na)
    policy_matrix = ones(Int64,ns)
    # create policy matrix
    for state in states_
        if isterminal(mdp, state)
            policy_matrix[stateindex(mdp, state)] = 1
        else
            def_action = actions(mdp, state)[1]
            policy_matrix[stateindex(mdp, state)] = actionindex(mdp, def_action)
        end
    end


    converged = false
    
    iters = 0
    while !converged # || iters < 30
        iters += 1
        # println(iters)

        # POLICY evaluation
        value_matrix = policy_evaluation(mdp, zeros(ns), policy_matrix, discount = discount_factor, belres = belres, verbose = verbose)
        
        # POLICY IMPROVEMENT
        policy_matrix, converged = policy_improvement(mdp, value_matrix, policy_matrix, qmat, discount = discount_factor)
        
    end

    if verbose
        println("Policy iteration iterations: ", iters)
    end

    if solver.include_Q
        return PolicyIterationPolicy(mdp, qmat, value_matrix, policy_matrix)
    else
        return PolicyIterationPolicy(mdp, utility=value_matrix, policy=policy_matrix, include_Q=false)
    end
end



function policy_evaluation(mdp::MDP, value_matrix::Vector, policy::Vector; discount::Float64 = 1.0, belres::Float64 = 1e-3, verbose = false)
    state_vec = ordered_states(mdp)

    val_len = length(value_matrix)
    value_matrices = fill(0.0, 2, val_len)
    switch = 1
    value_matrices[switch, :] = value_matrix
    i = 0
    while true
        i += 1
        delta = 0
        
        for (state_i, state) in enumerate(state_vec) # get value for each state
            
            # state_i = stateindex(mdp, state)
            old_v = value_matrices[switch, state_i]
            action_id = policy[state_i]
            action_vec = actions(mdp)
            action = action_vec[action_id]

            probability_distr = transition(mdp, state, action)

            # new_v = 0
            new_v_ = 0

            # V(s) = ∑T(s,π(s),s') * (r(s,π(s),s') + γ * V_old(s'))
            # Mausam_Kolobov - page 43
            for (next_state, prob) in weighted_iterator(probability_distr)
                if prob == 0
                    continue
                end

                r = reward(mdp, state, action, next_state)
                next_state_i = stateindex(mdp, next_state)

                new_v_ += prob * (r + discount * value_matrices[switch, next_state_i])
                
            end # prob distribution loop
            
            value_matrix[state_i] = new_v_

            delta = max(delta, abs(old_v - new_v_))
            
        end # state loop

        switch = mod(i, 2) + 1
        value_matrices[switch, :] = value_matrix
        if delta < belres
            break
        end

    end

    if verbose
        println("EVALUTAION DONE at iteration: ", i)
    end

    return value_matrices[switch, :]
    
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