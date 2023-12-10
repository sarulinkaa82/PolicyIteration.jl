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
    while !converged
        iters += 1

        # POLICY evaluation
        # value_matrix = zeros(ns)

        # @TODO: actually wouldnt even need to have value_matrix as an arg
        value_matrix = policy_evaluation2(mdp, zeros(ns), policy_matrix, discount = discount_factor, belres = belres)
        
        # POLICY IMPROVEMENT
        policy_matrix, converged = policy_improvement2(mdp, value_matrix, policy_matrix, qmat, discount = discount_factor)
        
        
    end

    println("Policy iteration terations: ", iters)

    if solver.include_Q
        return PolicyIterationPolicy(mdp, qmat, value_matrix, policy_matrix)
    else
        return PolicyIterationPolicy(mdp, utility=value_matrix, policy=policy_matrix, include_Q=false)
    end


end


function solve2(solver::PolicyIterationSolver, mdp::MDP; kwargs...)

    belres = solver.belres
    # discount_factor = discount(mdp)
    discount_factor = 1.0
    states_ = states(mdp)
    ns = length(states(mdp))
    na = length(actions(mdp))
    
    # println(states_)

    # initializing the value_matrix, Q-matrix and policy
    if !isempty(solver.init_util) # already initialized v_func
        @assert length(solver.init_util) == ns && "Error, utility dimension mismatch!"
        value_matrix = solver.init_util
    else
        value_matrix = zeros(ns)
    end

    if solver.include_Q
        qmat = zeros(ns, na)
    end

    policy_matrix = ones(Int64,ns)
    
    
    
    converged = false
    
    iters = 0
    while !converged
        iters += 1

        # POLICY evaluation
        value_matrix = policy_evaluation(mdp, value_matrix, discount_factor, belres)

        # println("value_matrix")
        # println(value_matrix)

        
        # POLICY IMPROVEMENT
        if !solver.include_Q
            policy_matrix, converged = policy_improvement(mdp, value_matrix, policy_matrix, discount_factor)
        else
            policy_matrix, qmat, converged = policy_improvement(mdp, value_matrix, policy_matrix, discount_factor, solver.include_Q, qmat)
        end
        
    end

    println("iterations: ", iters)

    if solver.include_Q
        return PolicyIterationPolicy(mdp, qmat, value_matrix, policy_matrix)
    else
        return PolicyIterationPolicy(mdp, utility=value_matrix, policy=policy_matrix, include_Q=false)
    end


end


function policy_evaluation2(mdp::MDP, value_matrix::Vector, policy::Vector; discount::Float64 = 1.0, belres::Float64 = 1e-3)
    state_vec = ordered_states(mdp)
    
    old_value_matrix = deepcopy(value_matrix)
    
    i = 0
    while true
        i += 1
        delta = 0

        # println(value_matrix)
        # println(old_value_matrix)
        
        for state in state_vec # get value for each state
            state_i = stateindex(mdp, state)
            
            old_v = value_matrix[state_i]

            action_id = policy[state_i]
            action_vec = actions(mdp)
            action = action_vec[action_id]

            probability_distr = transition(mdp, state, action)

            new_v = 0
            # println("state ", state, " action ", action)
            # println(probability_distr)

            # V(s) = ∑T(s,π(s),s') * (r(s,π(s),s') + γ * V_old(s'))
            # Mausam_Kolobov - page 43
            for (next_state, prob) in weighted_iterator(probability_distr)
                if prob == 0
                    continue
                end

                r = reward(mdp, state, action, next_state)
                next_state_i = stateindex(mdp, next_state)

                new_v += prob * (r + discount * old_value_matrix[next_state_i])
                
                # if state == "s4"
                #     println(state)
                #     println(prob, r, old_value_matrix[next_state_i])
                #     println(next_state, "  ", next_state_i, " ", new_v)
                # end

            end # prob distribution loop
            
            value_matrix[state_i] = new_v
            delta = max(delta, abs(old_v - new_v))
            # println("old v mat: ", old_value_matrix)
            # println(new_v, " ", old_v)
            
        end # state loop


        # println("delta: ", delta)

        old_value_matrix = deepcopy(value_matrix)
        if delta < belres
            break
        end

    end
    println("EVALUTAION DONE at iteration: ", i)
    
    return value_matrix
    
end


function policy_evaluation(mdp::MDP, value_matrix::Vector,discount_factor::Float64 = 1.0, belres::Float64 = 1e-3)
    state_vec = ordered_states(mdp)
    delta = 0
    
    while delta < belres
        old_value_matrix = value_matrix
        
        for state in state_vec # iteration across the value function
            state_i = stateindex(mdp, state)
            action_vec = actions(mdp, state)
            
            old_v = value_matrix[state_i]
            max_v = -Inf
            
            for action in action_vec # compute value for each action
                new_v = 0
                probability_distr = transition(mdp, state, action) # transition distribution over neighbors
                
                # V(s) = sum(T(s'|s,a) [r + V(s')]) # sum across all possible states you can end up in
                for (next_state, prob) in weighted_iterator(probability_distr)
                    if prob == 0 
                        continue
                    end

                    r = reward(mdp, state, action, next_state)
                    next_state_i = stateindex(mdp, next_state)
                    new_v += prob * (r + discount_factor * old_value_matrix[next_state_i])
                end
                
                if new_v > max_v # find the action with the highest value
                    max_v = new_v
                end
            end
            # set new value into the matrix and change delta
            value_matrix[state_i] = max_v
            delta = max(delta, abs(old_v - max_v))
            
        end # iter across val matrix
    end # residual, policy evaluation
    
    return value_matrix
    
end


function policy_improvement2(mdp::MDP, value_matrix::Vector, policy_matrix::Vector, q_mat::Matrix; discount::Float64 = 1.0)
    
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


function policy_improvement(mdp::MDP, value_matrix::Vector, policy_matrix::Vector, discount_factor::Float64 = 1)

    state_vec = ordered_states(mdp)
    converged = true

    for state in state_vec

        action_vec = actions(mdp, state)
        state_i = stateindex(mdp, state)
        
        old_policy = policy_matrix[state_i]
        new_policy = nothing
        max_v = -Inf
        
        for action in action_vec
            action_i = actionindex(mdp, action)
            action_val = 0
            probability_distr = transition(mdp, state, action)
            
            for (next_state, prob) = weighted_iterator(probability_distr)
                if prob == 0
                    continue
                end
                r = reward(mdp, state, action, next_state)
                next_state_i = stateindex(mdp, next_state)
                action_val += prob * (r + discount_factor * value_matrix[next_state_i])
            end

            if action_val > max_v # find the action with the highest value
                max_v = action_val
                new_policy = action_i
            end

        end

        if !isnothing(new_policy)
            new_policy = 0
        end
        policy_matrix[state_i] = new_policy

        if old_policy != policy_matrix[state_i]
            converged = false
        end
    end

    return policy_matrix, converged
    
end

function policy_improvement(mdp::MDP, value_matrix::Vector, policy_matrix::Vector, 
    discount_factor::Float64, is_qmat::Bool, q_mat::Matrix)


    # @assert is_qmat == true "is_qmat is false, wrong method used"
    # @assert !isnothing(q_mat)  "Error, q_mat should not be nothing!"

    state_vec = ordered_states(mdp)
    converged = true

    for state in state_vec

        action_vec = actions(mdp, state)
        state_i = stateindex(mdp, state)
        
        old_policy = policy_matrix[state_i]
        new_policy = nothing
        max_v = -Inf
        
        for action in action_vec
            action_i = actionindex(mdp, action)
            action_val = 0
            probability_distr = transition(mdp, state, action)
            
            for (next_state, prob) = weighted_iterator(probability_distr)
                if prob == 0
                    continue
                end
                r = reward(mdp, state, action, next_state)
                next_state_i = stateindex(mdp, next_state)
                action_val += prob * (r + discount_factor * value_matrix[next_state_i])
            end

            if action_val > max_v # find the action with the highest value
                max_v = action_val
                new_policy = action_i
            end

            # Q-MAT == true option
            if is_qmat
                q_mat[state_i, action_i] = action_val
            end

        end

        # @assert !isnothing(new_policy) "Error, policy must not be null!"
        if isnothing(new_policy)
            # println(actions(mdp)[1])
            new_policy = 1
        end
        policy_matrix[state_i] = new_policy

        if old_policy != policy_matrix[state_i]
            converged = false
        end
    end

    if is_qmat
        return policy_matrix, q_mat, converged
    else
        return policy_matrix, converged
    end
end


function solve(::PolicyIterationSolver, ::POMDP)
    throw("""
           PolicyIterationError: `solve(::PolicyIterationSolver, ::POMDP)` is not supported,
          `PolicyIterationSolver` supports MDP models only, look at QMDP.jl for a POMDP solver that assumes full observability.
           """)
end