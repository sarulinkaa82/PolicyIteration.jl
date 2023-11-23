mutable struct PolicyIterationSolver <: Solver
    max_iterations::Int64 # max number of iterations
    belres::Float64 # the Bellman Residual
    verbose::Bool 
    include_Q::Bool
    init_util::Vector{Float64}
end

# Default constructor
function PolicyIterationSolver(;max_iterations::Int64 = 100, 
    belres::Float64 = 1e-3,
    verbose::Bool = false,
    include_Q::Bool = false,
    init_util::Vector{Float64}=Vector{Float64}(undef, 0))    

    return PolicyIterationSolver(max_iterations, belres, verbose, include_Q, init_util)
end


function solve(solver::PolicyIterationSolver, mdp::MDP; kwargs...)

    max_iterations = solver.max_iterations
    belres = solver.belres
    discount_factor = discount(mdp)
    ns = length(states(mdp))
    na = length(actions(mdp))

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

        
        # POLICY IMPROVEMENT
        # policy_improvement_res = policy_improvement(mdp, value_matrix, policy_matrix, discount_factor, solver.include_Q, qmat)
        if !solver.include_Q
            policy_matrix, converged = policy_improvement(mdp, value_matrix, policy_matrix, discount_factor)
        else
            policy_matrix, qmat, converged = policy_improvement(mdp, value_matrix, policy_matrix, discount_factor, solver.include_Q, qmat)
        end

        # if !solver.include_Q
        #     policy_matrix, converged = policy_improvement_res
        # else
        #     policy_matrix, qmat, converged = policy_improvement_res
        # end
        
    end

    if solver.include_Q
        return PolicyIterationPolicy(mdp, qmat, value_matrix, policy_matrix)
    else
        return PolicyIterationPolicy(mdp, utility=value_matrix, policy=policy_matrix, include_Q=false)
    end


end


function policy_evaluation(mdp::MDP, value_matrix::Vector,discount_factor::Float64 = 0.9, belres::Float64 = 1e-3)
    state_vec = ordered_states(mdp)
    delta = 0

    while delta < belres
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
                    new_v += prob * (r + discount_factor * value_matrix[next_state_i])
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

function policy_improvement(mdp::MDP, value_matrix::Vector, policy_matrix::Vector, 
    discount_factor::Float64 = 0.9)


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

        @assert !isnothing(new_policy) "Error, policy must not be null!"
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

        @assert !isnothing(new_policy) "Error, policy must not be null!"
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


function solve(::ValueIterationSolver, ::POMDP)
    throw("""
           ValueIterationError: `solve(::ValueIterationSolver, ::POMDP)` is not supported,
          `ValueIterationSolver` supports MDP models only, look at QMDP.jl for a POMDP solver that assumes full observability.
           """)
end