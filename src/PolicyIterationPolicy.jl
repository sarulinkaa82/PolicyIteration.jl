struct PolicyIterationPolicy{Q<:AbstractMatrix, U<:AbstractVector, P<:AbstractVector, A, M<:MDP} <: Policy
    qmat::Q
    util::U 
    policy::P 
    action_map::Vector{A}
    include_Q::Bool 
    mdp::M
end


# constructor with an optinal initial value function argument
function PolicyIterationPolicy(mdp::Union{MDP,POMDP}; 
                               utility::AbstractVector{Float64}=zeros(length(states(mdp))),
                               policy::AbstractVector{Int64}=zeros(Int64, length(states(mdp))), 
                               include_Q::Bool=true)
    ns = length(states(mdp))
    na = length(actions(mdp))
    @assert length(utility) == ns "Input utility dimension mismatch"
    @assert length(policy) == ns "Input policy dimension mismatch"
    action_map = ordered_actions(mdp)

    if include_Q
        qmat = zeros(ns, na)
    else
        qmat = zeros(0,0)
    end

    return PolicyIterationPolicy(qmat, utility, policy, action_map, include_Q, mdp)
end

# constructor for solved q, util and policy
function PolicyIterationPolicy(mdp::Union{MDP,POMDP}, 
                               q::AbstractMatrix{F}, 
                               util::AbstractVector{F}, 
                               policy::Vector{Int64}) where {F}

    action_map = ordered_actions(mdp)
    include_Q = true
    return PolicyIterationPolicy(q, util, policy, action_map, include_Q, mdp)
end

# constructor for default Q-matrix
function PolicyIterationPolicy(mdp::Union{MDP,POMDP}, q::AbstractMatrix)
    
    (ns, na) = size(q)
    p = zeros(Int64, ns)
    u = zeros(ns)
    for i = 1:ns
        p[i] = argmax(q[i,:])
        u[i] = maximum(q[i,:])
    end
    action_map = ordered_actions(mdp)
    include_Q = true
    return PolicyIterationPolicy(q, u, p, action_map, include_Q, mdp)
end


# returns the fields of the policy type
function locals(p::PolicyIterationPolicy)
    return (p.qmat,p.util,p.policy,p.action_map)
end

function POMDPs.action(policy::PolicyIterationPolicy, s)
    sidx = stateindex(policy.mdp, s)
    aidx = policy.policy[sidx]
    return policy.action_map[aidx]
end

function POMDPs.value(policy::PolicyIterationPolicy, s)
    sidx = stateindex(policy.mdp, s)
    policy.util[sidx]
end

value(policy::PolicyIterationPolicy, s, a) = actionvalues(policy, s)[actionindex(policy.mdp, a)]

function POMDPTools.Policies.actionvalues(policy::PolicyIterationPolicy, s)
    if !policy.include_Q
        error("Policy does not contain the Q matrix. Use the include_Q=true keyword argument in the solver.")
    else
        sidx = stateindex(policy.mdp, s)
        return policy.qmat[sidx,:]
    end
end

function Base.show(io::IO, mime::MIME"text/plain", p::PolicyIterationPolicy)
    println(io, "PolicyIterationPolicy:")
    ds = get(io, :displaysize, displaysize(io))
    ioc = IOContext(io, :displaysize=>(first(ds)-1, last(ds)))
    showpolicy(ioc, mime, p.mdp, p)
end