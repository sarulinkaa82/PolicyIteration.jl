using POMDPs
using POMDPModels, POMDPLinter
using POMDPTools

export generate_test_domain

struct GWPos
    x::Int
    y::Int

    function GWPos(x::Int, y::Int)
        new(x, y)
    end
end

Base.@kwdef struct CustomDomain <: MDP{GWPos, Symbol}
    size::Tuple{Int64, Int64}
    grid::Matrix
    t_prob::Float64                = 0.7
    discount::Float64              = 0.95
    rewards::Dict{String, Float64} = Dict("step" => -1.0, "E" => 0.0, "D" => -50.0)
end

# file = open("C:/repos/jukia_solvers/PolicyIteration.jl/examples/maze-7-A1.txt", "r")

# init domain like:
# sizee, mat = generate_test_domain(filepath)
# mdp = CustomDomain(size = sizee, grid = mat)
function generate_test_domain(filepath::String)
    
    df = open(filepath, "r")

    size_string = readline(df)
    lb, ub = split(size_string, " ")
    size = (parse(Int64, lb), parse(Int64, ub))

    grid = fill("def", size)

    i = 1
    for line in readlines(df)
        row = split(line, "")
        row = String.(row)
        grid[i, :] = row
        i += 1
    end
    close(df)

    return size, grid
end

# STATES

function POMDPs.states(mdp::CustomDomain)
    state_vec = Vector{GWPos}()
    
    for x in 1:mdp.size[1], y in 1:mdp.size[2]
        if mdp.grid[x, y] != "#"
            pos = GWPos(x, y)
            push!(state_vec, pos)
        end
    end    
    return state_vec
end

function POMDPs.stateindex(mdp::CustomDomain, s::GWPos)
    all_states = POMDPs.states(mdp)
    for index in eachindex(all_states) # linear indexing
        if all_states[index] == s
            return index
        end
    end
    return -1
end

function POMDPs.initialstate(mdp::CustomDomain)
    state_vec = POMDPs.states(mdp)
    init_state_prob = zeros(Int64, length(state_vec))
    init_state_prob[mdp.size[2] + 2] = 1
    # println(POMDPs.states(mdp)[mdp.size[2] + 2])
    return init_state_prob
end

# ACTIONS

function POMDPs.actions(mdp::CustomDomain)
    return (:up, :down, :left, :right)
end

const aind = Dict(:up=>1, :down=>2, :left=>3, :right=>4)

function POMDPs.actionindex(mdp::CustomDomain, a::Symbol)
    return aind[a]
end

# TRANSITIONS

function POMDPs.isterminal(mdp::CustomDomain, s::GWPos)
    if mdp.grid[s.x, s.y] == "E"
        return true
    else
        return false
    end
end

function POMDPs.transition(mdp::CustomDomain, s::GWPos, a::Symbol)
    if isterminal(mdp, s) || !in_bounds(mdp, s)
        return Deterministic(s)
    end

    possible_dest_nr = length(POMDPs.actions(mdp)) + 1
    destinations = Vector{GWPos}(undef, possible_dest_nr)
    probabilities = zeros(possible_dest_nr)


    for (i, action) in enumerate(actions(mdp)) # up, down, left, right
        if action == a # probability of transitioning to the desired cell
            prob = mdp.t_prob 
        else # probability of transitioning to another cell
            # prob = (1.0 - mdp.tprob)/(length(actions(mdp)) - 1) 
            prob = (1.0 - mdp.t_prob) / 3
        end
        # println(action, " ", prob)

        dest = next_state(mdp, s, action)
        destinations[1] = s
        destinations[i+1] = dest

        if !in_bounds(mdp, dest) # hit a wall and come back to s
            probabilities[1] += prob
        else
            probabilities[i+1] += prob
        end
    end

    return SparseCat(destinations, probabilities)
end

function in_bounds(mdp::CustomDomain, s::GWPos) 
    if 1 < s.x < mdp.size[1] && 1 < s.y < mdp.size[2]
        return mdp.grid[s.x, s.y] != "#"
    else
        return false
    end
end

function next_state(mdp::CustomDomain, s::GWPos, a::Symbol)
    x = s.x
    y = s.y
    # println("here")


    if a == :up
        x -= 1
    elseif a == :down
        x += 1
    elseif a == :left
        y -= 1
    elseif a == :right
        y += 1
    end

    # check if in boundaries
    new_state = GWPos(x, y)
    if !in_bounds(mdp, new_state)
        return s
    else
        # println("not supposed to be here")
        return new_state
    end
end

# REWARDS

function POMDPs.reward(mdp::CustomDomain, s::GWPos, a::Symbol)
    ns = next_state(mdp, s, a)
    # println(s)
    # println(ns)

    if mdp.grid[ns.x, ns.y] == "D"
        return mdp.rewards["D"]
    elseif mdp.grid[ns.x, ns.y] == "E"
        return mdp.rewards["E"]
    else
        return mdp.rewards["step"]
    end
end

function POMDPs.reward(mdp::CustomDomain, s::GWPos, a::Symbol, sp::GWPos)
    
    # println(s)
    # println(ns)

    if mdp.grid[sp.x, sp.y] == "D"
        return mdp.rewards["D"]
    elseif mdp.grid[sp.x, sp.y] == "E"
        return mdp.rewards["E"]
    else
        return mdp.rewards["step"]
    end
end


# DISCOUNT

function POMDPs.discount(mdp::CustomDomain)
    return mdp.discount
end

# CONVERSIONS? - not sure if needed

function POMDPs.convert_a(::Type{V}, a::Symbol, m::SimpleGridWorld) where {V<:AbstractArray}
    convert(V, [aind[a]])
end
function POMDPs.convert_a(::Type{Symbol}, vec::V, m::SimpleGridWorld) where {V<:AbstractArray}
    actions(m)[convert(Int, first(vec))]
end