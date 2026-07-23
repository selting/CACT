using JuMP
using Distances
using HiGHS



# function solve_tsp(solver::TSPSolver, locations::Matrix)::TSPResult

# end

function build_tsp_model(d, n, optimizer)
    model = Model(optimizer)
    set_silent(model)
    @variable(model, x[1:n, 1:n], Bin, Symmetric)
    @objective(model, Min, sum(d .* x)/2)
    @constraint(model, [i in 1:n], sum(x[i, :]) == 2)  # flow conservation?
    @constraint(model, [i in 1:n], x[i, i] == 0)  # no self-loops
    return model
end

function subtour(edges::Vector{Tuple{Int,Int}}, n)
    shortest_subtour, unvisited = collect(1:n), Set(collect(1:n))
    while !isempty(unvisited)
        this_cycle, neighbors = Int[], unvisited
        while !isempty(neighbors)
            current = pop!(neighbors)
            push!(this_cycle, current)
            if length(this_cycle) > 1
                pop!(unvisited, current)
            end
            neighbors = [j for (i, j) in edges if i == current && j in unvisited]
        end
        if length(this_cycle) < length(shortest_subtour)
            shortest_subtour = this_cycle
        end
    end
    return shortest_subtour
end

function selected_edges(x::Matrix{Float64}, n)
    return Tuple{Int,Int}[(i, j) for i in 1:n, j in 1:n if x[i, j] > 0.5]
end

subtour(x::Matrix{Float64}) = subtour(selected_edges(x, size(x, 1)), size(x, 1))
subtour(x::AbstractMatrix{VariableRef}) = subtour(value.(x))

function solve_tsp(solver::ExactJuMPSolver, locations::Matrix)::TSPResult
    # solve the traveling salesperson problem using JuMP
    # copied from the JuMP tutorial website
    # using the Iterative model: whenever a new subtour elimination constraint is added, start from scratch
    distances = pairwise(Euclidean(), locations')
    n = size(distances, 1)

    model = build_tsp_model(distances, n, solver.optimizer)
    set_optimizer_attribute(model, "threads", 1)  # force single threaded solving
    set_optimizer_attribute(model, "time_limit", solver.time_limit)
    set_optimizer_attribute(model, "mip_rel_gap", solver.mip_rel_gap)
    JuMP.optimize!(model, )
    assert_is_solved_and_feasible(model)
    total_time = solve_time(model)
    cycle = subtour(model[:x])

    while 1 < length(cycle) < n
        # println("Found cycle of length $(length(cycle))")
        S = [(i, j) for (i, j) in Iterators.product(cycle, cycle) if i < j]
        @constraint(model, sum(model[:x][i, j] for (i, j) in S) <= length(cycle) - 1)
        JuMP.optimize!(model)
        assert_is_solved_and_feasible(model)
        # global time_iterated += solve_time(iterative_model)
        total_time += solve_time(model)
        # global cycle = subtour(iterative_model[:x])
        cycle = subtour(model[:x])
    end

    return TSPResult(objective_value(model), cycle, total_time, termination_status(model))
end

function solve_tsp(::NearestNeighborSolver, locations::Matrix)::TSPResult
    t0 = time()
    distances = pairwise(Euclidean(), locations')
    n = size(distances, 1)

    visited = falses(n)
    tour = [1]
    visited[1] = true
    for _ in 2:n
        current = tour[end]
        next = argmin(j -> visited[j] ? Inf : distances[current, j], 1:n)
        push!(tour, next)
        visited[next] = true
    end
    obj = sum(distances[tour[i], tour[i % n + 1]] for i in 1:n)

    return TSPResult(obj, tour, time() - t0, false)
end