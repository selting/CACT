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

function solve_tsp(::HeldKarpSolver, locations::Matrix)::TSPResult
    t0 = time()
    distances = pairwise(Euclidean(), locations')
    n = size(distances, 1)

    n == 1 && return TSPResult(0.0, [1], time() - t0, true)

    # Held-Karp DP. City 1 is the fixed start/end of the cycle.
    # C[j, mask+1] = min cost of a path starting at city 1, visiting exactly the
    # cities in `mask`, and ending at city j. Cities are 0-indexed within the
    # bitmask (`mask` always has bit 0 set); the array is 1-indexed, hence mask+1.
    # parent[j, mask+1] holds the city visited just before j, for tour recovery.
    full = (1 << n) - 1
    C = fill(Inf, n, 1 << n)
    parent = zeros(Int, n, 1 << n)

    # base case: the path 1 -> j
    for j in 2:n
        mask = 1 | (1 << (j - 1))
        C[j, mask+1] = distances[1, j]
        parent[j, mask+1] = 1
    end

    for mask in 0:full
        (mask & 1) == 0 && continue          # subset must contain city 1
        for j in 2:n
            jbit = 1 << (j - 1)
            (mask & jbit) == 0 && continue   # j must be in the subset
            prev = mask ⊻ jbit               # subset before arriving at j
            best, bestk = C[j, mask+1], parent[j, mask+1]
            for k in 2:n
                (prev & (1 << (k - 1))) == 0 && continue
                val = C[k, prev+1] + distances[k, j]
                if val < best
                    best, bestk = val, k
                end
            end
            C[j, mask+1], parent[j, mask+1] = best, bestk
        end
    end

    # close the cycle: return from the last city back to city 1
    best_obj, last_city = Inf, 1
    for j in 2:n
        val = C[j, full+1] + distances[j, 1]
        if val < best_obj
            best_obj, last_city = val, j
        end
    end

    # reconstruct the tour by walking predecessors back to city 1
    tour = Int[last_city]
    mask, j = full, last_city
    while true
        k = parent[j, mask+1]
        mask ⊻= (1 << (j - 1))
        push!(tour, k)
        k == 1 && break
        j = k
    end
    reverse!(tour)  # now starts at city 1

    return TSPResult(best_obj, tour, time() - t0, true)
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