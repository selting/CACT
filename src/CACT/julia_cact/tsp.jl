using JuMP
using Distances
using HiGHS


function solve_tsp(locations::Matrix)
    # solve the traveling salesperson problem using JuMP
    # copied from the JuMP tutorial website

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

    # using the Iterative model: whenever a new subtour elimination constraint is added, start from scratch

    distances = pairwise(Euclidean(), locations')
    n = size(distances, 1)
    optimizer = HiGHS.Optimizer
    iterative_model = build_tsp_model(distances, n, optimizer)
    JuMP.optimize!(iterative_model)
    assert_is_solved_and_feasible(iterative_model)
    time_iterated = solve_time(iterative_model)
    cycle = subtour(iterative_model[:x])

    while 1 < length(cycle) < n
        # println("Found cycle of length $(length(cycle))")
        S = [(i, j) for (i, j) in Iterators.product(cycle, cycle) if i < j]
        @constraint(iterative_model, sum(iterative_model[:x][i, j] for (i, j) in S) <= length(cycle) - 1)
        JuMP.optimize!(iterative_model)
        assert_is_solved_and_feasible(iterative_model)
        # global time_iterated += solve_time(iterative_model)
        time_iterated += solve_time(iterative_model)
        # global cycle = subtour(iterative_model[:x])
        cycle = subtour(iterative_model[:x])
    end

    # println("Objective Value: $(objective_value(iterative_model))")
    # println("Runtime: $(time_iterated)")

    function plot_tour(X, Y, x)
        plot = Plots.plot()
        for (i, j) in selected_edges(x, size(x, 1))
            Plots.plot!([X[i], X[j]], [Y[i], Y[j]]; legend=false)
        end
        return plot
    end

    # X, Y = locations[:, 1], locations[:, 2]
    # plot_tour(X, Y, value.(iterative_model[:x]))

    return objective_value(iterative_model)
end