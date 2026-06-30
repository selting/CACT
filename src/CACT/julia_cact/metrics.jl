using Distances
function rmse(y_pred::Vector, y::Vector)
    rmse = sqrt(mean((y_pred - y) .^ 2))
    return rmse
end

function hausdorff_distance(set_a, set_b)
    function directed_hausdorff_distance(set_a, set_b)
        # directed (pompeiou)-hausdorff distance: max of distances between each point x in set_a and its nearest neighbor y in set_b
        pairwise_dist = pairwise(Euclidean(), set_a, set_b, dims=1)
        nearest_neighbor_dist = minimum(pairwise_dist, dims=2)
        max_ = maximum(nearest_neighbor_dist)
        return max_
    end

    hausdorff = max(directed_hausdorff_distance(set_a, set_b), directed_hausdorff_distance(set_b, set_a))
    return hausdorff
end


function test_distance(set_a, set_b)
    return 0.5
end

const TRUE_OBJECTIVE_REGISTRY = Dict{Symbol,Function}(
    :hausdorff_distance => hausdorff_distance,
    :test_distance => test_distance,
)

const PROXY_OBJECTIVE_REGISTRY = Dict{Symbol,Function}(
    :RMSE=>rmse,
)