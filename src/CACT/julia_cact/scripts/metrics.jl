using Distances

abstract type TrueObjectiveFunction end

struct HausdorffDistance <: TrueObjectiveFunction end
Base.string(::HausdorffDistance) = "HausdorffDistance"

struct NormalizedHausdorffDistance <: TrueObjectiveFunction
    x_min::Real
    x_max::Real
    y_min::Real
    y_max::Real
end
Base.string(NormalizedHausdorffDistance) = "NormalizedHausdorffDistance"

struct TestDistance <: TrueObjectiveFunction end

function directed_hausdorff_distance(set_a, set_b)
    # directed (pompeiou)-hausdorff distance: max of distances between each point x in set_a and its nearest neighbor y in set_b
    pairwise_dist = pairwise(Euclidean(), set_a, set_b, dims=1)
    nearest_neighbor_dist = minimum(pairwise_dist, dims=2)
    max_ = maximum(nearest_neighbor_dist)
    return max_
end

function compute_true_objective(f::HausdorffDistance, set_a, set_b)
    hausdorff = max(directed_hausdorff_distance(set_a, set_b), directed_hausdorff_distance(set_b, set_a))
    return hausdorff
end

function compute_true_objective(f::NormalizedHausdorffDistance, set_a, set_b)
    hausdorff = max(directed_hausdorff_distance(set_a, set_b), directed_hausdorff_distance(set_b, set_a))
    denom = euclidean((f.x_min, f.y_min), (f.x_max, f.y_max))
    norm_hausdorff = hausdorff/denom
    return norm_hausdorff
end

function compute_true_objective(f::TestDistance, set_a, set_b)
    return 0.5
end

#---------------------------------------------------------------------

abstract type ProxyObjectiveFunction end

struct RMSE <: ProxyObjectiveFunction end
Base.string(::RMSE) = "RMSE"
struct MSE <: ProxyObjectiveFunction end
Base.string(::MSE) = "MSE"

struct HuberLoss <: ProxyObjectiveFunction
    delta::Float64
end
Base.string(l::HuberLoss) = "Huberloss($l.delta)"

compute_proxy_objective(::RMSE, y_pred, y) = sqrt(mean((y_pred .- y) .^ 2))
compute_proxy_objective(::MSE, y_pred, y) = mean((y_pred .- y) .^ 2)

function compute_proxy_objective(l::HuberLoss, y_pred, y)
    r = abs.(y_pred .- y)
    return mean(ifelse.(r .<= l.delta, 0.5 .* r .^ 2, l.delta .* (r .- 0.5 * l.delta)))
end