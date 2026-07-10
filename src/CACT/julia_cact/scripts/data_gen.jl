using Distributions
using Random
include("target_function.jl")

function draw_bundles(; rng, num_bundles::Int, auction_pool::Matrix)
    # use a set of tuples to track uniqueness cleanly
    unique_bundles = Set()
    bundles = []  # TODO better preallocate the space and populate in the loop
    pool_size = size(auction_pool, 1)

    while length(bundles) < num_bundles
        # 1 & 2: Random size between 1 and size_auction_pool (inclusive)
        bundle_size = rand(rng, 1:pool_size)

        # draw items without replacement
        bundle_indices = sample(rng, 1:pool_size, bundle_size, replace=false)
        bundle_ind_sorted = Tuple(sort(bundle_indices))

        # bundle=auction_pool[bundle_indices, :]

        if !(bundle_ind_sorted in unique_bundles)
            push!(unique_bundles, bundle_ind_sorted)
            push!(bundles, auction_pool[bundle_indices, :])
        end
    end
    return bundles
end

abstract type LocationGenerator end

struct UniformLocationGenerator <: LocationGenerator end

struct ClusteredLocationGenerator <: LocationGenerator
    num_clusters::Int
    cluster_std::Float64
end

struct GridGenerator <: LocationGenerator end

# Common interface, dispatched on generator type
function generate_locations(::UniformLocationGenerator, rng, x_min, x_max, y_min, y_max, num_locations)::Matrix{Float64}
    x = rand(rng, Uniform(x_min, x_max), num_locations)
    y = rand(rng, Uniform(y_min, y_max), num_locations)
    return [x y]
end

function generate_locations(gen::ClusteredLocationGenerator, rng, x_min, x_max, y_min, y_max, num_locations)::Matrix{Float64}
    center_x = rand(rng, Uniform(x_min, x_max), gen.num_clusters)
    center_y = rand(rng, Uniform(y_min, y_max), gen.num_clusters)
    assignments = rand(rng, 1:gen.num_clusters, num_locations)

    # TODO implement rejection sampling instead of clamping?!
    x = clamp.(center_x[assignments] .+ rand(rng, Normal(0, gen.cluster_std), num_locations), x_min, x_max)
    y = clamp.(center_y[assignments] .+ rand(rng, Normal(0, gen.cluster_std), num_locations), y_min, y_max)
    return [x y]
end

function generate_locations(::GridGenerator, x_min, x_max, y_min, y_max, num_locations)::Matrix{Float64}
    # distribute num_locations as evenly as possible on a grid
    rows = columns = sqrt(num_locations)
    # if its not an integer, we need to round
    if rows % 1 != 0
        rows = ceil(rows)
        if columns % 1 < 0.5
            columns = floor(columns)
        else
            columns = ceil(columns)
        end
    end
    rows = Int64(rows)
    columns = Int64(columns)
    println("distributing $num_locations in a $rows by $columns grid")
    # exclude the very edges of the map (+2 for the edges; [2:end-1] to exclude them)
    x = LinRange(x_min, x_max, rows + 2)[2:end-1]
    y = LinRange(y_min, y_max, columns + 2)[2:end-1]
    grid = vec(collect(Iterators.product(x, y)))
    return grid[1:num_locations]
end

struct InputData
    true_base_locations::Matrix{Float64}
    auction_pool_locations::Matrix{Float64}
    bundles
    true_carrier_bids
    # x0
end

function generate_input_data(;
    true_location_generator::LocationGenerator,
    auction_pool_location_generator::LocationGenerator,
    rng,
    x_min,
    x_max,
    y_min,
    y_max,
    true_num_locations::Int,
    size_auction_pool::Int,
    num_bundles::Int,
    tsp_solver::TSPSolver,
    pred_num_locations::Int
    )::InputData
    true_base_locations = generate_locations(true_location_generator, rng, x_min, x_max, y_min, y_max, true_num_locations)
    auction_pool_locations = generate_locations(auction_pool_location_generator, rng, x_min, x_max, y_min, y_max, size_auction_pool)

    bundles = draw_bundles(rng=rng, num_bundles=num_bundles, auction_pool=auction_pool_locations)
    true_carrier_bids = compute_bids(tsp_solver, true_base_locations, bundles)

    # x0 = rand.(rng, Uniform.(repeat([x_min, y_min], pred_num_locations), repeat([x_max, y_max], pred_num_locations)))
    return InputData(
        true_base_locations,
        auction_pool_locations,
        bundles,
        true_carrier_bids,
        # x0
    )
end
