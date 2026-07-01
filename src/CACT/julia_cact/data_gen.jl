using Distributions
# using StatsBase
using Random

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

struct UniformGenerator <: LocationGenerator end

struct ClusteredGenerator <: LocationGenerator
    num_clusters::Int
    cluster_std::Float64
end

# Common interface, dispatched on generator type
function generate_locations(::UniformGenerator, rng, x_min, x_max, y_min, y_max, num_locations)
    x = rand(rng, Uniform(x_min, x_max), num_locations)
    y = rand(rng, Uniform(y_min, y_max), num_locations)
    return [x y]
end

function generate_locations(gen::ClusteredGenerator, rng, x_min, x_max, y_min, y_max, num_locations)
    center_x = rand(rng, Uniform(x_min, x_max), gen.num_clusters)
    center_y = rand(rng, Uniform(y_min, y_max), gen.num_clusters)
    assignments = rand(rng, 1:gen.num_clusters, num_locations)

    # TODO implement rejection sampling instead of clamping?!
    x = clamp.(center_x[assignments] .+ rand(rng, Normal(0, gen.cluster_std), num_locations), x_min, x_max)
    y = clamp.(center_y[assignments] .+ rand(rng, Normal(0, gen.cluster_std), num_locations), y_min, y_max)
    return [x y]
end


function generate_input_data(;generator::LocationGenerator, seed, x_min, x_max, y_min, y_max, true_num_locations::Int, size_auction_pool::Int, num_bundles::Int, pred_num_locations::Int)
    rng = MersenneTwister(seed)

    true_base_locations = generate_locations(generator, rng, x_min, x_max, y_min, y_max, true_num_locations)
    auction_pool = generate_locations(UniformGenerator(), rng, x_min, x_max, y_min, y_max, size_auction_pool) 

    bundles = draw_bundles(rng=rng, num_bundles=num_bundles, auction_pool=auction_pool)
    true_carrier_bids = compute_bids(true_base_locations, bundles)

    x0 = rand.(rng, Uniform.(repeat([x_min, y_min], pred_num_locations), repeat([x_max, y_max], pred_num_locations)))
    return Dict(
        "true_base_locations"=>true_base_locations,
        "auction_pool"=>auction_pool,
        "bundles"=>bundles,
        "true_carrier_bids"=>true_carrier_bids,
        "x0"=>x0
    )
end
