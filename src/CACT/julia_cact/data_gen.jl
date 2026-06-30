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

function generate_input_data(; seed, x_min, x_max, y_min, y_max, true_num_locations::Int, size_auction_pool::Int, num_bundles::Int, pred_num_locations::Int)
    rng = MersenneTwister(seed)

    true_base_locations_x = rand.(rng, Uniform.(x_min, x_max), true_num_locations)
    true_base_locations_y = rand.(rng, Uniform.(y_min, y_max), true_num_locations)
    true_base_locations = [true_base_locations_x true_base_locations_y]

    auction_pool_x = rand.(rng, Uniform.(x_min, x_max), size_auction_pool)
    auction_pool_y = rand.(rng, Uniform.(y_min, y_max), size_auction_pool)
    auction_pool = [auction_pool_x auction_pool_y]

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
