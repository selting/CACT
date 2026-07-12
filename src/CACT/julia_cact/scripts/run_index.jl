using DataFrames, Arrow, JLD2

function flatten_for_index(r::RunResult)
    return (
        seed=r.seed,
        true_location_generator=string(typeof(r.true_location_generator)),
        auction_pool_location_generator=string(typeof(r.auction_pool_location_generator)),
        x_min=r.x_min,
        x_max=r.x_max,
        y_min=r.y_min,
        y_max=r.y_max,
        size_auction_pool=r.size_auction_pool,
        num_bundles=r.num_bundles,
        pred_num_locations=r.pred_num_locations,

        tsp_solver=string(typeof(r.tsp_solver)),
        optimizer=string(typeof(r.optimizer)),
        x0_seeder=string(typeof(r.x0_seeder)),
        proxy_objective_function=string(typeof(r.proxy_objective_function)),
        true_objective_functions=join([string(typeof(f)) for f in r.true_objective_functions], ", "),

        # true_base_locations=r.true_base_locations,
        # auction_pool_locations=r.auction_pool_locations,
        # bundles=r.bundles,
        # bids=r.bids,
        # optimize_results=r.optimize_results, # don't store the results in the index
        run_id=string(r.run_id),
        timestamp=r.timestamp,
        tags=join(r.tags, ", "),
        )
end

function build_index(; dir=datadir("exp_raw"))
    files = filter(f -> endswith(f, ".jld2"), readdir(dir; join=true))
    rows = [flatten_for_index(JLD2.load(f, "res")) for f in files]
    df = DataFrame(rows)
    Arrow.write(joinpath(dir, "..", "index_exp_raw.arrow"), df)
    return df
end

df = build_index()