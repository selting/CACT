function run_simulation(config::CactConfig)

    rng = Xoshiro(config.seed)

    input_data = generate_input_data(;
        rng=rng,
        true_location_generator=config.true_base_location_generator,
        auction_pool_location_generator=config.auction_pool_location_generator,
        x_min=config.x_min,
        x_max=config.x_max,
        y_min=config.y_min,
        y_max=config.y_max,
        true_num_locations=config.true_num_locations,
        size_auction_pool=config.size_auction_pool,
        num_bundles=config.num_bundles,
        tsp_solver=config.tsp_solver,
        pred_num_locations=config.pred_num_locations
    )
    optimize_result = auctioneer_optimize(
        config.optimizer,
        rng=rng,
        bundles=input_data.bundles,
        bids=input_data.true_carrier_bids,
        tsp_solver=config.tsp_solver,
        pred_num_locations=config.pred_num_locations,
        _true_base_locations=input_data.true_base_locations,
        x0_seeder=config.x0_seeder,
        params_lower_bounds=[config.x_min, config.y_min],  # TODO avoid the back and forth from x_min, x_max, y_min, y_max to vectors of min and max values. I guess its best to stick to xmin, xmax where possible
        params_upper_bounds=[config.x_max, config.y_max],
        proxy_objective_function=config.proxy_objective_function,
        true_objective_functions=config.true_objective_functions,
    )

    return (input_data, optimize_result)
end
