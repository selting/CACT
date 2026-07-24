using Test
using julia_cact

# small config factory so each test can tweak just the optimizer
function _small_config(optimizer; seed=1)
    return CactConfig(
        seed = seed,
        true_base_location_generator = UniformLocationGenerator(),
        auction_pool_location_generator = UniformLocationGenerator(),
        x_min = 0.0, x_max = 100.0, y_min = 0.0, y_max = 100.0,
        size_auction_pool = 8,
        num_bundles = 12,
        true_num_locations = 6,
        pred_num_locations = 6,
        tsp_solver = HeldKarpSolver(),
        optimizer = optimizer,
        x0_seeder = UniformRandomSeeder(),
        proxy_objective_function = RMSE(),
        true_objective_functions = (HausdorffDistance(), NormalizedHausdorffDistance(0, 100, 0, 100)),
        tags = ("unit_test",),
    )
end

@testset "run_simulation" begin
    @testset "input data has the expected shapes" begin
        _, res = run_simulation(_small_config(NoOpt()))
        @test res.proxy_objective_opt isa Float64
    end

    @testset "accumulators are concretely typed (not Vector{Any})" begin
        _, res = run_simulation(_small_config(NLOPT(:LN_NELDERMEAD, 32)))
        @test res.x_trajectory isa Vector{Matrix{Float64}}
        @test res.incumbent_x_trajectory isa Vector{Matrix{Float64}}
        @test res.proxy_objective_trajectory isa Vector{Float64}
        @test res.incumbent_proxy_objective_trajectory isa Vector{Float64}
        @test res.true_objectives_trajectory isa Dict{Symbol,Vector{Float64}}
        @test res.num_evals isa Int
    end

    @testset "trajectories are internally consistent" begin
        _, res = run_simulation(_small_config(RandomSearch(24)))
        n = length(res.proxy_objective_trajectory)
        @test n == length(res.x_trajectory)
        @test n == length(res.incumbent_proxy_objective_trajectory)
        # the incumbent proxy is the running minimum, so it must be non-increasing
        @test issorted(res.incumbent_proxy_objective_trajectory; rev=true)
        # and the recorded optimum equals the best incumbent
        @test res.proxy_objective_opt ≈ minimum(res.proxy_objective_trajectory)
    end

    @testset "every optimizer dispatch produces a valid result" begin
        for opt in (NoOpt(), NLOPT(:GN_DIRECT_L_RAND, 16), RandomSearch(16))
            _, res = run_simulation(_small_config(opt; seed=3))
            @test res.proxy_objective_opt ≥ 0.0          # RMSE is non-negative
            @test haskey(res.true_objectives_opt, :HausdorffDistance)
        end
    end
end
