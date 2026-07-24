using Test
using Random
using Distances
using julia_cact

# recompute a tour's cycle cost independently, and assert it's a valid permutation
function _tour_cost(locations, tour)
    d = pairwise(Euclidean(), locations')
    n = size(d, 1)
    @test sort(tour) == collect(1:n)          # every city visited exactly once
    return sum(d[tour[i], tour[i % n + 1]] for i in 1:n)
end

@testset "tsp" begin
    @testset "Held-Karp matches HiGHS exact solver" begin
        Random.seed!(42)
        # n=2 is infeasible in the HiGHS 2-matching formulation, so start at n=3
        for n in (3, 5, 8), _ in 1:5
            locs = rand(n, 2) .* 100
            hk = solve_tsp(HeldKarpSolver(), locs)
            ex = solve_tsp(ExactJuMPSolver(), locs)
            @test hk.objective ≈ ex.objective atol = 1e-6
            @test hk.optimal
        end
    end

    @testset "Held-Karp recovers a valid optimal tour" begin
        Random.seed!(7)
        for n in (3, 6, 9)
            locs = rand(n, 2) .* 100
            res = solve_tsp(HeldKarpSolver(), locs)
            @test _tour_cost(locs, res.tour) ≈ res.objective atol = 1e-9
        end
    end

    @testset "Held-Karp edge cases" begin
        single = solve_tsp(HeldKarpSolver(), reshape([10.0, 20.0], 1, 2))
        @test single.objective == 0.0
        @test single.tour == [1]

        # two coincident points on top of two others: cost is a well-defined round trip
        locs = [0.0 0.0; 3.0 4.0]           # distance 5 each way
        two = solve_tsp(HeldKarpSolver(), locs)
        @test two.objective ≈ 10.0 atol = 1e-9
    end

    @testset "Nearest-neighbor returns a valid (not necessarily optimal) tour" begin
        Random.seed!(1)
        locs = rand(6, 2) .* 100
        nn = solve_tsp(NearestNeighborSolver(), locs)
        hk = solve_tsp(HeldKarpSolver(), locs)
        @test _tour_cost(locs, nn.tour) ≈ nn.objective atol = 1e-9
        @test nn.objective ≥ hk.objective - 1e-9   # heuristic can't beat the exact optimum
    end
end
