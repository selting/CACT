using Test
using julia_cact

@testset "metrics" begin
    @testset "proxy objectives (RMSE / MSE / Huber)" begin
        y      = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        @test compute_proxy_objective(RMSE(), y_pred, y) == 0.0
        @test compute_proxy_objective(MSE(), y_pred, y) == 0.0

        # residuals [3, 4] -> MSE = mean([9,16]) = 12.5, RMSE = sqrt(12.5)
        @test compute_proxy_objective(MSE(), [0.0, 0.0], [3.0, 4.0]) ≈ 12.5
        @test compute_proxy_objective(RMSE(), [0.0, 0.0], [3.0, 4.0]) ≈ sqrt(12.5)

        # all residuals below delta -> Huber is the quadratic branch: mean(0.5 r^2)
        @test compute_proxy_objective(HuberLoss(10.0), [0.0, 0.0], [1.0, 1.0]) ≈ 0.5
        # large residual with small delta -> linear branch dominates, stays below the pure-quadratic value
        @test compute_proxy_objective(HuberLoss(1.0), [0.0], [10.0]) < compute_proxy_objective(MSE(), [0.0], [10.0])
    end

    @testset "Hausdorff distance" begin
        a = [0.0 0.0]
        b = [3.0 4.0]
        @test compute_true_objective(HausdorffDistance(), a, b) ≈ 5.0
        # symmetric
        @test compute_true_objective(HausdorffDistance(), a, b) ==
              compute_true_objective(HausdorffDistance(), b, a)
        # identical sets -> zero
        pts = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        @test compute_true_objective(HausdorffDistance(), pts, pts) ≈ 0.0
    end

    @testset "Normalized Hausdorff in [0,1]" begin
        f = NormalizedHausdorffDistance(0, 100, 0, 100)  # diagonal = sqrt(2)*100
        corner_a = [0.0 0.0]
        corner_b = [100.0 100.0]
        @test compute_true_objective(f, corner_a, corner_b) ≈ 1.0          # spans the full diagonal
        @test compute_true_objective(f, corner_a, corner_a) ≈ 0.0
        # a mid-range configuration stays strictly inside (0,1)
        v = compute_true_objective(f, [0.0 0.0], [50.0 0.0])
        @test 0.0 < v < 1.0
    end
end
