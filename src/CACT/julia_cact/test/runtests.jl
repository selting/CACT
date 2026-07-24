using julia_cact
using Test

@testset "julia_cact" begin
    include("test_tsp.jl")
    include("test_metrics.jl")
    include("test_run.jl")
end
