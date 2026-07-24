using StatsPlots
using AlgebraOfGraphics, CairoMakie
include("data_prep.jl")

ssdf, agg_msdf = get_results()

@df ssdf StatsPlots.boxplot(string.(:optimizer), :proxy_objective_opt)
@df ssdf StatsPlots.dotplot!(string.(:optimizer), :proxy_objective_opt)

# then plot convergence, via AlgebraOfGraphics so faceting/coloring/linestyling on
# any parameter column is just a matter of changing which column a keyword below points to.
# Any of the four can be set to `nothing` to drop that aesthetic/facet entirely.
# NoOpt only ever has a single evaluation (no "steps"), so it can't share the :step x-axis
# with the other optimizers -- draw it as a flat benchmark line + band spanning each facet instead.
color_col = :optimizer_combined       # swap to any group_cols entry to recolor by that instead, or `nothing`
col_facet = :pred_num_locations       # facet columns of the grid, or `nothing`
row_facet = :size_auction_pool        # facet rows of the grid, or `nothing`
linestyle_col = nothing            # swap to any group_cols entry to re-style by that instead, or `nothing`

# mapping() errors if an aesthetic is passed `nothing` directly, so only pass through
# the keywords whose column is actually set
skip_nothing(; kwargs...) = (; (k => v for (k, v) in pairs(kwargs) if v !== nothing)...)

curve_aes = skip_nothing(; color=color_col, linestyle=linestyle_col)
static_aes = skip_nothing(; color=color_col)
facet_aes = skip_nothing(;
    col=col_facet === nothing ? nothing : col_facet => nonnumeric,
    row=row_facet === nothing ? nothing : row_facet => nonnumeric,
)

# split off NoOpt by the fixed :optimizer column, independent of color_col/linestyle_col above
# (which may be disabled or point elsewhere)
curves_df = filter(:optimizer => !=("NoOpt"), agg_msdf)
baseline_df = filter(:optimizer => ==("NoOpt"), agg_msdf)

curve_layer = data(curves_df) * mapping(
                  :step, :incumbent_proxy_objective_mean;
                  curve_aes..., facet_aes...,
              ) * visual(Lines)
curve_band_layer = data(curves_df) * mapping(
                       :step, :incumbent_proxy_objective_lower, :incumbent_proxy_objective_upper;
                       static_aes..., facet_aes...,
                   ) * visual(Band, alpha=0.25)

baseline_layer = data(baseline_df) * mapping(
                     :proxy_objective_mean;
                     static_aes..., facet_aes...,
                 ) * visual(HLines, linestyle=:dash)
baseline_band_layer = data(baseline_df) * mapping(
                          :proxy_objective_lower, :proxy_objective_upper;
                          static_aes..., facet_aes...,
                      ) * visual(HSpan, alpha=0.15)

fig = draw(curve_band_layer + curve_layer + baseline_band_layer + baseline_layer;
    axis=(; xlabel="step", ylabel="proxy objective (mean ± std)"))
fig
