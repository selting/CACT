from plotnine import (
    theme_gray,
    theme,
    element_text,
    element_rect,
    element_line,
    scale_color_manual,
)

c1, c2 = "#5B4F41", "#FCF9F4"

my_theme = theme_gray() + theme(
    # figure_size=(8, 6),
    text=element_text(
        color=c1,
        size=13,
    ),  # family="STIXGeneral"),
    # legend_position="none",
    panel_background=element_rect(fill=c2),
    panel_grid_major=element_line(color=c1, linetype="dashdot", alpha=0.1),
    # panel_grid_minor=element_blank(),
    panel_border=element_rect(color=c1),
    # axis_text=element_text(size=8),
    strip_text=element_text(color="white"),
    strip_background=element_rect(fill=c1, color=c1),
)

# Use the same scale for the color and fill aesthetics
my_scale_color_and_fill = scale_color_manual(
    values=[
        "#DB735C",
        "#2A91A2",
        "#F8B75A",
        "#8CBA80",
        "#474973",
        ],
    aesthetics=("fill", "color"),
)
