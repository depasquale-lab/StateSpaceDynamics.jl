using CSV, DataFrames, CategoricalArrays
using StatsPlots
using Statistics

# Load data
df = CSV.read("results/lds_benchmark_results.csv", DataFrame)

# Prepare columns
df.seq_len = categorical(df.seq_len, ordered=true)
df.implementation = categorical(df.implementation)
df.config = string.(df.latent_dim) .* "x" .* string.(df.obs_dim)

# Compute bar means
bar_df = combine(groupby(df, [:seq_len, :implementation]),
    :time_sec => mean => :mean_time)

# Compute per-config means (for dots)
dot_df = combine(groupby(df, [:seq_len, :implementation, :config]),
    :time_sec => mean => :mean_time)

# Marker and color mappings
marker_types = [:circle, :rect, :diamond, :star5, :cross, :utriangle, :hexagon, :xcross, :pentagon, :dtriangle]
unique_configs = unique(dot_df.config)
config_marker = Dict(cfg => marker_types[mod1(i, length(marker_types))] for (i, cfg) in enumerate(unique_configs))

impl_colors = Dict(
    "pykalman" => :blue,
    "Dynamax" => :green,
    "StateSpaceDynamics.jl" => :orange,
)

implementations = levels(df.implementation)

# Plot loop
plots = []
for (i, sl) in enumerate(levels(df.seq_len))
    sub_bar = bar_df[bar_df.seq_len .== sl, :]
    sub_dot = dot_df[dot_df.seq_len .== sl, :]

    # Base bar plot (no xticks/labels)
    p = bar(
        sub_bar.implementation,
        sub_bar.mean_time,
        yticks = :auto,
        xticks = false,
        xguide = false,  # removes x-axis label
        xlabel = "",
        ylabel = "Mean Time (s)",
        title = "n = $(sl)",
        color = [impl_colors[String(impl)] for impl in sub_bar.implementation],
        alpha = 0.6,
        lw = 0.5,
        legend = false,
        size = (400, 350),
    )

    # Overlay dots with same color as bars
    for row in eachrow(sub_dot)
        scatter!(
            [row.implementation],
            [row.mean_time],
            marker = (config_marker[row.config], 6),
            color = impl_colors[String(row.implementation)],
            alpha = 1.0,
            label = false,
            yticks = :auto,
            xticks = false,
        )
    end

    # Legend only once
    if i == 1
        # Impl (bar) legend
        for (name, col) in impl_colors
            bar!([NaN], [NaN], color=col, label=name)
        end
        # Config (marker) legend
        for (cfg, mkr) in config_marker
            scatter!([NaN], [NaN], marker=(mkr, 6), color=:gray, label=cfg)
        end
    end

    push!(plots, p)
end

# Final layout: tight
p1 = plot(plots..., layout = (1, length(plots)), legend = :outerleft, size=(length(plots)*400, 350))
savefig(p1, "results/lds_bench.svg")
