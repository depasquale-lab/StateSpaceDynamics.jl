using CategoricalArrays
using StatsPlots
using DataFrames
using CSV
using Statistics

# Load data
dir = "results/"
filepath = joinpath(dir, "lds_benchmark_results.csv")
df = CSV.read(filepath, DataFrame)

# Ensure consistent ordering for sequence length (x-axis)
seq_order = sort(unique(df.seq_len))
df.seq_len_cat = categorical(string.(df.seq_len), ordered=true, levels=string.(seq_order))

# Ensure consistent ordering for implementation (bar group order)
df.implementation = categorical(df.implementation)
implementations = levels(df.implementation)

# Compute mean execution time per (sequence length, implementation) group
agg_df = combine(groupby(df, [:seq_len_cat, :implementation]),
    :time_sec => mean => :mean_time)

# Plot bar chart
@df agg_df groupedbar(
    :seq_len_cat,
    :mean_time,
    group = :implementation,
    bar_position = :dodge,
    legend = :top,
    xlabel = "Sequence Length",
    ylabel = "Execution Time (s)",
    title = "Execution Time by Sequence Length and Implementation",
    alpha = 0.7,
    lw = 0.5,
)

# Prepare for dot overlay
seq_lens = levels(df.seq_len_cat)
n_groups = length(implementations)
width = 0.8  # Total width of grouped bars
bar_width = width / n_groups

# Map seq_len_cat to integer positions
xvals = Dict(seq => i for (i, seq) in enumerate(seq_lens))

# Compute aligned x coordinates for dots
x_points = Float64[]
for row in eachrow(df)
    base_x = xvals[string(row.seq_len)]
    group_idx = findfirst(==(row.implementation), implementations)
    x_pos = base_x - width / 2 + (group_idx - 0.5) * bar_width
    jitter = (rand() - 0.5) * bar_width * 0.6  # Controlled jitter
    push!(x_points, x_pos + jitter)
end

# Overlay scatter plot of individual runs
scatter!(x_points .- 0.5, df.time_sec,
    group = df.implementation,
    ms = 3,
    alpha = 0.8,
    markerstrokewidth = 0.1,
    label = "",
)
