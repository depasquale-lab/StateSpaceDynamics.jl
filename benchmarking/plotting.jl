using Pkg
Pkg.activate("benchmarking")
using CSV
using DataFrames
using Plots
using Statistics: median, quantile
using Measures

function plot_lds_benchmark(fp::AbstractString, sp::AbstractString="benchmarking/results/lds_benchmark_results.svg")
    df = CSV.read(fp, DataFrame)
   
    # Filter valid combinations (obs_dim >= latent_dim)
    df_filtered = filter(row -> row.obs_dim >= row.latent_dim, df)
   
    # Group and aggregate
    gb = groupby(df_filtered, [:implementation, :latent_dim, :obs_dim, :seq_len])
    agg = combine(gb, :time_sec => median => :median_time)
   
    # Get unique values
    lvals = sort(unique(agg.latent_dim))  # columns (x-axis)
    ovals = sort(unique(agg.obs_dim))     # rows (y-axis)
    impls = sort(unique(agg.implementation))

    # Calculate global axis limits
    x_min, x_max = extrema(agg.seq_len)
    y_min, y_max = extrema(agg.median_time)

    # Use actual sequence length values as ticks
    x_ticks = sort(unique(agg.seq_len))
   
    subplots = Plots.Plot[]
   
    # Loop: obs_dim varies by row (i), latent_dim varies by column (j)
    for (i, o) in enumerate(reverse(ovals))  # Start from highest obs_dim (top row)
        for (j, l) in enumerate(lvals)       # Left to right latent_dim
            # Skip invalid combinations
            if o < l
                push!(subplots, plot(framestyle=:none, showaxis=false))
                continue
            end
           
            # Get data for this latent_dim/obs_dim combination
            panel = @view agg[(agg.latent_dim .== l) .& (agg.obs_dim .== o), :]
           
            if isempty(panel)
                push!(subplots, plot(framestyle=:none, showaxis=false))
                continue
            end
           
            # Only show x-label on bottom row, y-label on left column
            xlabel_text = (i == length(ovals)) ? "T" : ""
            ylabel_text = (j == 1) ? "Time (s)" : ""
           
            p = plot(
                xlabel = xlabel_text,
                ylabel = ylabel_text,
                framestyle = :box,
                grid = :y,
                minorgrid = true,
                legend = false,  # Turn off individual legends
                xlims = (x_min * 0.9, x_max * 1.1),
                ylims = (y_min * 0.9, y_max * 1.1),
                xticks = x_ticks
            )
           
            # Plot a line for each implementation
            for impl in impls
                impl_data = @view panel[panel.implementation .== impl, :]
                if !isempty(impl_data)
                    plot!(p, impl_data.seq_len, impl_data.median_time,
                          label=impl, marker=:circle, linewidth=2)
                end
            end
           
            push!(subplots, p)
        end
    end
   
    # Create main plot: rows = obs_dim, cols = latent_dim
    plt = plot(subplots...;
               layout=(length(ovals), length(lvals)),
               size=(300*length(lvals), 250*length(ovals)))

    # Create legend plot
    legend_plot = plot(framestyle=:none, showaxis=false, size=(150, 250*length(ovals)))
    for impl in impls
        plot!(legend_plot, [NaN], [NaN], label=impl, linewidth=2, marker=:circle)
    end
   
    # Combine with legend on the right
    final_plt = plot(plt, legend_plot, layout=@layout([a{0.85w} b{0.15w}]))
   
    plot!(final_plt; tickfontsize=11, guidefontsize=13, legendfontsize=12)
    savefig(sp)
   
    return final_plt
end

using CSV, DataFrames, Plots, Statistics, Measures

function plot_hmm_benchmark(fp::AbstractString, sp::AbstractString="benchmarking/results/hmm_benchmark_results.png")
    df = CSV.read(fp, DataFrame)
    df = filter(row -> !isnan(row.time_sec), df)

    gb = groupby(df, [:implementation, :num_states, :seq_len])
    agg = combine(gb, :time_sec => median => :median_time)

    state_vals = sort(unique(agg.num_states))
    impls = sort(unique(agg.implementation))

    x_min, x_max = extrema(agg.seq_len)
    y_min, y_max = extrema(agg.median_time)
    x_ticks = sort(unique(agg.seq_len))

    subplots = Plots.Plot[]

    # create per-column panels with NO x-label (we will add a single global xlabel)
    for (j, ns) in enumerate(state_vals)
        panel = @view agg[agg.num_states .== ns, :]
        if isempty(panel)
            push!(subplots, plot(framestyle=:none, showaxis=false))
            continue
        end

        p = plot(
            xlabel = "",                  # WARNING: keep empty so there is no per-panel xlabel
            ylabel = (j == 1) ? "Time (s)" : "",   # optional local ylabel only on first column
            title = "States = $ns",
            framestyle = :box,
            grid = :y,
            minorgrid = true,
            legend = false,
            xlims = (x_min * 0.9, x_max * 1.1),
            ylims = (y_min * 0.9, y_max * 1.1),
            xticks = x_ticks
        )

        for impl in impls
            impl_data = @view panel[panel.implementation .== impl, :]
            if !isempty(impl_data)
                plot!(p, impl_data.seq_len, impl_data.median_time, label = impl, marker = :circle, linewidth = 2)
            end
        end
        push!(subplots, p)
    end

    # Compose the grid of subplots (no global labels here)
    plt = plot(subplots...; layout = (1, length(state_vals)),
               size = (350 * length(state_vals), 300),
               left_margin = 18mm, bottom_margin = 18mm)

    # Legend panel
    legend_plot = plot(framestyle = :none, showaxis = false, size = (150, 300))
    for impl in impls
        plot!(legend_plot, [NaN], [NaN], label = impl, linewidth = 2, marker = :circle)
    end

    # Combine with legend and *set the global labels on the combined plot*
    final_plt = plot(plt, legend_plot,
                     layout = @layout([a{0.85w} b{0.15w}]),
                     left_margin = 20mm, bottom_margin = 20mm,
                     xlabel = "Sequence Length (T)",   # ONE global xlabel
                     ylabel = "Time (s)")              # ONE global ylabel (optional)

    # Tidy fonts, etc.
    plot!(final_plt; tickfontsize = 11, guidefontsize = 13, legendfontsize = 12)
    savefig(sp)
    return final_plt
end


# Usage:
plot_hmm_benchmark("benchmarking/results/hmm_benchmark_results.csv")
plot_lds_benchmark("benchmarking/results/lds_benchmark_results.csv")

