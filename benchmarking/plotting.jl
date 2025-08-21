using Pkg
Pkg.activate("benchmarking")
using CSV
using DataFrames
using Plots
using Statistics: median, quantile

function plot_lds_benchmark(fp::AbstractString, sp::AbstractString="benchmarking/results/lds_benchmark_results.png")
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

plot_lds_benchmark("benchmarking/results/lds_benchmark_results.csv")