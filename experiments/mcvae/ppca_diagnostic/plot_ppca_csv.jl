"""Render the PPCA publication summary figure from ``ppca_results.csv``.

The CSV contains summary moments rather than the individual repetitions, so
the estimator panels use means with two-standard-error bars.  The main axes
focus on the low-variance estimators and use insets to retain the scale of
ordinary REINFORCE.

Example:

    JULIA_DEPOT_PATH=/tmp/dmh_julia_depot:/home/rubense/.julia \
    /opt/julia-1.10.2/bin/julia --project=experiments \
        experiments/mcvae/ppca_diagnostic/plot_ppca_csv.jl \
        experiments/mcvae/ppca_diagnostic/ppca_results.csv
"""

using CairoMakie
using Statistics

struct SummaryRow
    K::Int
    method::String
    mean::Float64
    variance::Float64
    stderr::Float64
    target_evaluations::Float64
end

struct SampleRow
    K::Int
    method::String
    repetition::Int
    estimate::Float64
    gradient::Float64
end

const METHOD_SPECS = [
    (name="REINFORCE_ordinary", label="LR\n(N=1)", legend="LR (N=1)", color="#4C78A8"),
    (name="Thin_leave_one_out_N10", label="LR + control\n(N=10)", legend="LR + control (N=10)", color="#F58518"),
    (name="DMH_all_MaximumReflection", label="DMH-all\n(N=1)", legend="DMH-all (N=1)", color="#54A24B"),
    (name="DMH_one_MaximumReflection_Pruned", label="DMH-pruned\n(N=1)", legend="DMH-pruned (N=1)", color="#E45756"),
]

function read_summary(path)
    lines = readlines(path)
    isempty(lines) && error("empty PPCA results CSV: $(path)")
    header = split(first(lines), ',')
    columns = Dict(name => findfirst(==(name), header) for name in header)
    required = ["K", "method", "mean", "variance", "stderr", "expensive_evals"]
    all(haskey(columns, name) for name in required) ||
        error("PPCA results CSV is missing one of $(required)")

    rows = SummaryRow[]
    for line in Iterators.drop(lines, 1)
        isempty(strip(line)) && continue
        fields = split(line, ',')
        method = fields[columns["method"]]
        K = parse(Int, fields[columns["K"]])
        if method == "Analytical_PPCA_reference"
            continue
        end
        push!(rows, SummaryRow(
            K,
            method,
            parse(Float64, fields[columns["mean"]]),
            parse(Float64, fields[columns["variance"]]),
            parse(Float64, fields[columns["stderr"]]),
            parse(Float64, fields[columns["expensive_evals"]]),
        ))
    end
    return rows
end

function read_samples(path)
    lines = readlines(path)
    isempty(lines) && error("empty PPCA samples CSV: $(path)")
    header = split(first(lines), ',')
    columns = Dict(name => findfirst(==(name), header) for name in header)
    required = ["K", "method", "repetition", "estimate", "gradient"]
    all(haskey(columns, name) for name in required) ||
        error("PPCA samples CSV is missing one of $(required)")

    samples = SampleRow[]
    for line in Iterators.drop(lines, 1)
        isempty(strip(line)) && continue
        fields = split(line, ',')
        push!(samples, SampleRow(
            parse(Int, fields[columns["K"]]),
            fields[columns["method"]],
            parse(Int, fields[columns["repetition"]]),
            parse(Float64, fields[columns["estimate"]]),
            parse(Float64, fields[columns["gradient"]]),
        ))
    end
    return samples
end

function read_references(path)
    lines = readlines(path)
    header = split(first(lines), ',')
    columns = Dict(name => findfirst(==(name), header) for name in header)
    references = Dict{Int,Float64}()
    for line in Iterators.drop(lines, 1)
        isempty(strip(line)) && continue
        fields = split(line, ',')
        method = fields[columns["method"]]
        method == "Analytical_PPCA_reference" || continue
        K = parse(Int, fields[columns["K"]])
        references[K] = parse(Float64, fields[columns["mean"]])
    end
    return references
end

function find_row(rows, K, method)
    matches = filter(row -> row.K == K && row.method == method, rows)
    length(matches) == 1 || error("expected one $(method) row for K=$(K), found $(length(matches))")
    return first(matches)
end

gradient_value(row) = -row.mean

function horizon_offsets(horizons)
    offsets = length(horizons) == 1 ? [0.0] : collect(range(-0.18, 0.18; length=length(horizons)))
    return Dict(K => offsets[index] for (index, K) in enumerate(horizons))
end

function horizon_markers(horizons)
    available = [:circle, :diamond, :utriangle, :square, :star5]
    length(horizons) <= length(available) || error("too many K values for publication markers")
    return Dict(K => available[index] for (index, K) in enumerate(horizons))
end

function horizon_marker_labels(horizons)
    available = ["○", "◇", "△", "□", "☆"]
    length(horizons) <= length(available) || error("too many K values for publication markers")
    return ["$(available[index]) K=$(K)" for (index, K) in enumerate(horizons)]
end

function gradient_limits(rows, references, horizons; full=false)
    values = Float64[]
    for K in horizons
        push!(values, -references[K])
        for spec in (full ? METHOD_SPECS : METHOD_SPECS[2:end])
            row = find_row(rows, K, spec.name)
            center = gradient_value(row)
            push!(values, center - 2row.stderr)
            push!(values, center + 2row.stderr)
        end
    end
    low, high = extrema(values)
    padding = max((full ? 0.08 : 0.14) * (high - low), full ? 1.0 : 0.08)
    return low - padding, high + padding
end

function draw_gradient_points!(axis, rows, references, horizons, lower, upper;
                               markersize=13, linewidth=1.6, whiskerwidth=7)
    offsets = horizon_offsets(horizons)
    markers = horizon_markers(horizons)
    for K in horizons
        marker = markers[K]
        for (position, spec) in enumerate(METHOD_SPECS)
            row = find_row(rows, K, spec.name)
            center = gradient_value(row)
            lower <= center <= upper || continue
            x = position + offsets[K]
            CairoMakie.errorbars!(
                axis, [x], [center], [2row.stderr];
                color=spec.color, linewidth=linewidth, whiskerwidth=whiskerwidth,
            )
            CairoMakie.scatter!(
                axis, [x], [center];
                color=spec.color, marker=marker,
                strokecolor=:white, strokewidth=0.7, markersize=markersize,
            )
        end
    end
end

function sample_values(samples, K, spec)
    return filter(isfinite, [
        sample.gradient for sample in samples
        if sample.K == K && sample.method == spec.name
    ])
end

function sample_gradient_limits(samples, references, horizons; full=false)
    values = Float64[]
    for K in horizons
        push!(values, -references[K])
        for spec in (full ? METHOD_SPECS : METHOD_SPECS[2:end])
            append!(values, sample_values(samples, K, spec))
        end
    end
    isempty(values) && error("cannot make PPCA plot without finite samples")
    if full && length(values) > 2
        sorted_values = sort(values)
        low = quantile(sorted_values, 0.01)
        high = quantile(sorted_values, 0.99)
    else
        low, high = extrema(values)
    end
    padding = max((full ? 0.08 : 0.14) * (high - low), full ? 1.0 : 0.08)
    return low - padding, high + padding
end

function draw_gradient_boxes!(axis, samples, horizons, lower, upper;
                              markersize=8, boxwidth=0.22,
                              skip_names=Set{String}())
    offsets = horizon_offsets(horizons)
    markers = horizon_markers(horizons)
    for K in horizons
        for (position, spec) in enumerate(METHOD_SPECS)
            spec.name in skip_names && continue
            values = sample_values(samples, K, spec)
            isempty(values) && continue
            x = position + offsets[K]
            CairoMakie.boxplot!(
                axis, fill(x, length(values)), values;
                width=boxwidth,
                color=(spec.color, 0.55),
                strokecolor=spec.color,
                strokewidth=1.0,
                show_outliers=false,
            )
            center = mean(values)
            lower <= center <= upper || continue
            CairoMakie.scatter!(axis, [x], [center];
                                color=spec.color, marker=markers[K],
                                strokecolor=:white, strokewidth=0.7,
                                markersize=markersize)
        end
    end
end

function fill_inset!(axis, xlimits, ylimits)
    xlow, xhigh = xlimits
    ylow, yhigh = ylimits
    points = CairoMakie.Point2f[
        CairoMakie.Point2f(xlow, ylow),
        CairoMakie.Point2f(xhigh, ylow),
        CairoMakie.Point2f(xhigh, yhigh),
        CairoMakie.Point2f(xlow, yhigh),
    ]
    CairoMakie.poly!(axis, points; color=(:white, 1.0), strokecolor=:transparent)
end

function make_plot(rows, references, output_directory; samples=nothing)
    CairoMakie.activate!()
    horizons = sort(unique(row.K for row in rows))
    isempty(horizons) && error("expected at least one K value")
    all(haskey(references, K) for K in horizons) || error("missing analytical reference")
    using_samples = samples !== nothing
    if using_samples
        all(!isempty(sample_values(samples, K, spec))
            for K in horizons for spec in METHOD_SPECS) ||
            error("PPCA samples CSV is missing a plotted method/horizon")
    end

    # Match the publication typography used by data_contamination.  The PDF is
    # vector output, so its physical size is independent of raster DPI.
    figure = CairoMakie.Figure(size=(850, 360), figure_padding=(10, 10, 10, 10))

    lower_limits = using_samples ?
        sample_gradient_limits(samples, references, horizons) :
        gradient_limits(rows, references, horizons)
    full_gradient_limits = using_samples ?
        sample_gradient_limits(samples, references, horizons; full=true) :
        gradient_limits(rows, references, horizons; full=true)
    labels = [spec.label for spec in METHOD_SPECS]

    axis = CairoMakie.Axis(
        figure[1, 1],
        ylabel="decoder-bias gradient",
        xticks=(1:length(METHOD_SPECS), labels),
        xticklabelrotation=π / 4,
        xticklabelsize=11,
        yticklabelsize=12,
        ylabelsize=16,
    )
    # Axis.backgroundcolor only covers the data rectangle.  Put an opaque
    # layout element behind the inset as well, so its y-axis decorations do
    # not show the main plot through them.
    CairoMakie.Box(
        figure[1, 1],
        width=125,
        height=100,
        halign=:left,
        valign=:top,
        alignmode=CairoMakie.Outside(8, 0, 0, 12),
        tellwidth=false,
        tellheight=false,
        color=:white,
        strokecolor=:transparent,
    )
    inset = CairoMakie.Axis(
        figure[1, 1],
        width=125,
        height=100,
        halign=:left,
        valign=:top,
        alignmode=CairoMakie.Outside(8, 0, 0, 12),
        tellwidth=false,
        tellheight=false,
        backgroundcolor=(:white, 1.0),
        xticks=(1:length(METHOD_SPECS), labels),
        xticklabelsvisible=false,
        xticksvisible=false,
        xlabelvisible=false,
        yticklabelsize=9,
        ylabelvisible=false,
        xgridvisible=false,
        ygridvisible=false,
    )
    for current_axis in (axis, inset)
        CairoMakie.xlims!(current_axis, 0.45, length(METHOD_SPECS) + 0.55)
        limits = current_axis === axis ? lower_limits : full_gradient_limits
        CairoMakie.ylims!(current_axis, limits...)
    end
    fill_inset!(inset, (0.45, length(METHOD_SPECS) + 0.55), full_gradient_limits)
    if using_samples
        draw_gradient_boxes!(axis, samples, horizons, lower_limits...;
                             markersize=8, boxwidth=0.22,
                             skip_names=Set([METHOD_SPECS[1].name]))
        draw_gradient_boxes!(inset, samples, horizons, full_gradient_limits...;
                             markersize=7, boxwidth=0.20)
    else
        draw_gradient_points!(axis, rows, references, horizons, lower_limits...;
                              markersize=13, linewidth=1.6, whiskerwidth=7)
        draw_gradient_points!(inset, rows, references, horizons, full_gradient_limits...;
                              markersize=11, linewidth=1.3, whiskerwidth=6)
    end
    CairoMakie.hlines!(axis, [-references[K] for K in horizons]; color=:black, linestyle=:dash, linewidth=1.0)
    CairoMakie.hlines!(inset, [-references[K] for K in horizons]; color=:black, linestyle=:dash, linewidth=0.8)
    CairoMakie.Label(figure[1, 1, CairoMakie.TopLeft()], "A";
                     font=:bold, fontsize=18, halign=:left)
    CairoMakie.text!(
        axis,
        join(horizon_marker_labels(horizons), "   ");
        position=(length(METHOD_SPECS) + 0.45, last(lower_limits) - 0.02 * (last(lower_limits) - first(lower_limits))),
        align=(:right, :top),
        fontsize=11,
    )

    cost_axis = CairoMakie.Axis(
        figure[1, 2],
        xlabel="log-density + gradient\nevaluations per estimate",
        ylabel="variance",
        xscale=log10,
        xlabelsize=15,
        ylabelsize=16,
        xticklabelsize=11,
        yticklabelsize=11,
        topspinevisible=false,
        rightspinevisible=false,
    )
    close_variances = [
        find_row(rows, K, spec.name).variance
        for K in horizons for spec in METHOD_SPECS[2:end]
    ]
    cost_low, cost_high = extrema(close_variances)
    cost_padding = max(0.20 * (cost_high - cost_low), 0.002)
    CairoMakie.ylims!(cost_axis, max(0.0, cost_low - cost_padding), cost_high + cost_padding)
    markers = horizon_markers(horizons)
    for spec in METHOD_SPECS[2:end]
        costs = [find_row(rows, K, spec.name).target_evaluations for K in horizons]
        variances = [find_row(rows, K, spec.name).variance for K in horizons]
        CairoMakie.lines!(cost_axis, costs, variances; color=spec.color, linewidth=1.5, label=spec.legend)
        for (index, K) in enumerate(horizons)
            CairoMakie.scatter!(cost_axis, [costs[index]], [variances[index]];
                                color=spec.color, marker=markers[K],
                                markersize=11)
        end
    end
    CairoMakie.axislegend(cost_axis, position=:lt, nbanks=1, labelsize=10)
    CairoMakie.Label(figure[1, 2, CairoMakie.TopLeft()], "B";
                     font=:bold, fontsize=18, halign=:left)

    mkpath(output_directory)
    CairoMakie.save(joinpath(output_directory, "mcvae_ppca.pdf"), figure)
    return figure
end

csv_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "ppca_results.csv")
output_directory = length(ARGS) >= 2 ? ARGS[2] : dirname(csv_path)
samples_path = length(ARGS) >= 3 ? ARGS[3] : joinpath(dirname(csv_path), "ppca_samples.csv")
rows = read_summary(csv_path)
references = read_references(csv_path)
samples = isfile(samples_path) ? read_samples(samples_path) : nothing
make_plot(rows, references, output_directory; samples=samples)
