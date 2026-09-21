"""Write the PPCA counterfactual-coupling ablation as a LaTeX table.

Usage:

    JULIA_DEPOT_PATH=/tmp/dmh_julia_depot:/home/rubense/.julia \
    /opt/julia-1.10.2/bin/julia --project=experiments \
        experiments/mcvae/ppca_diagnostic/make_ppca_supplement_table.jl \
        experiments/mcvae/ppca_diagnostic/ppca_results.csv \
        experiments/mcvae/ppca_diagnostic/ppca_coupling_supplement.tex

The table contains the three DMH-all future constructions. The coupling
ablations are for the supplement and are not part of the publication figure.
"""

using Printf

struct CouplingRow
    K::Int
    method::String
    mean::Float64
    stderr::Float64
    variance::Float64
    expensive_evals::Float64
    variance_x_expensive_evals::Float64
end

const TABLE_METHODS = [
    (name="DMH_all_MaximumReflection", label="maximum reflection"),
    (name="DMH_all_SynchronousCRN", label="CRN"),
    (name="DMH_all_IndependentFutures", label="independent"),
]

function read_summary(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("empty PPCA results CSV: $(path)")
    header = split(first(lines), ',')
    columns = Dict(name => findfirst(==(name), header) for name in header)
    required = [
        "K", "method", "mean", "stderr", "variance", "expensive_evals",
        "variance_x_expensive_evals",
    ]
    all(haskey(columns, name) for name in required) ||
        error("PPCA results CSV is missing one of $(required)")

    rows = CouplingRow[]
    for line in Iterators.drop(lines, 1)
        isempty(strip(line)) && continue
        fields = split(line, ',')
        method = fields[columns["method"]]
        method in (spec.name for spec in TABLE_METHODS) || continue
        push!(rows, CouplingRow(
            parse(Int, fields[columns["K"]]),
            method,
            parse(Float64, fields[columns["mean"]]),
            parse(Float64, fields[columns["stderr"]]),
            parse(Float64, fields[columns["variance"]]),
            parse(Float64, fields[columns["expensive_evals"]]),
            parse(Float64, fields[columns["variance_x_expensive_evals"]]),
        ))
    end
    return rows
end

function find_row(rows, K, method)
    matches = filter(row -> row.K == K && row.method == method, rows)
    length(matches) == 1 ||
        error("expected one $(method) row for K=$(K), found $(length(matches))")
    return first(matches)
end

function format_mean_se(row::CouplingRow)
    uncertainty_digits = round(Int, 1000 * row.stderr)
    return @sprintf("%.3f(%d)", row.mean, uncertainty_digits)
end

function write_table(path::AbstractString, rows)
    horizons = sort(unique(row.K for row in rows))
    isempty(horizons) && error("no PPCA coupling rows found")
    for K in horizons, spec in TABLE_METHODS
        find_row(rows, K, spec.name)
    end

    open(path, "w") do io
        println(io, raw"\begin{table}[t]")
        println(io, raw"\centering")
        println(io, raw"\caption{Comparison of different couplings of DMH-all in the PPCA experiment of \cref{sec:mcvae}. Each $K$ uses the same augmented chain.}")
        println(io, raw"\label{tab:ppca-coupling-ablation}")
        println(io, raw"\begin{tabular}{ S[table-format=2.0] l S[table-format=-1.3(2)] S[scientific-notation=true, exponent-product=\cdot, table-format=1.3e2] S[table-format=5.1] S[scientific-notation=true, exponent-product=\cdot, table-format=1.3e2] }")
        println(io, raw"\toprule")
        println(io, raw"\multicolumn{1}{c}{$K$} & Coupling & \multicolumn{1}{c}{Mean (SE)} & \multicolumn{1}{c}{Variance} & \multicolumn{1}{c}{Target evals.} & \multicolumn{1}{c}{Variance $\times$ evals.} ", '\\', '\\')
        println(io, raw"\midrule")
        for K in horizons
            for spec in TABLE_METHODS
                row = find_row(rows, K, spec.name)
                values = [
                    string(K),
                    spec.label,
                    format_mean_se(row),
                    @sprintf("%.3e", row.variance),
                    @sprintf("%.1f", row.expensive_evals),
                    @sprintf("%.3e", row.variance_x_expensive_evals),
                ]
                println(io, join(values, " & "), ' ', '\\', '\\')
            end
        end
        println(io, raw"\bottomrule")
        println(io, raw"\end{tabular}")
        println(io, raw"\end{table}")
    end
    return path
end

csv_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "ppca_results.csv")
output_path = length(ARGS) >= 2 ?
    ARGS[2] : joinpath(dirname(csv_path), "ppca_coupling_supplement.tex")
write_table(output_path, read_summary(csv_path))
println("Wrote $(output_path)")
