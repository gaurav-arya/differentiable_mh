#text # Analyzing how the DMH performs for targets with varying support

##cell
cd(dirname(@__DIR__))  #hide
push!(LOAD_PATH, @__DIR__)  #hide
push!(LOAD_PATH, joinpath(dirname(@__DIR__), "Analysis"))  #hide
push!(LOAD_PATH, joinpath(dirname(dirname(@__DIR__)), "src")) # (DMH)  #hide

using DataContaminationProblem 
using DataFrames
using DelimitedFiles
using DifferentiableMH
using Distributions
using PDMats
using LinearAlgebra
using LogDensityProblems
using MCMCChains
using StochasticAD
using Statistics
using Turing
using ProgressMeter
using CairoMakie
import Random
import Analysis: take_samples, get_primal_chain_slim, get_raw_chain_slim, get_primal_timing, get_derivative_timing
import Analysis

# Set up StochasticAD to use importance sampled pruning
weights_st_backend = StrategyWrapperFIsBackend(
    PrunedFIsBackend(Val(:weights)), StochasticAD.StraightThroughStrategy())
dictfis_st_backend = StrategyWrapperFIsBackend(
    DictFIsBackend(), StochasticAD.StraightThroughStrategy())
;

##cell

function model_pdf(x, θ)
    r2 = LinearAlgebra.norm_sqr(x)
    out = (1 - θ) * StochasticAD.propagate(r2 -> r2 < 1, r2) + θ * StochasticAD.propagate(r2 -> 9 < r2 < 25, r2)
    out
end

##cell
# Run 1D chain
Random.seed!(20240408);
Random.seed!(StochasticAD.RNG, 20240528);
problem = make_data_contamination_problem(model_pdf, [0.0], 1000; f = LinearAlgebra.norm_sqr, burn_in=500, theta=1e-6);
data = get_raw_chain_slim(problem; target="primal", alg_id="pruning_mvd", get_chain = Analysis._get_chain_full);

##cell
#text Trajectory 1D
fig = Figure(size=(500,250))
# ax_inference = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2", ylabelrotation=0,
#     xlabelsize=16, ylabelsize=16, aspect=1, width=350,height=350)
ax_inference = Axis(fig[1, 1], xlabel = L"\text{Iterations}", ylabelrotation=0,
    xlabelsize=16, ylabelsize=16, ylabel=L"x") #, xticks=(0:50, map(i -> (i % 5 == 0 || (i == 1)) ? repr(i) : "", 1:51)))

# plot primal
primals_1d = map(x -> StochasticAD.value(x[1]), data.chain[500:1000])
scatterlines!(ax_inference, 500:1000, primals_1d; color = (:black, 1.0), markersize=3)

# plot perturbations
Δs_1d = map(x -> StochasticAD.perturbations(x[1])[1].Δ, data.chain[500:1000])
ws_1d = map(x -> StochasticAD.perturbations(x[1])[1].weight, data.chain[500:1000])
max_abs_ws_1d = maximum(abs.(ws_1d))
scatterlines!(ax_inference, 500:1000, Δs_1d .+ primals_1d; color = [(:red, 2 * w / max_abs_ws_1d) for w in ws_1d], markersize=3)

xlims!(ax_inference, 500, 1000)

elem_primal = LineElement(color = :black, marker = :circle)
elem_residual = LineElement(color = :red, marker = :circle)
axislegend(ax_inference, [elem_primal, elem_residual], [L"\text{MH Chain}", L"\text{DMH Augmentations}"]; backgroundcolor = (:white, 0.8), position=:rb)

fig

##cell
#text Trajectory 2D

Random.seed!(1234);
Random.seed!(StochasticAD.RNG, 1234);

burn_in = 1000
n = 2000

problem = make_data_contamination_problem(model_pdf, [0.0, 0.0], n; f = LinearAlgebra.norm_sqr, burn_in, theta=1e-6)
data = get_raw_chain_slim(problem; target="primal", alg_id="pruning_mvd", get_chain = Analysis._get_chain_full)

##cell
# plot 2d

fig = Figure(size=(400,480))
ax_inference = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2", ylabelrotation=0,
    xlabelsize=16, ylabelsize=16, aspect=1, width=350,height=350) #, xticks=(0:50, map(i -> (i % 5 == 0 || (i == 1)) ? repr(i) : "", 1:51)))

poly!(ax_inference, Circle(Point2f(0,0), 5), color = (:red, 0.2))
poly!(ax_inference, Circle(Point2f(0,0), 3), color = (:white, 1))
poly!(ax_inference, Circle(Point2f(0,0), 1), color = (:black, 0.2))

primals = map(x -> StochasticAD.value.(x), data.chain[burn_in:n])
Δs = map(x -> map(z -> StochasticAD.perturbations(z)[1].Δ, x), data.chain[burn_in:n])
ws = map(x -> StochasticAD.perturbations(x[1])[1].weight, data.chain[burn_in:n])
max_abs_ws = maximum(abs.(ws))
scatterlines!(ax_inference, map(x -> x[1], primals), map(x -> x[2], primals); color = (:black, 1.0), markersize=3)
scatterlines!(ax_inference, map(x -> x[1], primals) .+ map(x -> x[1], Δs), map(x -> x[2], primals) .+ map(x -> x[2], Δs); color = [(:red, 2 * w / max_abs_ws) for w in ws], markersize=3)

elem_primal = LineElement(color = :black, marker = :circle)
elem_residual = LineElement(color = :red, marker = :circle)
elem_original = PolyElement(color = (:black, 0.2))
elem_contaminated = PolyElement(color = (:red, 0.2))
Legend(fig[2,1], [elem_primal, elem_residual, elem_original, elem_contaminated], [L"\text{MH Chain}", L"\text{DMH Augmentations}", L"\text{Original density}", L"\text{Contamination to density}"]; backgroundcolor = (:white, 0.8), nbanks = 2)

fig

##cell
#text Collect variance asymptotics w.r.t. theta 

means = []
stds = []

thetas = vcat([1e-6], 0.01:0.01:0.2)
nruns = 100

for theta in thetas 
    @show theta
    ests = []
    for i in 1:nruns
        problem = make_data_contamination_problem(model_pdf, [0.0], 1000; f = x -> norm(x)^2, burn_in=500, theta = theta)
        data = get_raw_chain_slim(problem; target="primal", alg_id="pruning_mvd", get_chain = Analysis._get_chain_full)
        primals = map(x -> StochasticAD.value.(x), data.chain[500:1000])
        est = StochasticAD.delta(data.ret)
        push!(ests, est)
    end
    push!(means, mean(ests))
    push!(stds, std(ests))
end


##cell
#text Comparison to score

using ForwardDiff

function score(x, theta; baseline = 0)
    return (norm(x)^2 - baseline) * ForwardDiff.derivative(theta -> log(model_pdf(x, theta)), theta)
end

score_means = []
score_stds = []

for theta in thetas 
    ests = []
    for i in 1:nruns
        problem = make_data_contamination_problem(model_pdf, [0.0], 1000; f = LinearAlgebra.norm_sqr, burn_in=500, theta = theta)
        data = get_raw_chain_slim(problem; target="primal", alg_id="pruning_mvd")
        primals = map(x -> StochasticAD.value(x[1]), data.chain[500:1000])
        baseline = mean(map(x -> norm(x)^2, primals))
        scores = map(x -> (z = score(x, theta; baseline); if isnan(z) error(x) end; z), primals)
        est = mean(scores[length(scores) ÷ 2:end])
        push!(ests, est)
    end
    push!(score_means, mean(ests))
    push!(score_stds, std(ests))
end

means
score_means

##cell
#text Plot variance comparison

fig = Figure(size=(500,180))
ax = Axis(fig[1, 1], xlabel = L"\theta", ylabel = L"\text{Variance}") #, yscale =log10)

score_vars = score_stds.^2
dmh_vars = stds.^2 
score_var_errs = map(v -> sqrt(2 / (nruns - 1)) * v, score_vars) .* 1.96
dmh_var_errs = map(v -> sqrt(2 / (nruns - 1)) * v, dmh_vars) .* 1.96

scatterlines!(ax, thetas[3:end], score_vars[3:end]; label = "Likelihood Ratio", color = :blue)
band!(ax, thetas[3:end], score_vars[3:end] .- score_var_errs[3:end], score_vars[3:end] .+ score_var_errs[3:end]; color = (:blue, 0.2))
scatterlines!(ax, thetas, dmh_vars; label = "DMH", color = :orange)
band!(ax, thetas, dmh_vars .- dmh_var_errs, dmh_vars .+ dmh_var_errs; color = (:orange, 0.2))

axislegend(ax)

fig

#-
##cell
# Publication figure
fig = Figure(size=(850,450))

# A
ax_inference = Axis(fig[1,1], xlabel = "Iterations", ylabelrotation=0,
    xlabelsize=16, ylabelsize=16, ylabel=L"x") #, xticks=(0:50, map(i -> (i % 5 == 0 || (i == 1)) ? repr(i) : "", 1:51)))
scatterlines!(ax_inference, 500:1000, primals_1d; color = (:black, 1.0), markersize=3)
scatterlines!(ax_inference, 500:1000, Δs_1d .+ primals_1d; color = [(:red, 2 * w / max_abs_ws_1d) for w in ws_1d], markersize=3)
xlims!(ax_inference, 500, 1000)
Label(fig[1,1,TopLeft()], "A", font=:bold, halign = :left)

# B
ax_inference = Axis(fig[1:2,2], xlabel = L"x_1", ylabel = L"x_2", ylabelrotation=0,
    xlabelsize=16, ylabelsize=16, aspect=1, width=350,height=350) #, xticks=(0:50, map(i -> (i % 5 == 0 || (i == 1)) ? repr(i) : "", 1:51)))

poly!(ax_inference, Circle(Point2f(0,0), 5), color = (:red, 0.2))
poly!(ax_inference, Circle(Point2f(0,0), 3), color = (:white, 1))
poly!(ax_inference, Circle(Point2f(0,0), 1), color = (:black, 0.2))

scatterlines!(ax_inference, map(x -> x[1], primals), map(x -> x[2], primals); color = (:black, 1.0), markersize=3)
scatterlines!(ax_inference, map(x -> x[1], primals) .+ map(x -> x[1], Δs), map(x -> x[2], primals) .+ map(x -> x[2], Δs); color = [(:red, 2 * w / max_abs_ws) for w in ws], markersize=3)

Label(fig[1:2,2,TopLeft()], "B", font=:bold, halign = :left)

# C
ax = Axis(fig[2,1], xlabel = L"\theta", ylabel = "Variance") #, yscale =log10)
scatterlines!(ax, thetas[3:end], score_vars[3:end]; label = "Likelihood ratio", color = :blue, marker=:diamond)
band!(ax, thetas[3:end], score_vars[3:end] .- score_var_errs[3:end], score_vars[3:end] .+ score_var_errs[3:end]; color = (:blue, 0.2))
scatterlines!(ax, thetas, dmh_vars; label = "DMH", color = :red, marker=:rect)
band!(ax, thetas, dmh_vars .- dmh_var_errs, dmh_vars .+ dmh_var_errs; color = (:red, 0.2))
Label(fig[2,1,TopLeft()], "C", font=:bold, halign = :left)

colgap!(fig.layout, 30)

save("../assets/data_contamination.pdf", fig);

##cell
#text Timing data for the estimator
Random.seed!(1234);
Random.seed!(StochasticAD.RNG, 1234);
problem = make_data_contamination_problem(model_pdf, [0.0, 0.0], 10000; f = LinearAlgebra.norm_sqr, burn_in = 10000, theta=1e-6)
primal_timing = get_primal_timing(problem; target="primal")
derivative_timing = get_derivative_timing(
    problem; target="primal", backend=weights_st_backend)
dictfis_timing = get_derivative_timing(
    problem; target="primal", backend=dictfis_st_backend)

(;
    primal_ns = primal_timing.ns,
    derivative_ns = derivative_timing.ns,
    derivative_dictfis_ns = dictfis_timing.ns,
    ratio = derivative_timing.ns / primal_timing.ns,
    dictfis_ratio = dictfis_timing.ns / primal_timing.ns,
)
