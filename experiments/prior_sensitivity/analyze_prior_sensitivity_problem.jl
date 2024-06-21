#text # Analyzing prior sensitivity 

##cell
cd(dirname(@__DIR__))
push!(LOAD_PATH, @__DIR__)
push!(LOAD_PATH, joinpath(dirname(@__DIR__), "Analysis"))
push!(LOAD_PATH, joinpath(dirname(dirname(@__DIR__)), "src")) # (DMH)  #src

using PriorSensitivityProblem
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
import Analysis: take_samples, get_raw_chain_slim

Random.seed!(20240408);
Random.seed!(StochasticAD.RNG, 20240528);

# Monkey patch Bijectors for StochasticTriple
Bijectors._eps(::Type{StochasticTriple{T,V,FI}}) where {T,V,FI} = eps(V)

# Set up StochasticAD to use the stochastic derivatives in the paper
backend = StrategyWrapperFIsBackend(PrunedFIsBackend(Val(:wins)), StochasticAD.StraightThroughStrategy())  # aka uniformly pruning MVD
alg = StochasticAD.ForwardAlgorithm(backend)
;

#=
# Introduction
The idea here is to do prior sensitivity analysis for a simple linear regression model, following

> Kallioinen, N., Paananen, T., Bürkner, P.-C. & Vehtari, A. Detecting and diagnosing prior and likelihood sensitivity with power-scaling. Stat Comput 34, 57 (2024).

The analysis scales the prior with an exponent $\alpha$.
Hence, $\alpha > 1$ upweighs the prior while $\alpha < 1$ downweighs it, and if the resulting posterior is
sensitive to the prior, then it should be sensitive to changes in $\alpha$.
Derivatives are taken with respect to $`\log_2 \alpha`$ (according to the paper recommendation) in order to have better
scaling properties of the sensitivity.

The paper uses a more sophisticated metric of sensitivity based on distances between base and perturbed posteriors,
and essentially differentiates that measurement wrt the ECDF of the posteriors.
The main trouble for us with that metric is that its derivative is discontinuous at the reference point; they average both sides in their formula.
Here we will as an illustration consider the derivative of the posterior mean, which is mentioned in the paper
and has been considered previously.

While the paper works out importance sampling estimators for this purpose, we can run it straight away with the DMH!
=#

##cell
#=
# Simple example
Consider first a small hierarchical model as a test case.
```math
\mu \sim \mathsf{N}(0,1), \qquad
\sigma \sim \mathsf{N}^+(2.0,2.5), \qquad
y_i \sim \mathsf{N}(\mu,\sigma^2)
```
We observe demo data according to a `priorsense` vignette.
=#

# priors
@model function demo_model(y; m=0.0)
    μ ~ Normal(m,1)
    σ ~ truncated(Normal(2.0, 2.5); lower=0.0)
    N = length(y)
    for i in 1:N
        y[i] ~ Normal(μ, σ)
    end
end;

# Wrapper for handling Turing models
function make_targetlogpdf(model, args...; kwargs...)
    m = model(args...; kwargs...)
    vi = DynamicPPL.VarInfo(m)
    vi = DynamicPPL.link!!(vi, m)  # transforms to unconstrained space
    function model_logpdf(x, θ)
        vals = DynamicPPL.unflatten(vi, x)
        # FIXME: is there a better way than this really stupid hack to also
        # get the Jacobian term for transformed variables?
        loglik = DynamicPPL.loglikelihood(m, vals)
        logpri = DynamicPPL.logjoint(m, vals) - loglik
        loglik + 2^θ * logpri
    end
end;
untransform(p) = [p[1:end-1]..., exp(p[end])];

#text Example demo data from `priorsense`
obs = [9.5, 10.2, 9.1, 9.1, 10.3, 10.9, 11.7, 10.3, 9.6, 8.6, 9.1,
       11.1, 9.3, 10.5, 9.7, 10.3, 10.0, 9.8, 9.6, 8.3, 10.2, 9.8,
       10.0, 10.0, 9.1];

#text Set up model
# Start from prior means
init = [0.0, 1.071];

model_logpdf = make_targetlogpdf(demo_model,obs);
problem = make_prior_sensitivity_problem(model_logpdf, init, 200000; f = untransform, burn_in=0)
problem.targets["primal"].X(stochastic_triple(problem.settings.p; backend=alg.backend), problem.settings)

# FIXME take_samples does not support vector output :(
# samples = take_samples(problem, discrete_alg_flags = ["pruning"], store_samples = true)

##cell
#=
The derivatives tell us about the relative sensitivity of the parameters.
The paper vignette thinks $\mu$ is too sensitive, but we don't have a normalized metric here for ourselves...
Let's try to adjust the prior on $\mu$ to be closer to the data and see if it has less influence.
=#
model_logpdf = make_targetlogpdf(demo_model,obs; m=mean(obs));
problem = make_prior_sensitivity_problem(model_logpdf, init, 200000; f = untransform, burn_in=0)
problem.targets["primal"].X(stochastic_triple(problem.settings.p; backend=alg.backend), problem.settings)

#text The prior sensitivity was reduced by several orders of magnitude, and the prior is now less influential on the final estimate.

##cell
#=
# Case study: Body fat data

Section 5.1 of the Kallioinen et al. paper considers a linear regression model for body fat percentage, with data from

> Johnson, R. W. Fitting Percentage of Body Fat to Simple Body Measurements. Journal of Statistics Education 4, 6 (1996).

and code at https://github.com/n-kall/powerscaling-sensitivity/tree/master/case-studies/bodyfat

The model is
```math
\begin{gathered}
y_i \sim \mathsf{N}(\mu_i, \sigma^2), \qquad \mu_i = \beta_0 + \sum_{k=1}^{13} \beta_k x_{ik} \\
\beta_0 \sim t_3(0.0,9.2), \qquad \beta_k \sim \mathsf{N}(0,1), \qquad \sigma \sim t_3^+(0,9.2)
\end{gathered}
```
The idea is that the prior for the regression coefficients $`\beta_k`$ are chosen to be "uninformative", although this inadvertently
fails as we will discover in the following analysis.

They use a subset of the observations and covariates, and we will attempt to recreate the case study as closely as possible.
Thus, following their Stan code, we will actually center the covariates and translate the intercept prior according to the response mean.
Even though we cannot (yet) interpret the scale of the sensitivity for the different parameters relative to each other,
we are still able to identify the absence of sensitivity as in the previous example.
=#

#text Load the `bodyfat` data
basepath = dirname(@__DIR__)
raw_data, raw_header = DelimitedFiles.readdlm(joinpath(basepath, "prior_sensitivity/bodyfat.txt"), ';', header = true)
df = DataFrame(raw_data, vec(raw_header))
obs_names = ["wrist", "weight_kg", "thigh", "neck", "knee", "hip", "height_cm", "forearm", "chest", "biceps", "ankle", "age", "abdomen", "siri"]
obs = df[!, obs_names]
μ_obs = mean.(eachcol(obs))
σ_obs = std.(eachcol(obs));

# Prepare a centered data matrix (covariate estimate is unchanged!)
Xd = Matrix(obs[:,1:13]) .- μ_obs[1:13]';

#text Set up the model.
@model function bodyfat(
    X, y;
    prior_scales = 2.5 .* std(y) ./ vec(std(X; dims=1)),
    prior_β0_loc = mean(y)  # assumes centering
)
    βk ~ arraydist(Normal.(0,prior_scales))
    β0 ~ LocationScale(prior_β0_loc,9.2,TDist(3))
    σ ~ truncated(LocationScale(0.0,9.2,TDist(3)); lower=0.0)
    return y ~ MvNormal(β0 .+ X * βk, σ^2 * I)
end;
model_logpdf = make_targetlogpdf(bodyfat, Xd, obs[:,14]; prior_scales = ones(13));

#text Start from zero vector
init = zeros(15);

#=
Set an attempt at a reasonable proposal distribution.
It uses the OLS variance with a step scaling and some knowledge from running NUTS of the σ posterior.
=#
design_matrix = hcat(Xd, ones(nrow(obs)));
vc = cholesky(design_matrix' * design_matrix);
A = PDMat(0.02 .* cat(4.25.^2 .* inv(vc), 0.0375161^2; dims=(1,2)));
proposal = RandomWalkMHProposal{Vector{Float64}}(MvNormal(zero(init), A));

##cell
#=
Run the DMH and display diagnostics for the primal.
It takes a long while to keep the whole history!
=#
function raw_chains_to_summary(n_chains, get_raw, names)
    outputs = @showprogress map(1:n_chains) do _
        raw = get_raw()
        samples = @views reduce(hcat, map(state -> untransform(StochasticAD.value.(state)), raw.chain[(raw.settings.burn_in + 2):end]))'
        deltas = StochasticAD.delta.(raw.ret)
        return samples, deltas
    end
    samples = cat(first.(outputs)..., dims=3)
    deltas = hcat(last.(outputs)...)'
    Chains(samples, names), DataFrame(deltas, names)
end;
problem = make_prior_sensitivity_problem(model_logpdf, init, 350000; f = untransform, proposal, burn_in=100000)
out_primal, out_dual = raw_chains_to_summary(4,
    () -> get_raw_chain_slim(problem; target="primal", alg_id="pruning_uniform_mvd"),
    [obs_names[1:13]; "Intercept(c)"; "σ"]);
GC.gc();  # for people like me with puny computers

out_primal  # prints summary diagnostics

##cell
#=
One covariate stands out: `wrist`.
The results are less stable than one would like, so we could probably do with a better MCMC method,
but the results are the same as those detected by the quantitative metric in the Kallioinen et al. paper.
The values are hard to interpret since we have not accounted for scaling in the sensitivity estimates.

The conclusion in the paper is that the regression coefficient priors have the
wrong scale for the data and thus are unintentionally more informative than desired.
We now specify a model that accounts for the scale of the covariates relative to the response, by instead using
the priors $`\beta_k \sim \mathsf{N}(0, (2.5 s_y/s_{x_k})^2)`$.
=#

model_logpdf2 = make_targetlogpdf(bodyfat, Xd, obs[:,14]; );
problem2 = make_prior_sensitivity_problem(model_logpdf2, init, 350000; f = untransform, proposal, burn_in=100000)
out_primal2, out_dual2 = raw_chains_to_summary(4,
    () -> get_raw_chain_slim(problem2; target="primal", alg_id="pruning_uniform_mvd"),
    [obs_names[1:13]; "Intercept(c)"; "σ"]);
GC.gc();

out_primal2

#-
function primal_plot(before, after, subset; kwargs...)
    μ_before, μ_after = mean(before), mean(after)
    q_before, q_after = quantile(before; q=[0.025, 0.975]), quantile(after; q=[0.025, 0.975])
    ix = 1:length(subset)
    f = Figure(size=(350,450))
    ax = Axis(f[1,1]; yticks=(ix, string.(μ_before[subset,1])), yreversed=true, kwargs...)
    dodge = 0.2

    rangebars!(ax, ix .- dodge, q_before[subset,2], q_before[subset,3]; direction=:x)
    scatter!(ax, μ_before[subset,2], ix .- dodge; markersize=12)

    rangebars!(ax, ix .+ dodge, q_after[subset,2], q_after[subset,3]; direction=:x)
    scatter!(ax, μ_after[subset,2], ix .+ dodge; markersize=12)

    return f
end;
primal_plot(out_primal, out_primal2, 1:13)

#-
function dual_plot(before, after; kwargs...)
    df_before, df_after = describe(before, :mean, :std), describe(after, :mean, :std)
    ix = 1:nrow(df_before)
    f = Figure(size=(350,450))
    ax = Axis(f[1,1]; yticks=(ix, string.(df_before.variable)), yreversed=true, kwargs...)
    dodge = 0.2
    color = Makie.wong_colors()

    barplot!(ax, ix .- dodge, df_before.mean; direction=:x, width=0.5, strokewidth=1, color=(color[1], 0.33), strokecolor=color[1])
    errorbars!(ax, df_before.mean, ix .- dodge, df_before.std ./ √(nrow(df_before)); direction=:x, whiskerwidth=10, color=color[1])

    barplot!(ax, ix .+ dodge, df_after.mean; direction=:x, width=0.5, strokewidth=1, color=(color[2], 0.33), strokecolor=color[2])
    errorbars!(ax, df_after.mean, ix .+ dodge, df_after.std ./ √(nrow(df_after)); direction=:x, whiskerwidth=10, color=color[2])

    return f
end;
dual_plot(out_dual, out_dual2)

#=
We see that the prior sensitivity is now reduced, so that our goal of uninformative priors is closer to being achieved.
(Note that improper priors would have not been sensitive to power scaling.)
=#


##cell
#! format: off #src
using Literate #src

function preprocess(content) #src
    new_lines = map(split(content, "\n")) do line #src
        if endswith(line, "#src") #src
            line #src
        elseif startswith(line, "##cell") #src
            "#src" #src
        elseif startswith(line, "#text") #src
            replace(line, "#text" => "#") #src
        # try and save comments; strip necessary since Literate.jl also treats indented comments on their own line as markdown. #src
        elseif startswith(strip(line), "#") && !startswith(strip(line), "#=") && !startswith(strip(line), "#-") #src
            # TODO: should be replace first occurence only? #src
            replace(line, "#" => "##") #src
        # special for this to load text file #src
        elseif startswith(line, "basepath = ") #src
            "basepath = $(repr(dirname(@__DIR__)))" #src
        else #src
            line #src
        end #src
    end #src
    return join(new_lines, "\n") #src
end #src

withenv("JULIA_DEBUG" => "Literate") do #src
    @time Literate.markdown(@__FILE__, joinpath(pwd(), "..", "docs", "src", "tutorials"); execute = true, flavor = Literate.CommonMarkFlavor(), preprocess = preprocess) #src
end #src