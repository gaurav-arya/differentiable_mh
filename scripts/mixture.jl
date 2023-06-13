import Pkg; Pkg.activate(@__DIR__)
cd(dirname(@__DIR__)) 

using Distributions
using CairoMakie
using StochasticAD
using ProgressMeter
using DifferentiableMH 
using StaticArrays
using Optimisers
using ForwardDiff
using BenchmarkTools

function choose_gaussian(i=nothing)
    return Normal(2 + [-9, 0, 6][i] * 0.5, 4)
end

function generative_model(i=nothing)
    (i === nothing) && (i = rand(1:3))
    return rand(choose_gaussian(i))
end
generative_model(i::StochasticAD.StochasticTriple) = StochasticAD.propagate(i -> generative_model(i), i)

function logprob(i=nothing; obs)
    if i === nothing
        return log(0.5 * exp(logprob(1; obs)) + 0.5 * exp(logprob(2; obs)))
    end
    return logpdf(choose_gaussian(i), obs)
end
logprob(i::StochasticAD.StochasticTriple; kwargs...) = StochasticAD.propagate(i -> logprob(i; kwargs...), i)

function enumerated_posterior(; obs)
    distr = exp.([logprob(i; obs) for i in 1:3])
    return distr / sum(distr) 
end
proposal(i) = rand(vcat(1:i-1, i+1:3)) # always propose one of the other options for i
proposal(i_st::StochasticAD.StochasticTriple{T}) where {T} = StochasticAD.propagate(proposal, i_st)
proposal(i_st::StochasticAD.StochasticTriple{T}) where {T} = proposal_residual(i_st)
# maximum residual coupling
function proposal_residual(i_st::StochasticAD.StochasticTriple{T}) where {T}
    @assert iszero(StochasticAD.delta(i_st))
    i = StochasticAD.value(i_st)
    i_prop = proposal(i) 
    Δs = map(i_st.Δs) do Δ
        j = i + Δ
        if i == j
           j_prop = i_prop 
        elseif i_prop !== j
           j_prop = i_prop 
        else
            j_prop = i 
        end
        return j_prop - i_prop
    end
    return StochasticAD.StochasticTriple{T}(i_prop, zero(i_prop), Δs)
end

# Objective for optimizing inference
ami(x, J) = (x == J)
ami(x::StochasticAD.StochasticTriple, J) = StochasticAD.propagate(ami, x, J)
function obj(obs; N=5000)
    samples, S = mh(i -> logprob(i; obs), proposal, 1; f=x->SA[ami(x, 1), ami(x, 2), ami(x, 3)], iters=50+N, burn_in=50, init=SA[0, 0, 0])
    return entropy(S/length(samples))
end

function make_figure()
    fig = Figure(resolution=(700, 600), fontsize=17)
    obs_arr = -16:0.01:16
    alphaval = 0.2

    # plot logprobs of generative model
    ax1 = Axis(fig[1,1], ylabel="Probability", xlabel="Observation")
    _prob(obs, i) = exp(logprob(i; obs))
    band!(ax1, obs_arr, zero.(obs_arr), _prob.(obs_arr, 1), label="i=1", color=(:blue, alphaval))
    band!(ax1, obs_arr, zero.(obs_arr), _prob.(obs_arr, 2), label="i=2", color=(:orange, alphaval))
    band!(ax1, obs_arr, zero.(obs_arr), _prob.(obs_arr, 3), label="i=3", color=(:green, alphaval))
    axislegend(ax1; position=:lt)

    ## plot posteriors on i for varying observations
    ax2 = Axis(fig[2,1:2], ylabel="Probability", xlabel="Observation")
    ylims!(ax2, -0.1, 1.2)
    xlims!(ax2, -14, 14) 
    _posterior(obs) = enumerated_posterior(; obs)
    posterior1_arr = (first∘_posterior).(obs_arr)
    posterior2_arr = ((i->i[2])∘_posterior).(obs_arr)
    band!(ax2, obs_arr, zero.(obs_arr), posterior1_arr, label="i=1", color=(:blue, alphaval))
    band!(ax2, obs_arr, posterior1_arr, posterior1_arr .+ posterior2_arr, label="i=2", color=(:orange, alphaval))
    band!(ax2, obs_arr, posterior1_arr .+ posterior2_arr, one.(obs_arr), label="i=3", color=(:green, alphaval))
    axislegend(ax2; position=:lt)

    ## plot inference trajectory
    obs = 4.0
    samples, _... = mh(i -> logprob(i; obs), proposal, 1; f=(x)->[x^2,x], init=[0,0])
    ax3 = Axis(fig[1, 2], ylabel="Sample", xlabel="Sample number")
    scatter!(ax3, 1:length(samples), samples, color=:black, marker='x')

    ## plot gradients 
    ax4 = Axis(fig[2,1:2], ylabel="Value", yaxisposition = :right) 
    ylims!(ax4, -0.1, 1.2)
    xlims!(ax4, -14, 14) 
    hidespines!(ax4)
    hidexdecorations!(ax4)

    obs_arr_small = -14:0.5:14
    triples = @showprogress [stochastic_triple(obj, obs) for obs in obs_arr_small]
    objs = StochasticAD.value.(triples)
    grads = derivative_contribution.(triples)
    scatterlines!(ax4, obs_arr_small, objs, color=:ivory4, markersize=7, label="Objective")
    scatterlines!(ax4, obs_arr_small, grads, color=:black, markersize=7, label="Gradient")
    axislegend(ax4; position=:rt)

    save("plots/mcmc_estimator.png", fig; px_per_unit=4)

    fig
end

# Gradient descent
function do_descent()
    obs = [8.0]
    opt = Adam(1e-1)
    setup = Optimisers.setup(opt, obs)
    for i in 1:100
        i % 10 == 0 && println("Iteration $i")
        g = derivative_estimate(o -> -obj(o[1]), obs)
        @show g
        setup, obs = Optimisers.update(setup, obs, g)
        @show obs
    end
end

function obj_scoregrad(obs; N=5000)
    samples, S = mh_score(i -> logprob(i; obs), proposal, 1; f=x->SA[ami(x, 1), ami(x, 2), ami(x, 3)], iters=50+N, burn_in=50, init=SA[0, 0, 0])
    return entropy(S/length(samples))
end

function graph_variance()
    Ns = [50, 200, 500, 2000, 5000]
    means_st = Float64[]
    means_score = Float64[]
    stds_st = Float64[]
    stds_score = Float64[]
    obs = 0.4
    for N in Ns
        @show N
        samples_st = [stochastic_triple(obs -> obj(obs; N), obs) for i in 1:(100000 ÷ N)]
        samples_score = [stochastic_triple(obs -> obj_scoregrad(obs; N), obs) for i in 1:(100000 ÷ N)]
        derivs_st = StochasticAD.derivative_contribution.(samples_st)
        derivs_score = StochasticAD.delta.(samples_score)
        @show mean(derivs_st) mean(derivs_score)
        @show std(derivs_st) std(derivs_score)
        push!(means_st, mean(derivs_st))
        push!(means_score, mean(derivs_score))
        push!(stds_st, std(derivs_st))
        push!(stds_score, std(derivs_score))
    end

    fig = Figure(resolution=(600, 400))
    ax = Axis(fig[1, 1], yscale=log10, xscale=log10, xlabel="Metropolis-Hastings Chain Length T", ylabel="Variance")
    scatterlines!(ax, Ns, stds_score.^2, label = "Score method", linewidth=2)
    scatterlines!(ax, Ns, stds_st.^2, label="Stochastic derivatives", linewidth=2)
    lines!(ax, Ns, (1)./Ns, label = "1/T", linewidth=2, linestyle=:dot, color=:black)
    axislegend(ax; position=(1.0, 0.65))
    save("plots/score_comparison.png", fig; px_per_unit=4)
    fig
end

make_figure()
do_descent()
graph_variance()