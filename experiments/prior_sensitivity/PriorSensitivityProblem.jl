module PriorSensitivityProblem

# FIXME: this is to a large degree a copy of the GaussianMHProblem setup,
# so it feels like we can probably refactor to something general

__precompile__(false)

using Distributions
using StochasticAD
using LinearAlgebra
using DifferentiableMH
import Analysis: MarkovX


function X_kernel_init(θ, settings, options = (;))
    (; model_logpdf, n, f, burn_in, init, proposal, proposal_coupling) = settings
    return mh_kernel_init(x -> model_logpdf(x,θ), proposal, init; f, iters=n, burn_in, f_init=zero(f(init)), proposal_coupling)
end
    
function X_kernel(x_aug, kernel_params)
    return mh_kernel(x_aug, kernel_params)
end

function X_f(x_aug, settings, options = (;))
    (; n, burn_in) = settings
    return mh_f(x_aug; iters=n, burn_in)
end

"""
    make_prior_sensitivity_problem(model_logpdf, init, n=10000; f=first, burn_in=nothing)

Perform `n` steps of RWMH to sample from the posterior with power-scaling applied to the prior.
"""
function make_prior_sensitivity_problem(model_logpdf, init, n=10000;
        f = identity, burn_in = nothing,
        proposal = RandomWalkMHProposal{typeof(init)}(MvNormal(zero(init), 2.38^2/length(init) * I)))
    targets = Dict(
        "primal" => (; X = MarkovX(X_kernel, X_kernel_init, X_f), flags = [], name = "Primal")
    )
    burn_in = isnothing(burn_in) ? n ÷ 2 : burn_in
    settings = (; model_logpdf, n, p = 0.0, f, burn_in, init, proposal, proposal_coupling = MaximumReflectionProposalCoupling())
    return (; targets, settings)
end

export make_prior_sensitivity_problem

end