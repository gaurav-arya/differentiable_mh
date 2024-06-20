module MHTuningProblem

using Statistics
using DifferentiableMH
using Distributions
using LinearAlgebra
using StochasticAD
using StochasticADExtra
using ArgCheck
using Functors

# TODO: port to kernel structure?

# Monkey patch necessary for mixture model; the original function branches
import LogExpFunctions
LogExpFunctions._logsumexp_onepass(X::Vector{StochasticTriple{T,V,FI}}) where {T,V,FI} =
   StochasticAD.propagate(LogExpFunctions._logsumexp_onepass, X; keep_deltas = Val(true))
function LogExpFunctions._logsumexp_onepass_op(x1::StochasticTriple{T}, x2::StochasticTriple{T}) where {T}
    xmax, a = if isnan(x1) || isnan(x2)
        z = oftype(x1, NaN)
        z, exp(z)
    else
        max(x1,x2), -abs(x1 - x2)
    end
    r = exp(a)
    return xmax, r
end
function LogExpFunctions._logsumexp_onepass_op(x::StochasticTriple{T}, xmax::StochasticTriple{T}, r::StochasticTriple{T}) where {T}
    _xmax, _r = if isnan(x) || isnan(xmax)
        # ensure that `NaN` is propagated correctly for complex numbers
        z = oftype(x, NaN)
        z, r + exp(z)
    else
        r1 = (r + one(r))  * exp(xmax - x)
        r2 = r + exp(x - xmax)
        i1 = max(x - xmax, 0)/(x - xmax)
        i2 = max(xmax - x, 0)/(xmax - x)

        max(x,xmax), r1*i1 + r2*i2
    end
    return _xmax, _r
end

# Monkey patch to passthrough scalars
LinearAlgebra.diagm(x::Real) = x

# Monkey patch to allow for fmap'ing Normal's
Functors.@functor Normal
Functors.@functor MvNormal

"""
    mh_acf(target, rproposal, θ, x0; iters = 1500)

Run a Metropolis-Hastings chain and estimate the 1-lag autocovariance.
Ideally a good proposal should have a low autocovariance.
"Folklore" suggests that the optimal acceptance rate for 1D problems is about
44%, contrary to most references you find online about asymptotic scaling limits
for Gaussians.

This function is differentiable by StochasticAD, via stochastic derivatives.
It showcases several extensions which "just work": support for
parameter-dependent proposals, and a multi-sample version of the estimator (i.e.
correctly combines across alternatives).
Note that you'd theoretically write this forwards, not backwards.
"""
function mh_acf(target, proposal::AbstractMHProposal{T}, x0; iters = 1500, proposal_coupling = nothing) where {T <: Real}
    # Canonicalize to T
    x = convert(T, x0)
    xprev = convert(T, x0) 

    acc = zero(logpdf(MHProposalDistribution(x, proposal, proposal_coupling), x))
    m = zero(x)
    cov = m^2 - mean(target)^2

    for i in 1:iters
        xprev, (x, coin) = x, step_mh(x, target, proposal; proposal_coupling)

        acc += StochasticAD.smooth_triple(coin)
        m += StochasticAD.smooth_triple(x)
        cov += StochasticAD.smooth_triple(xprev * x)
    end

    # acf apparently usually uses the length of the full chain as denominator
    (; sample_acc = acc/iters, sample_mean = m/iters, sample_autocorr = cov/(iters+1))
end

function mh_acf(target, proposal::AbstractMHProposal{T}, x0; iters = 1500, proposal_coupling = nothing) where {T <: Vector{<:Real}}
    # Canonicalize to T
    x = convert(T, x0)
    xprev = convert(T, x0) 

    acc = zero(logpdf(MHProposalDistribution(x, proposal, proposal_coupling), x))
    m = zero(x)
    cov = m * m' - diagm(mean(target).^2)

    for i in 1:iters
        xprev, (x, coin) = x, step_mh(x, target, proposal; proposal_coupling)

        acc += StochasticAD.smooth_triple.(coin)
        m += StochasticAD.smooth_triple.(x)
        cov += StochasticAD.smooth_triple.(xprev * x')
    end

    # acf apparently usually uses the length of the full chain as denominator
    # We make up the multidimensional extension as the determinant of the cross-covariance matrix.
    (; sample_acc = acc/iters, sample_mean = m/iters, sample_autocorr = det(cov./(iters+1)))
end

function step_mh(x, target, proposal; proposal_coupling = nothing)
    xᵒ = rand(MHProposalDistribution(x, proposal, proposal_coupling))
    logα = min(logpdf(target, xᵒ) - logpdf(target, x) + logratio_proposal(proposal, x, xᵒ), 0.0)
    coin = rand(Bernoulli(exp(logα)))
    x = x + (xᵒ - x) * coin #[x, xᵒ][1 + coin]
    return x, coin
end

function X(θ, settings, options=(;))
    (; n, target, x0, f) = settings
    if x0 isa Real
        proposal = RandomWalkMHProposal{typeof(x0 * θ)}(Normal(0, θ))
    else
        proposal = RandomWalkMHProposal{typeof(x0 * θ)}(MvNormal(zero(x0), θ^2 * I))
    end
    return f(mh_acf(target, proposal, x0; iters = n, proposal_coupling = MaximumReflectionProposalCoupling()))
end

function make_mh_tuning_problem(n=50; target, θ = nothing, x0 = 0.0, f = out -> out.sample_autocorr)
    if isnothing(θ)
        θ = 2.38 * std(target) / sqrt(length(x0))
    end
    targets = Dict(
        "primal" => (; X, flags = [], name = "Primal"),
    )
    settings = (; n, p = θ, target, x0, f)
    return (; targets, settings)
end

export make_mh_tuning_problem

end