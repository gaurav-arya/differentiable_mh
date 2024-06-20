## Define RandomWalkMHProposal

"""
    RandomWalkMHProposal{T}(step_distribution) <: AbstractMHProposal

Make a random-walk Metropolis-Hastings proposal `P` for states of type `T`, supporting the
`AbstractMHProposal` interface with proposal given by perturbing the current state by a sample
from `step_distribution`. 

Here, `step_distribution` is expected to support `Base.rand` and `Distributions.logpdf`.
"""
struct RandomWalkMHProposal{T, PD} <: AbstractMHProposal{T}
    step_distribution::PD
end
Functors.functor(::Type{<:RandomWalkMHProposal{T}}, proposal) where {T} = (proposal.step_distribution,), fields -> RandomWalkMHProposal{T}(fields...)

function RandomWalkMHProposal{T}(step_distribution::PD) where {T, PD}
    return RandomWalkMHProposal{T, PD}(step_distribution)
end

function rand_proposal(rng::Random.AbstractRNG, P::RandomWalkMHProposal, x)
    return x + rand(rng, P.step_distribution) 
end

function logpdf_proposal(P::RandomWalkMHProposal, x, y)
    return logpdf(P.step_distribution, y - x)
end

function logratio_proposal(P::RandomWalkMHProposal, x, y)
    return 0.0
end

## Define MaximumReflectionProposalCoupling 

"""
    MaximumReflectionProposalCoupling 

Performs maximal reflection coupling with pathwise IPA.
We have simplified since the RWMH proposal is symmetric.
"""
struct MaximumReflectionProposalCoupling <: AbstractMHProposalCoupling end

stdnormlogpdf(x) = -(LinearAlgebra.norm_sqr(x) + length(x)*log(2π))/2

# Monkey patch because vectors of stochastic triples don't like the branching norm function
LinearAlgebra.norm_sqr(xs::Vector{StochasticTriple{T,V,FI}}) where {T,V,FI} =
   StochasticAD.propagate(LinearAlgebra.norm_sqr, xs; keep_deltas = Val(true))

# Maximal reflection coupling with pathwise IPA.
# We have simplified since the RWMH proposal is symmetric.
function coupled_proposal(rng::Random.AbstractRNG, proposal::AbstractMHProposal{T}, ::MaximumReflectionProposalCoupling, y, x, x_prop) where {T}
    if !(proposal isa RandomWalkMHProposal) && proposal.step_distribution isa Union{Normal, MvNormal}
        error("GaussianRandomWalkProposalCoupling only supports Random Walk MH proposals with Gaussian steps.")
    end
    if T <: Real
        @argcheck proposal.step_distribution isa Normal
        xlogp = logpdf_proposal(proposal, x, x_prop)
        # TODO: think more about RNG that we use here: maybe coupled_proposal should take a separate RNG argument as part of the interface?
        if log(rand(rng)) + xlogp ≤ logpdf_proposal(proposal, y, x_prop)
            y_prop = x_prop
        else
            y_prop = x + y - x_prop
        end
        return y_prop
    elseif T <: Vector{<:Real}
        @argcheck proposal.step_distribution isa MvNormal
        Σ = proposal.step_distribution.Σ
        Δ = y - x
        x_jump = whiten(Σ, x_prop - x)
        diff = whiten(Σ, Δ)
        if log(rand(rng)) + stdnormlogpdf(x_jump) ≤ stdnormlogpdf(x_jump - diff)
            return x_prop + zero(Δ)
        else
            diff_new = diff .* (1 - 2 * dot(diff, x_jump) / LinearAlgebra.norm_sqr(diff))
            #= # Version which avoids LinearAlgebra here to placate Enzyme
            a1 = sum(diff .* x_jump) #dot(diff, x_jump)
            a2 = sum(abs2, diff) #LinearAlgebra.norm_sqr(diff)
            diff_new = diff .* (1 - 2 * a1 / a2)
            =#
            return x_prop + unwhiten(Σ, diff_new)
        end
    else
        error("Unsupported state space type $T.")
    end
end

