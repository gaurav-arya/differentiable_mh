## Define AbstractMHProposal interface

"""
    AbstractMHProposal{T}

An abstract type for Metropolis-Hastings proposals on instances of T.
A new proposal needs to implement [`rand_proposal`](@ref) and [`logpdf_proposal`](@ref).

Users of a `proposal` (e.g. implementers of Metropolis-Hastings algorithms) should wrap
their proposals in `MHProposalDistribution`, to e.g. enable providing custom `proposal_coupling`'s
to customize automatic differentiatinon.
"""
abstract type AbstractMHProposal{T} end

"""
    rand_proposal(rng::AbstractRNG, ::AbstractMHProposal{T}, x::T)::T

Propose the next step `x_prop::T` given current Metropolis-Hastings state `x::T`.
"""
function rand_proposal end

"""
    logpdf_proposal(proposal::AbstractMHProposal{T}, x::T, x_prop::T)

Return log-density of `x_prop` under `rand_proposal(proposal, x)`.
"""
function logpdf_proposal end

"""
    logratio_proposal(proposal::AbstractMHProposal{T}, x::T, x_prop::T)

Return log of density part in acceptance ratio from `x` to `x_prop` under `proposal`.
Defaults to `logpdf_proposal(proposal, x_prop, x) - logpdf_proposal(proposal, x, x_prop)`
but could be replaced by more efficient implementations.
"""
function logratio_proposal(proposal::AbstractMHProposal, x, x_prop)
    return logpdf_proposal(proposal, x_prop, x) - logpdf_proposal(proposal, x, x_prop)
end

## Define AbstractMHProposalCoupling 

"""
    AbstractMHProposalCoupling 

An abstract type for proposal coupling strategies, supporting [`coupled_proposal`](@ref).
"""
abstract type AbstractMHProposalCoupling end

"""
    coupled_proposal(rng::AbstractRNG, P::AbstractMHProposal, CP::AbstractMHProposalCoupling, y, x, x_prop)

Sample a `y_prop`, where for all `x` and `y` it must hold that `y_prop` is marginally distributed 
from `rand_proposal(P, y)` when `x_prop` is drawn from `rand_proposal(P, x)`.
"""
function coupled_proposal end

## Implement basic independent coupling

struct IndependentMHProposalCoupling end

function coupled_proposal(rng::Random.AbstractRNG, P::AbstractMHProposal, ::IndependentMHProposalCoupling, y, x, x_prop)
    return rand_proposal(rng, P, y)
end

## Define MHProposalDistribution wrapper

"""
    MHProposalDistribution(x::T, proposal::AbstractMHProposal, proposal_coupling::AbstractMHProposalCoupling)

Given a Metropolis-Hastings state `x` and a Metropolis-Hastings proposal `proposal`,
form the sample proposal at that point, implementing `Base.rand` and `Distributions.logpdf`.

If `proposal_coupling` is not `nothing`, then `MHProposalDistribution` will automatically be a valid `StochasticAD`-differentiable 
primitive that uses `proposal_coupling` to propagate alternatives.
"""
struct MHProposalDistribution{T, P <: AbstractMHProposal, PC <: Union{AbstractMHProposalCoupling, Nothing}}
    x::T
    proposal::P
    proposal_coupling::PC
end

Functors.@functor MHProposalDistribution

Base.rand(rng::Random.AbstractRNG, d::MHProposalDistribution) = rand_proposal(rng, d.proposal, d.x)
Base.rand(d::MHProposalDistribution) = rand(Random.default_rng(), d)
function Distributions.logpdf(d::MHProposalDistribution, x_prop)
    return logpdf_proposal(d.proposal, d.x, x_prop)
end

## Define randst and StochasticTriple rand overload on AbstractMHProposal's 

# TODO: can StochasticAD.propagate take care of some of boilerplate for defining a primitive here?
# (A: yes, it should, if we generalize it sufficiently.)
function Base.rand(rng::Random.AbstractRNG, d_st::MHProposalDistribution{<:Union{StochasticTriple{T}, Vector{<:StochasticTriple{T}}}}) where {T}
    x_st = d_st.x
    proposal_coupling = d_st.proposal_coupling
    if isnothing(proposal_coupling)
        error("No default proposal coupling strategy has been defined for sampling from an MHProposalDistribution. Please provide one.")
    end
    if x_st isa Real
        # Get IPA terms
        x_withdelta = StochasticAD.strip_Δs(x_st; use_dual = Val(false)) 
        proposal_withdelta = d_st.proposal # assume already just IPA
        x_prop_withdelta = rand_proposal(rng, proposal_withdelta, x_withdelta)  # could use rand itself here, but then we get StackOverflow issues; so just use the lower primitive.

        # Get raw primal terms
        x = StochasticAD.value(x_st)
        proposal = StochasticAD.structural_map(StochasticAD.get_value, d_st.proposal)
        x_prop = StochasticAD.value(x_prop_withdelta)

        # Propagate the discrete perturbations in x_st
        x_prop_Δs = let x = x, proposal = proposal, x_prop = x_prop
            map(x_st.Δs) do Δ
                y = x + Δ
                y_prop = coupled_proposal(rng, proposal, proposal_coupling, y, x, x_prop)
                return y_prop - x_prop
            end
        end
        # Form triple
        return StochasticAD.StochasticTriple{T}(x_prop, StochasticAD.delta(x_prop_withdelta), x_prop_Δs)
    elseif x_st isa Vector{<:Real} # Implementation based on StochasticAD.propagate
        # Get IPA terms
        x_withdelta = map(st -> StochasticAD.strip_Δs(st; use_dual = Val(false)), x_st)
        proposal_withdelta = d_st.proposal # assume already just IPA
        x_prop_withdelta = rand_proposal(rng, proposal_withdelta, x_withdelta)

        # Get raw primal terms
        x = StochasticAD.value.(x_st)
        proposal = StochasticAD.structural_map(StochasticAD.get_value, d_st.proposal)
        x_prop = StochasticAD.value.(x_prop_withdelta)

        # Propagate the perturbations in x_st
        FIs = StochasticAD.backendtype(eltype(x_st))
        x_Δs = map(Base.Fix2(StochasticAD.get_Δs, FIs), x_st)
        x_Δs_coupled = StochasticAD.couple(FIs, x_Δs; rep = first(x_Δs), out_rep = StochasticAD.value.(x))
        x_prop_Δs_coupled = let x = x, proposal = proposal, x_prop = x_prop
            map(x_Δs_coupled) do Δ
                y = x + Δ
                y_prop = coupled_proposal(rng, proposal, proposal_coupling, y, x, x_prop)
                return y_prop - x_prop
            end
        end
        x_prop_Δs = StochasticAD.scalarize(x_prop_Δs_coupled; out_rep = x)
        # Form triple
        return map(x_prop_withdelta, x_prop_Δs) do x_prop_withdelta_i, x_prop_Δs_i
            StochasticAD.StochasticTriple{T}(StochasticAD.value(x_prop_withdelta_i), StochasticAD.delta(x_prop_withdelta_i), x_prop_Δs_i)
        end
    else
        error("Unsupported state space type $T.")
    end
end
