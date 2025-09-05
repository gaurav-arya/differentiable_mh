## Define IndependentMHProposal

"""
    IndependentMHProposal{T}(distribution) <: AbstractMHProposal

Make an independent Metropolis-Hastings proposal `P` for states of type `T`, supporting the
`AbstractMHProposal` interface with proposal given by sampling from `distribution`. 

Here, `distribution` is expected to support `Base.rand` and `Distributions.logpdf`.
"""
struct IndependentMHProposal{T, PD} <: AbstractMHProposal{T}
    distribution::PD
end
Functors.functor(::Type{<:IndependentMHProposal{T}}, proposal) where {T} = (proposal.distribution,), fields -> IndependentMHProposal{T}(fields...)

function IndependentMHProposal{T}(distribution::PD) where {T, PD}
    return IndependentMHProposal{T, PD}(distribution)
end

function rand_proposal(rng::Random.AbstractRNG, P::IndependentMHProposal, x)
    return rand(rng, P.distribution) 
end

function logpdf_proposal(P::IndependentMHProposal, x, y)
    return logpdf(P.distribution, y)
end

## Implement basic independent coupling

"""
    IndependentMHProposalCoupling 

Performs a simple independent coupling using the same proposal for both chains.
"""
struct IndependentMHProposalCoupling <: AbstractMHProposalCoupling  end

function coupled_proposal(rng::Random.AbstractRNG, P::AbstractMHProposal, ::IndependentMHProposalCoupling, y, x, x_prop)
    return x_prop
end