"""
    MALAProposal(gaussian_increment, γ, ∇logπ) <: AbstractMHProposal

Construct a Metropolis-adjusted Langevin proposal by wrapping a Gaussian
random-walk proposal whose covariance is `γ * cov(gaussian_increment)`:

The proposal is `N(x + γ*C*∇logπ(x)/2, γ*C)`.

`gaussian_increment` supplies the base covariance `C` and is normally a
zero-mean `Normal` or `MvNormal`.  The gradient may be any callable accepted
by `StochasticAD.propagate`.  The embedded
`RandomWalkMHProposal` is also the object passed to the RWM Gaussian coupling
after the two MALA proposal means have been shifted.
"""
struct MALAProposal{T,P<:RandomWalkMHProposal,G,F} <: AbstractMHProposal{T}
    inner_proposal::P
    γ::G
    ∇logπ::F
end

Functors.functor(::Type{<:MALAProposal{T}}, proposal) where {T} =
    (proposal.inner_proposal, proposal.γ, proposal.∇logπ),
    fields -> MALAProposal{T}(fields...)

function MALAProposal{T}(
    inner_proposal::RandomWalkMHProposal{T},
    γ::G,
    ∇logπ::F,
) where {T,G,F}
    @argcheck γ > 0
    @argcheck inner_proposal.step_distribution isa Union{Normal,MvNormal}
    return MALAProposal{T,typeof(inner_proposal),G,F}(
        inner_proposal, γ, ∇logπ
    )
end

function MALAProposal{T}(
    gaussian_increment::D,
    γ::G,
    ∇logπ::F,
) where {T,D<:Normal,G,F}
    @argcheck γ > 0
    @argcheck iszero(gaussian_increment.μ) "MALA gaussian_increment must be centered"
    inner = RandomWalkMHProposal{T}(
        Normal(zero(gaussian_increment.μ), sqrt(γ) * gaussian_increment.σ)
    )
    return MALAProposal{T}(inner, γ, ∇logπ)
end

function MALAProposal{T}(
    gaussian_increment::D,
    γ::G,
    ∇logπ::F,
) where {T,D<:MvNormal,G,F}
    @argcheck γ > 0
    @argcheck iszero(gaussian_increment.μ) "MALA gaussian_increment must be centered"
    inner = RandomWalkMHProposal{T}(
        # Scalar multiplication preserves structured PDMats (notably ScalMat
        # and PDiagMat), whereas broadcasting materializes them as dense
        # matrices.  Keeping the structure is especially important when the
        # state contains StochasticTriples: a dense covariance turns an
        # isotropic MALA drift into an O(d²) generic matrix-vector product.
        MvNormal(zero(gaussian_increment.μ), γ * gaussian_increment.Σ)
    )
    return MALAProposal{T}(inner, γ, ∇logπ)
end

function _gradient(P::MALAProposal, x)
    # Treat the score object as an input as well as the state.  In particular,
    # callable structs may contain differentiated target parameters.  Passing
    # only `x` to `propagate` silently drops those explicit derivatives once
    # `x` itself is stochastic.  MALA also has a continuous state derivative,
    # so retain deltas while propagating finite perturbations.
    apply_score(score, state) = score(state)
    inputs = (P.∇logπ, x)
    has_perturbations = any(StochasticAD.structural_iterate(inputs)) do leaf
        leaf isa StochasticAD.StochasticTriple && !isempty(leaf.Δs)
    end
    # Besides avoiding unnecessary alternative evaluation, the direct path is
    # important for backends whose empty coupled representation has no inferred
    # value type.  Ordinary operator overloading retains all continuous deltas.
    has_perturbations || return apply_score(inputs...)
    return StochasticAD.propagate(
        apply_score, inputs...; keep_deltas=Val(true))
end

function _mean(P::MALAProposal{T,PW}, x) where {T,PW<:RandomWalkMHProposal}
    step_distribution = P.inner_proposal.step_distribution
    if step_distribution isa Normal
        covariance = step_distribution.σ^2
        return x + covariance / 2 * _gradient(P, x)
    elseif step_distribution isa MvNormal
        return x + step_distribution.Σ / 2 * _gradient(P, x)
    end
    error("MALAProposal requires a Normal or MvNormal embedded random walk")
end

function rand_proposal(rng::Random.AbstractRNG, P::MALAProposal, x)
    return rand_proposal(rng, P.inner_proposal, _mean(P, x))
end

function logpdf_proposal(P::MALAProposal, x, y)
    return logpdf_proposal(P.inner_proposal, _mean(P, x), y)
end

function logratio_proposal(P::MALAProposal, x, y)
    gx = _gradient(P, x)
    gy = _gradient(P, y)
    step_distribution = P.inner_proposal.step_distribution
    if step_distribution isa Normal
        covariance = step_distribution.σ^2
        return 0.5 * (x - y) * (gy + gx) - covariance / 8 * (gy^2 - gx^2)
    elseif step_distribution isa MvNormal
        covariance = step_distribution.Σ
        return 0.5 * dot(x - y, gy + gx) -
            (dot(gy, covariance * gy) - dot(gx, covariance * gx)) / 8
    end
    error("MALAProposal requires a Normal or MvNormal embedded random walk")
end

function coupled_proposal(
    rng::Random.AbstractRNG,
    P::MALAProposal{T,PW,G,F},
    ::MaximumReflectionProposalCoupling,
    y,
    x,
    x_prop,
) where {T,PW,G,F}
    return coupled_proposal(
        rng, P.inner_proposal, MaximumReflectionProposalCoupling(),
        _mean(P, y), _mean(P, x), x_prop,
    )
end

function coupled_proposal(
    rng::Random.AbstractRNG,
    P::MALAProposal{T,PW,G,F},
    coupling::PappSherlockProposalCoupling,
    y,
    x,
    x_prop,
) where {T,PW,G,F}
    if P.inner_proposal.step_distribution isa Normal
        # In one dimension there is no orthogonal gradient subspace, so use
        # maximal reflection throughout.
        return coupled_proposal(
            rng, P, MaximumReflectionProposalCoupling(), y, x, x_prop
        )
    end
    # Apply the Gaussian coupling to the shifted MALA proposal laws. The
    # Papp/Sherlock results are for RWM and do not cover this extension.
    return coupled_proposal(
        rng, P.inner_proposal, coupling,
        _mean(P, y), _mean(P, x), x_prop,
    )
end
