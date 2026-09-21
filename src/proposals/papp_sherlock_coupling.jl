"""
    PappSherlockProposalCoupling(gradient, threshold)

Two-scale coupling for a multivariate Gaussian random-walk proposal. When the
chains are far apart, the proposal increments use the GCRefl construction;
when their standardized separation is below `threshold`, the coupling uses
maximal reflection so that the chains can meet.

`gradient` is evaluated at a state and should return the gradient of the
target log density. `threshold` is a squared distance in proposal-noise
coordinates, namely the coordinates in which the Gaussian proposal has an
identity covariance matrix.
"""
struct PappSherlockProposalCoupling{G} <: AbstractMHProposalCoupling
    gradient::G
    threshold::Float64
end

function _orthogonal_unit(v::AbstractVector{<:Real})
    # Select a coordinate direction that is not parallel to v, then remove
    # its component along v. This supplies a deterministic fallback when the
    # projected gradient is zero or numerically negligible.
    pivot = argmin(abs.(v))
    e = zeros(eltype(v), length(v))
    e[pivot] = one(eltype(v))
    u = e - dot(e, v) * v
    nu = norm(u)
    return iszero(nu) ? e : u / nu
end

function _projected_unit_gradient(
    g::AbstractVector{<:Real},
    separation_direction::AbstractVector{<:Real},
)
    # GCRefl uses gradient directions orthogonal to the separation direction.
    # The projection leaves the separation direction for the reflection part
    # of the coupling.
    ng = norm(g)
    if !isfinite(ng) || iszero(ng)
        return _orthogonal_unit(separation_direction)
    end
    projected = g / ng
    projected -= dot(projected, separation_direction) * separation_direction
    np = norm(projected)
    tolerance = sqrt(eps(real(float(one(eltype(projected))))))
    return (!isfinite(np) || np ≤ tolerance) ?
        _orthogonal_unit(separation_direction) : projected / np
end

function _noise_gradient(Σ, gradient::AbstractVector{<:Real})
    # Write a proposal jump as L*z with Σ = L*L'. A change in z changes the
    # state by L*dz, so the gradient in z-coordinates is L'*gradient.
    return cholesky(Σ).U * gradient
end

function _gcrefl_conditional_jump(
    rng::Random.AbstractRNG,
    x_jump::AbstractVector{<:Real},
    separation_direction::AbstractVector{<:Real},
    ex::AbstractVector{<:Real},
    ey::AbstractVector{<:Real},
)
    # Let e denote the separation direction. The GCRefl proposal increments
    # are
    #
    #   Zx = Z - <ex,Z> ex + η ex,
    #   Zy = Z - 2<e,Z>e - <ey,Z> ey + η ey.
    #
    # The API supplies Zx. Its ex component is η, and its orthogonal component
    # is the corresponding part of Z. The remaining component of Z is
    # independent standard Gaussian noise and can be sampled conditionally.
    η = dot(ex, x_jump)
    z_orthogonal = x_jump - η .* ex
    z = z_orthogonal + randn(rng) .* ex
    y_jump = z - dot(ey, z) .* ey + η .* ey
    y_jump -= 2 .* dot(separation_direction, z) .* separation_direction
    return y_jump
end

function coupled_proposal(
    rng::Random.AbstractRNG,
    proposal::AbstractMHProposal{T},
    coupling::PappSherlockProposalCoupling{G},
    y,
    x,
    x_prop,
) where {T,G}
    if !(proposal isa RandomWalkMHProposal) ||
       !(proposal.step_distribution isa MvNormal)
        error(
            "PappSherlockProposalCoupling requires a multivariate Gaussian " *
            "RandomWalkMHProposal."
        )
    end
    @argcheck coupling.threshold > 0 "GCRefl threshold must be positive"
    if !(T <: Vector{<:Real})
        error("Unsupported state space type $T.")
    end

    Σ = proposal.step_distribution.Σ
    Δ = y - x
    x_jump = whiten(Σ, x_prop - x)
    separation = whiten(Σ, Δ)
    separation_norm² = LinearAlgebra.norm_sqr(separation)

    if separation_norm² < coupling.threshold
        # Near the diagonal, use the maximal reflection coupling. Its common
        # part can produce an exact meeting after the shared MH accept/reject
        # uniform is applied.
        if log(rand(rng)) + stdnormlogpdf(x_jump) ≤
           stdnormlogpdf(x_jump - separation)
            return x_prop + zero(Δ)
        end
        reflected_jump = x_jump -
            2 * dot(separation, x_jump) / separation_norm² .* separation
        return y + unwhiten(Σ, reflected_jump)
    end

    # Far from the diagonal, use GCRefl. The two gradient directions are
    # formed in proposal-noise coordinates and projected away from the
    # separation direction before constructing the conditional y jump.
    separation ./= sqrt(separation_norm²)
    ex = _projected_unit_gradient(
        _noise_gradient(Σ, coupling.gradient(x)), separation
    )
    ey = _projected_unit_gradient(
        _noise_gradient(Σ, coupling.gradient(y)), separation
    )
    y_jump = _gcrefl_conditional_jump(
        rng, x_jump, separation, ex, ey
    )
    return y + unwhiten(Σ, y_jump)
end
