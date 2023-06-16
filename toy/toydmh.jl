# Toy implementation of differentiable 
# Gaussian random walk Metropolis-Hastings
# with maximal reflection coupling 

using LinearAlgebra
using Distributions
using ForwardDiff
using ProgressMeter
using Random
using Colors
using Statistics
using StatsBase
import Distributions.whiten
import Distributions.unwhiten

"""
    Chol{T}(L)

Wrapper for Cholesky decomposition of Σ = L*L'
"""
struct Chol{T}
    L::T
end
whiten(Σ::Chol, x) = Σ.L\x
unwhiten(Σ::Chol, z) = Σ.L*z


"""
    AdditiveGaussKernel(σ)

Create iso-Gaussian random walk proposal with variance σ².
"""
struct AdditiveGaussKernel{T}
    σ::T
end
(Q::AdditiveGaussKernel)(x) = MvNormal(x, Q.σ)
whiten(Q::AdditiveGaussKernel, x) = Q.σ\x
unwhiten(Q::AdditiveGaussKernel, z) = Q.σ*z

struct CrankNicolsonKernel{S,T}
    ρ::S
    σ::T
end
(Q::CrankNicolsonKernel)(x) = MvNormal(Q.ρ*x, sqrt(1-Q.ρ^2)*Q.σ)


function reflective_coupling(Σ, x, y)
    ϕ(z) = exp(-(z'*z)/2)
    Δ = whiten(Σ, y - x)
    e = normalize(Δ)
    z = randn!(similar(x))
    xᵒ = x + unwhiten(Σ, z)
    if rand()*ϕ(z) < ϕ(z - Δ) 
        yᵒ = xᵒ
    else 
        yᵒ = y + unwhiten(Σ, (z - (2*dot(e, z))*e))
    end
    xᵒ, yᵒ
end


"""
    coupled_rand(Q::AdditiveGaussKernel, x, y)  

Coupled proposal for two coupled chains in states `x` and `y`.
Uses maximal reflection coupling for Gaussian random walk updates.
"""
coupled_rand(Q::AdditiveGaussKernel, x, y) = reflective_coupling(Chol(Q.σ), x, y)
coupled_rand(Q::CrankNicolsonKernel, x, y) = reflective_coupling(Chol(sqrt(1 - Q.ρ^2)*Q.σ), Q.ρ*x, Q.ρ*y)

 
"""
    samples, S, ∂S, excursions =  dmh(x0, θ, K, logtarget, Q, f; S0=0.0, ∂S0=0.0)

Run Metropolis-Hastings with a given target `p(θ, x) = exp(logtarget(θ, x))` and
proposal kernel `Q`` and initial state `x0`.
Estimates the performance functional `S =∫ f(x) p(x, θ) dx` as sum over `K` samples
and its derivative `∂S`. Initial value for S and ∂S can be provided.
"""
function dmh(x0, θ, K, logtarget, Q, f; S0=0.0, ∂S0=0.0, dt = 0.5)
    xs = typeof(x0)[]
    accs = 0
    
    x = y = x0
    α(θ, x, xᵒ) = min(1.0, exp(logtarget(θ, xᵒ) - logtarget(θ, x) + logpdf(Q(xᵒ), x) - logpdf(Q(x), xᵒ)))
    ϖ = 0.0
    S = S0
    ∂S = ΔS = ∂S0

    excursions = 0 
    @showprogress dt for k in 1:K
        xᵒ, yᵒ = coupled_rand(Q, x, y)
        αx = α(θ, x, xᵒ)
        δαx = ForwardDiff.derivative(θ -> α(θ, x, xᵒ), θ)
        αy = α(θ, y, yᵒ)
        u = rand()
        accx = u < αx
        accy = u < αy
        xnew = accx ? xᵒ : x
        w = accx ? 1/αx * max(0, -δαx) : 1/(1 - αx) * max(0, δαx)
        ynew = accy ? yᵒ : y
        if ynew == xnew 
            recoupled = true
            ∂S += ϖ*ΔS
            ΔS = 0.0
            ϖ = 0.0
        else 
            recoupled = false
        end
        ϖ += w
        if rand()*ϖ < w
            ΔS = 0.0
            if accx 
                ynew = x
            else
                ynew = xᵒ
            end 
        end
        excursions += recoupled
        x, y = xnew, ynew
        ΔS += f(y) - f(x)
        S += f(x)
        accs += accx
        push!(xs, x)
    end
    xs, S/K, ∂S/K, excursions, accs/K
end

P(θ) = MvNormal([θ^2, θ], 1.2^2)
logtarget(θ, x) = logpdf(P(θ), x)
#σ = 1.0
#Q = AdditiveGaussKernel(σ)
σ = 3.0
Q = CrankNicolsonKernel(0.5, σ)

θ = 0.3
K = 1000000
x0 = [0.1, 0.1]

samples, S, ∂S, excursions, accp = @time dmh(x0, θ, K, logtarget, Q, x->x[1]+x[2])

θ² = θ^2
@show mean(samples) mean(P(θ))
@show accp
@show S, θ² + θ
@show ∂S, 2θ+1
avg_excursion_length = K/excursions
@show avg_excursion_length;
