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

"""
    AdditiveGaussKernel(σ)

Create iso-Gaussian random walk proposal with variance σ².
"""
struct AdditiveGaussKernel{T}
    σ::T
end
(Q::AdditiveGaussKernel)(x) = MvNormal(x, σ^2)

"""
    coupled_rand(Q::AdditiveGaussKernel, x, y)  

Coupled proposal for two coupled chains in states `x` and `y`.
Uses maximal reflection coupling for Gaussian random walk updates.
"""
function coupled_rand(Q::AdditiveGaussKernel, x, y)   
    x_prop = rand(Q(x))
    if rand() * pdf(Q(x), x_prop) ≤ pdf(Q(y), x_prop)
        y_prop = x_prop
    else
        y_prop = x_prop + (y - x)*(1 - 2*dot(y - x, x_prop - x)/sum(abs2, y - x))
    end
    return x_prop, y_prop
end

"""
    samples, S, ∂S, excursions =  dmh(x0, θ, K, logtarget, Q, f; S0=0.0, ∂S0=0.0)

Run Metropolis-Hastings with a given target `p(θ, x) = exp(logtarget(θ, x))` and
proposal kernel `Q`` and initial state `x0`.
Estimates the performance functional `S =∫ f(x) p(x, θ) dx` as sum over `K` samples
and its derivative `∂S`. Initial value for S and ∂S can be provided.
"""
function dmh(x0, θ, K, logtarget, Q, f; S0=0.0, ∂S0=0.0)
    xs = typeof(x0)[]
    accs = 0
    
    x = y = x0
    α(θ, x, xᵒ) = min(1.0, exp(logtarget(θ, xᵒ) - logtarget(θ, x) + logpdf(Q(xᵒ), x) - logpdf(Q(x), xᵒ)))
    ϖ = 0.0
    S = S0
    ∂S = ∂S0

    excursions = 0 
    @showprogress for k in 1:K
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
            ϖ = 0.0
        else 
            recoupled = false
        end
        ϖ += w
        if rand()*ϖ < w
            if accx 
                ynew = x
            else
                ynew = xᵒ
            end 
        end
        excursions += recoupled
        x, y = xnew, ynew
        ∂S += ϖ*(f(y) - f(x))
        S += f(x)
        accs += accx
        push!(xs, x)
    end
    xs, S/K, ∂S/K, excursions, accs/K
end

P(θ) = MvNormal([θ^2, θ], 1.2^2)
logtarget(θ, x) = logpdf(P(θ), x)
σ = 1.0
Q = AdditiveGaussKernel(σ)
θ = 1.3
K = 60000
x0 = [0.1, 0.1]
samples, S, ∂S, excursions, accp = @time dmh(x0, θ, K, logtarget, Q, x->x[1]+x[2])
θ² = θ^2
@show mean(samples)
@show accp
@show S, θ² + θ
@show ∂S, 2θ+1
avg_excursion_length = K/excursions
@show avg_excursion_length;
