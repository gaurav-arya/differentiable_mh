import Pkg; Pkg.activate(@__DIR__)
cd(dirname(@__DIR__)) 

using Distributions
using StochasticAD
using LinearAlgebra
using DifferentiableMH 
using CairoMakie
import Random

d(θ) = Normal(θ, 1)
σ = 2.38
proposal(x) = Normal(x, σ)
q(x, x_prop) = pdf(proposal(x), x_prop)
qrand(x) = rand(proposal(x))

# Define two coupling methods
function qrand_reflection(x_st::StochasticAD.StochasticTriple{T}) where {T}
    @assert iszero(StochasticAD.delta(x_st))
    x = StochasticAD.value(x_st)
    ω = rand()
    x_prop = quantile(Normal(x, σ), ω)
    Δs = map(x_st.Δs) do Δ
        y = x + Δ
        if rand() * q(x, x_prop) ≤ q(y, x_prop)
            y_prop = x_prop
        else
            y_prop = x_prop + (y - x)*(1 - 2*dot(y - x, x_prop - x)/sum(abs2, y - x))
        end
        return y_prop - x_prop
    end
    return StochasticAD.StochasticTriple{T}(x_prop, zero(x_prop), Δs)
end
function qrand_crn(x_st::StochasticAD.StochasticTriple{T}) where {T}
    @assert iszero(StochasticAD.delta(x_st))
    x = StochasticAD.value(x_st)
    ω = rand()
    x_prop = quantile(Normal(x, σ), ω)
    Δs = map(x_st.Δs) do Δ
        rand() # stupid hack to ensure consistent primal results with reflection; can do this with a better way.
        y = x + Δ
        y_prop = quantile(Normal(y, σ) , ω)
        return y_prop - x_prop
    end
    return StochasticAD.StochasticTriple{T}(x_prop, zero(x_prop), Δs)
end

# Set coupling
qrand(x::StochasticAD.StochasticTriple) = qrand_reflection(x)

begin
Random.seed!(1241)
Random.seed!(StochasticAD.RNG, 1234)
run_mh(θ) = mh(x -> logpdf(d(θ), x), qrand, 0.0; f=identity, iters=200)

samples = stochastic_triple(run_mh, 0.5).samples
vals = (st -> st.value).(samples)
alts = (st -> st.Δs.Δ).(samples)
weights = (st -> st.Δs.state.weight).(samples)
maximum(weights)
unique(weights)
end

begin
qrand2(x) = rand(proposal(x))
qrand2(x::StochasticAD.StochasticTriple) = qrand_crn(x)
run_mh2(θ) = mh(x -> logpdf(d(θ), x), qrand2, 0.0; f=identity, iters=200)
Random.seed!(T)
Random.seed!(StochasticAD.RNG, 1260)

samples2 = stochastic_triple(run_mh2, 0.5).samples
vals2 = (st -> st.value).(samples2)
alts2 = (st -> st.Δs.Δ).(samples2)
weights2 = (st -> st.Δs.state.weight).(samples2)
@assert vals2 == vals
maximum(weights2)
unique(weights2)
end

function make_figure(; bare=false)
    L=50
    H=100

    f = Figure(resolution=(600, 350))
    ax = Axis(f[1,1], xlabel="t", ylabel="Sample")
    xlims!(ax, L, H) 
    ylims!(ax, -1.1, 3.1) 
    scatterlines!(ax, L:H, vals[L:H], color=:grey, label="Primal")
    i = L
    j = L
    labeled = false
    while i <= H
        j = i
        while weights[i] == weights[j] && j <= H
            j += 1
        end
        if !labeled && (weights[i] == maximum(weights[L:H]))
            labeled = true
            label = (;label="Alternative")
        else
            label = (;)
        end
        scatterlines!(ax, (i-1):j, vcat(vals[i-1], alts[i:(j-1)], vals[j]); color=(:red, weights[i] / maximum(weights[L:H])), label...)
        i = j + 1
    end
    !bare && axislegend(ax; position=:rt, orientation=:horizontal)#, height=45)

    if bare
        hidedecorations!(ax)
        hidespines!(ax)
        resize!(f.scene, (600, 200))
    end

    f
end

f = make_figure()
save("plots/gaussian_plot_alts.png", f; px_per_unit=4)

fbare = make_figure(bare=true)
save("plots/gaussian_plot_alts_bare.png", fbare; px_per_unit=4)








