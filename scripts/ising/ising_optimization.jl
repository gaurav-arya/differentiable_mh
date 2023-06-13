include("IsingCore.jl")
using .IsingCore
using StochasticAD
using CairoMakie
using Optimisers
using LinearAlgebra
using Statistics

N = 12 # size of the lattice, N x N
T, E, M, C, X = @time ising_model_manyT(N, mc_move_c)

function optimize(θ0, move; η=0.01, n=200, n_batch=1)
    optimizer = Adam(η)
    θ = deepcopy(θ0)
    θ_trace = [θ]
    setup = Optimisers.setup(optimizer, θ)
    for i in 1:n
        i % 10 == 0 && println("Iteration $i")
        g = grad_estimator(θ, n_batch, move)
        setup, θ = Optimisers.update(setup, θ, g)
        push!(θ_trace, deepcopy(θ))
    end
    return θ_trace
end

function obj(T, move)
    config = initial_state(N)
    # outputs: E, M, C, X, configs
    -ising_model(N, T, config, move)[3]
end
grad_estimator(θ, n_batch, move) = [mean(derivative_estimate(T->obj(T, move), θ[1]) for i in 1:n_batch)]

T0 = [6.0]
θ_trace_c = optimize(T0, mc_move_c; η=0.003, n=1000, n_batch=1)
obj_trace_c = [-obj(θ[1], mc_move_c) for θ in θ_trace_c]

θ_trace = optimize(T0, mc_move; η=0.003, n=1000, n_batch=1)
obj_trace = [-obj(θ[1], mc_move) for θ in θ_trace]

begin
    fig1 = Figure(resolution=(1120, 525), fontsize = 30)
    fig2 = Figure(resolution=(1120, 525), fontsize = 30)
    ax1 = fig1[1, 1] = Axis(fig1, xlabel="Temperature", ylabel="Energy")
    ax2 = fig1[1, 2] = Axis(fig1, xlabel="Temperature", ylabel="Heat capacity")
    Label(fig1[1, 1, TopLeft()], "a)");
    Label(fig1[1, 2, TopLeft()], "b)");
    ax3 = fig2[1, 1] = Axis(fig2, ylabel="Heat capacity", xlabel="Iterations")
    ax4 = fig2[1, 2] = Axis(fig2, ylabel="Temperature", xlabel="Iterations")
    Label(fig2[1, 1, TopLeft()], "a)");
    Label(fig2[1, 2, TopLeft()], "b)");

    T_crit = 2 / log(1 + sqrt(2))
    T_hat = θ_trace[end]
    T_hat_c = θ_trace_c[end]

    scatter!(ax1, T, E, color=:black, markersize=10)
    vlines!(ax1, T_crit, color=Makie.Colors.RGB(0/255,90/255,181/255), linewidth=2)
    vlines!(ax1, T_hat_c, color=Makie.Colors.RGB(220/255,50/255,32/255), linewidth=2)
    #vlines!(ax1, T_hat, color=, linewidth=3)

    scatter!(ax2, T, C, color=:black, markersize=10)
    vlines!(ax2, T_crit, color=Makie.Colors.RGB(0/255,90/255,181/255), linewidth=2)
    vlines!(ax2, T_hat_c, color=Makie.Colors.RGB(220/255,50/255,32/255), linewidth=2)
    #vlines!(ax2, T_hat, color=, linewidth=3)

    lines!(ax3, obj_trace, color=:gray, label="uncoupled")
    lines!(ax3, obj_trace_c, color=Makie.Colors.RGB(220/255,50/255,32/255), label="coupled")

    lines!(ax4, [θ[1] for θ in θ_trace], color=:gray, label="uncoupled")
    lines!(ax4, [θ[1] for θ in θ_trace_c], color=Makie.Colors.RGB(220/255,50/255,32/255), label="coupled")
    fig1
end

save("plots/Ising_c_optimization_new.png", fig1; px_per_unit=4)
save("plots/Ising_c_optimization_app.png", fig2; px_per_unit=4)
