includet("IsingCore.jl")
using .IsingCore
using StochasticAD
using CairoMakie
using ProgressMeter
using Statistics
import Random

# Plot alternative
begin
Random.seed!(1234)
Random.seed!(StochasticAD.RNG, 1234)
N = 12
initial_config = initial_state(N)
move = (config, beta, N) -> mc_move_cc(config, beta, N, acceptreject)
triple_configs = stochastic_triple(T->ising_model(N, T, initial_config, mc_move_c)[end], 3.0);
samples = @showprogress [stochastic_triple(T->ising_model(N, T, initial_config, mc_move_c)[3], 3.0) for i in 1:10]
# samples_other = @showprogress [stochastic_triple(T->ising_model(N, T, initial_config, move)[3], 3.0) for i in 1:10]
end
mean(derivative_contribution.(samples))
std(derivative_contribution.(samples)) / 10

# mean(derivative_contribution.(samples_other))
# std(derivative_contribution.(samples_other)) / 10

begin
fig = Figure(resolution=(500,400), fontsize=20)
# ax = Axis(fig[1, 1]) 
# ax2 = Axis(fig[1, 3]) 
# ax3 = Axis(fig[1, 5]) 
j = 0
for i in 362:367 
    # ax = Axis(fig[1, j], ylabel="Primal")
    ax2 = Axis(fig[1 + (j ÷ 3), j%3], xlabel="t=$i", aspect=1)
    # hidespines!(ax)
    # hidespines!(ax2)
    # hidedecorations!(ax, label=(i != 353))
    hidedecorations!(ax2, label=(i!=362))
    ax2.xlabelvisible = true 
    orig = StochasticAD.value.(triple_configs[i])
    change = (st -> StochasticAD.PrunedFIsModule.pruned_value(st.Δs)).(triple_configs[i])
    alt = orig .+ change
    weights = (st -> st.Δs.state.valid * st.Δs.state.weight).(triple_configs[i])
    @show unique(weights)
    @assert length(unique(weights)) <= 2
    # hm1 = heatmap!(ax, orig, colorrange=(-1,1), colormap=:grays) 
    hm2 = heatmap!(ax2, alt, colorrange=(-1,1), colormap=:grays) 
    for i in 1:N
        for j in 1:N
            if change[i,j] != 0
                # scatter!(ax, [i], [j], color=(:white, 1.0), markersize=15)
                scatter!(ax2, [i], [j], color=(:red, 1.0), markersize=15)
            end
        end
    end
    # Colorbar(fig[1,j+2], hm1) 
    j += 1
end

# hm2 = heatmap!(ax2, change * try sort(unique(weights))[2] catch e 0 end, colorrange=(-10, 10))
# Colorbar(fig[1,4], hm2) 
# hm = heatmap!(ax3, weights, colorrange=(-1,1))
# Colorbar(fig[1,6], hm) 
save("plots/ising_plot_alts.png", fig; px_per_unit=4)
fig
end

sum(triple_configs[i])
triple_configs[i][4,8]
change[4,8]