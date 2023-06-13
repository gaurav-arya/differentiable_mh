module IsingCore

using Random, LinearAlgebra, Statistics, StatsBase, Printf, Distributions
using StochasticAD

function initial_state(N)
    # Generates a fixed spin configuration as initial condition

    # cold start
    state = ones(N, N)
    return state
end

function mc_move(config, beta, N)
    # Monte Carlo move using Metropolis

    range_i = shuffle!(collect(1:N))
    range_j = shuffle!(collect(1:N))
    for i in range_i
        for j in range_j
            s = config[i, j]
            nb = config[mod1(i + 1, N), j] + config[i, mod1(j + 1, N)] +
                 config[mod1(i - 1, N), j] + config[i, mod1(j - 1, N)]
            cost = 2s * nb # get_logpdf(x_proposed) - get_logpdf(x)

            flip_prob = min(1.0, exp(-cost * beta))
            coin = rand(Bernoulli(flip_prob))
            config[i, j] = [s, -s][1+coin]
        end
    end

    return config
end

function mc_move_c(config, beta, N)
    # Monte Carlo move using Metropolis

    range_i = shuffle!(collect(1:N))
    range_j = shuffle!(collect(1:N))
    for i in range_i
        for j in range_j
            s = config[i, j]
            s2 = rand((-1.0, 1.0))
            nb = config[mod1(i + 1, N), j] + config[i, mod1(j + 1, N)] +
                 config[mod1(i - 1, N), j] + config[i, mod1(j - 1, N)]
            cost = (s - s2) * nb # get_logpdf(x_proposed) - get_logpdf(x)
            # old proposal which always flips
            # s = config[i, j]
            # nb = config[mod1(i + 1, N), j] + config[i, mod1(j + 1, N)] +
            #      config[mod1(i - 1, N), j] + config[i, mod1(j - 1, N)]
            # cost = 2s * nb # get_logpdf(x_proposed) - get_logpdf(x)

            flip_prob = min(1.0, exp(-cost * beta))
            coin = rand(Bernoulli(flip_prob))
            # configᵒ = copy(config)
            # configᵒ[i, j] = -s
            # config = [config, configᵒ][1+coin]
            # s = coin == 1 ? -s : s
            config[i, j] = [s, s2][1+coin]
        end
    end

    return config
end

# couple accept/reject steps
function mc_move_cc(config, beta, N, acceptreject)
    # Monte Carlo move using Metropolis

    range_i = shuffle!(collect(1:N))
    range_j = shuffle!(collect(1:N))
    for i in range_i
        for j in range_j
            s = config[i, j]
            nb = config[mod1(i + 1, N), j] + config[i, mod1(j + 1, N)] +
                 config[mod1(i - 1, N), j] + config[i, mod1(j - 1, N)]
            cost = 2 * s * nb # get_logpdf(x_proposed) - get_logpdf(x)

            flip_prob = min(1.0, exp(-cost * beta))
            coin = acceptreject(s, flip_prob) #rand(Bernoulli(flip_prob))
            # configᵒ = copy(config)
            # configᵒ[i, j] = -s
            # config = [config, configᵒ][1+coin]
            # s = coin == 1 ? -s : s
            config[i, j] = [s, -s][1+coin]
        end
    end

    return config
end
acceptreject(_, flip_prob) = rand(Bernoulli(flip_prob))
function acceptreject(s_st::StochasticAD.StochasticTriple{T}, flip_prob_st::StochasticAD.StochasticTriple{T}) where {T}
    @assert iszero(StochasticAD.delta(s_st))
    # @assert iszero(StochasticAD.delta(flip_prob_st))
    s = StochasticAD.value(s_st)
    flip_prob = StochasticAD.value(flip_prob_st)
    ω = rand()
    accept = convert(Signed, ω < flip_prob)
    Δs_coupled = StochasticAD.couple((s_st.Δs, flip_prob_st.Δs); out_rep = (s, flip_prob))
    Δs1 = StochasticAD.δtoΔs(Bernoulli(flip_prob), accept, StochasticAD.delta(flip_prob_st), flip_prob_st.Δs)
    Δs2 = map(Δs_coupled) do (s_Δ, flip_prob_Δ)
        alt_flip_prob = flip_prob + flip_prob_Δ
        reverse = xor(abs(flip_prob - alt_flip_prob) < abs(1 - flip_prob - alt_flip_prob), s_Δ == 0)
        if reverse 
            accept_alt = ω > 1 - alt_flip_prob
        else 
            accept_alt = ω < alt_flip_prob
            # @assert max(alt_flip_prob, flip_prob) ≈ 1.0
            # p = min(flip_prob, alt_flip_prob)
            # accept_alt = ω < alt_flip_prob
        end
        # if accept_alt != accept
        #     # @show flip_prob alt_flip_prob s_Δ accept accept_alt
        # end
        return accept_alt - accept
    end
    return StochasticAD.StochasticAD.StochasticTriple{T}(accept, zero(accept), StochasticAD.combine((Δs1, Δs2)))
end

function calc_energy(config, N)
    energy = 0.0

    for i in 1:N
        for j in 1:N
            S = config[i, j]
            nb = config[mod1(i + 1, N), j] + config[i, mod1(j + 1, N)] +
                 config[mod1(i - 1, N), j] + config[i, mod1(j - 1, N)]
            energy -= nb * S
        end
    end
    return energy / 2.0  # To compensate for over-counting
end

function calc_mag(config)
    # Magnetization of a given configuration
    mag = sum(config) / length(config)^2
    return mag
end

function ising_model(N, T, _config, move)
    # samples spin configurations for the Ising model given an initial configuration _config and a temperature T

    eqSteps = 10^2  # number of MC sweeps for equilibration
    mcSteps = 10^3  # number of MC sweeps for calculation

    E1 = M1 = E2 = M2 = 0
    iT = 1.0 / T # inverse temperature
    iT2 = iT * iT

    # convert type to StochasticAD.StochasticTriples..
    config = _config + _config * T * 0

    configs = [copy(config)]

    for i in 1:eqSteps  # equilibration phase
        config = move(config, iT, N)
        push!(configs, copy(config))
    end

    for i in 1:mcSteps  # data generation phase
        config = move(config, iT, N)
        ene = calc_energy(config, N)
        mag = abs(calc_mag(config))

        E1 += StochasticAD.smooth_triple(ene)
        M1 += StochasticAD.smooth_triple(mag)
        E2 += StochasticAD.smooth_triple(ene^2)
        M2 += StochasticAD.smooth_triple(mag^2)

        push!(configs, copy(config))
    end

    E = E1 / mcSteps
    M = M1 / mcSteps
    C = (E2 / mcSteps - (E1 / mcSteps)^2) * iT2
    X = (M2 / mcSteps - (M1 / mcSteps)^2) * iT
    return E, M, C, X, configs
end

function ising_model_manyT(N, move, T=nothing)
    T_min = 0.05
    T_max = 10.0

    (T === nothing) && (T = range(T_min, T_max, length=50))
    nt = length(T)

    E, M, C, X = zeros(nt), zeros(nt), zeros(nt), zeros(nt)

    config = initial_state(N)  # initialization
    for tt in 1:nt
        println(T[tt])
        Et, Mt, Ct, Xt, _ = ising_model(N, T[tt], config, move)
        E[tt] = Et
        M[tt] = Mt
        C[tt] = Ct
        X[tt] = Xt
    end
    return T, E, M, C, X
end

export initial_state, mc_move, mc_move_c, mc_move_cc, calc_energy, calc_mag, ising_model, ising_model_manyT, acceptreject

end