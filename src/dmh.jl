"""
    mh(get_logpdf, proposal, x0; iters = 200, burn_in = 50, f, f_init=0.0, proposal_coupling = nothing, get_samples = Val(false))

Run Metropolis-Hastings with a given log PDF, proposal distribution, and initial state x0.
Return the finite-sample estimate of the posterior expectation `f`.
If `get_samples isa Val{true}`, also return the raw MH samples as the second element of a tuple.

This function is differentiable by StochasticAD, via stochastic derivatives, if a proposal coupling is provided.
"""
function mh(get_logpdf, proposal, x0; iters = 200, burn_in = 50, f, f_init=0.0, proposal_coupling = nothing, get_samples = Val(false))
    if get_samples isa Val{true}
        # TODO: fix below monstrosity to find correct type 
        fake_rng = copy(Random.default_rng())
        sampletype = [x0][0 + rand(fake_rng, Bernoulli(one(get_logpdf(x0))))] |> typeof
        samples = sampletype[]
    end
    x = x0
    S = f_init
    for i in 1:iters
        x_proposed = rand(MHProposalDistribution(x, proposal, proposal_coupling))
        logα = min(0.0, get_logpdf(x_proposed) - get_logpdf(x) + logratio_proposal(proposal, x, x_proposed))
        coin = rand(Bernoulli(exp(logα)))
        x = x + (x_proposed - x) * coin #[x, x_proposed][1 + coin]
        fx = f(x)
        if i > burn_in
            S += StochasticAD.structural_map(StochasticAD.smooth_triple, fx)
            if get_samples isa Val{true}
                push!(samples, x)
            end
        end
    end 
    avg = S / (iters - burn_in)
    if get_samples isa Val{true}
        return (avg, samples)
    else
        return avg
    end
end

"""
    mh_score(get_logpdf, proposal, x0; iters = 200, burn_in = 50, f, f_init=0.0)

Run Metropolis-Hastings with a given log PDF, proposal distribution, and initial state x0.
Return the finite-sample estimate of the posterior expectation `f`.
If `get_samples isa Val{true}`, also return the raw MH samples as the second element of a tuple.

This function is differentiable by ForwardDiff or StochasticAD, via the score method.
"""
function mh_score(get_logpdf, proposal, x0; iters = 200, burn_in = 50, f, f_init=0.0)
    x = x0
    S = f_init
    w = 0.0
    scoregrad = float.(init)
    for i in 1:iters
        x_proposed = rand(MHProposalDistribution(x, proposal, nothing)) # coupling irrelevant here
        logα = min(0.0, get_logpdf(x_proposed) - get_logpdf(x) + logratio_proposal(proposal, x, x_proposed))
        coin = rand(Bernoulli(exp(logα)))
        x = x + (x_proposed - x) * coin #[x, x_proposed][1 + coin]
        fx = f(x)
        prob = (StochasticAD.value(coin) == 1) ? exp(α) : 1-exp(α)
        w += StochasticAD.delta(log(prob))
        if i > burn_in
            S += StochasticAD.structural_map(StochasticAD.smooth_triple, fx)
            scoregrad += w * StochasticAD.structural_map(StochasticAD.value, fx)
        end
    end 
    Sdual = StochasticAD.structural_map((val, delta) -> ForwardDiff.Dual(val, delta), StochasticAD.value.(S), scoregrad)
    return Sdual / (iters - burn_in) 
end

"""
    mh_basic_kernel_init(get_logpdf, proposal, x0; iters = 200, proposal_coupling = nothing)

Initialize basic Markov kernel for the MH sample state. See also `mh_basic_kernel`.
"""
function mh_basic_kernel_init(get_logpdf, proposal, x0; iters = 200, proposal_coupling = nothing)
    return x0, iters, (; get_logpdf, proposal, proposal_coupling)
end

"""
    mh_basic_kernel(x, kernel_params)

Apply basic Markov kernel for the MH sample state. See also `mh_basic_kernel_init`.
"""
function mh_basic_kernel(x, kernel_params)
    (; get_logpdf, proposal, proposal_coupling) = kernel_params
    x_proposed = rand(MHProposalDistribution(x, proposal, proposal_coupling))
    logα = min(0.0, get_logpdf(x_proposed) - get_logpdf(x) + logratio_proposal(proposal, x, x_proposed))
    coin = rand(Bernoulli(exp(logα)))
    x = x + (x_proposed - x) * coin #[x, x_proposed][1 + coin]
    return x
end

"""
    mh_kernel_init(get_logpdf, proposal, x0; iters, burn_in, f, f_init=0.0, proposal_coupling = nothing)

Initialize basic Markov kernel for the MH sample state, with augmentation to accumulate finite-sample expectations. 
See also `mh_kernel`.
"""
function mh_kernel_init(args...; burn_in, f, f_init=0.0, mh_basic_kernel_init = mh_basic_kernel_init, kwargs...)
    x0, n, mh_kernel_params = mh_basic_kernel_init(args...; kwargs...) 
    return (x0, f_init, 0), n, (; mh_kernel_params, f, burn_in)
end

"""
    mh_kernel(x, kernel_params)

Apply basic Markov kernel for the MH sample state, with augmentation to accumulate finite-sample expectations. 
See also `mh_kernel`.
"""
function mh_kernel((x, S, i), kernel_params; mh_basic_kernel = mh_basic_kernel)
    (; mh_kernel_params, f, burn_in) = kernel_params
    x = mh_basic_kernel(x, mh_kernel_params)
    # Update posterior estimate accumulator 
    smooth_triple_map = Base.Fix1(StochasticAD.structural_map, StochasticAD.smooth_triple)
    i += 1
    if i > burn_in 
        # deepcopy preserves aliasing relationships within x, but ensures that pruning occuring during the application of f
        # does not affect x itself. This is valid because x and f(x) will never interact (indeed, f(x) is immediately smoothed).
        #
        # TODO: the composition of f and smoothing, which permits a deepcopy on the inside, should have its own construct, which is 
        # always legal as its own block due to the outer smoothing?
        S += smooth_triple_map(f(deepcopy(x)))
    end
    return (x, S, i)
end

function mh_f((_, S, _); iters, burn_in)
    return S / (iters - burn_in)
end