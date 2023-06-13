"""
    mh(get_logpdf, rand_proposal, x0; iters = 200, burn_in = 50, f, init=0.0)

Run Metropolis-Hastings with a given log PDF, proposal distribution, and initial state x0.
The provided f computes quantities for which derivatives are desired.
Return the MH samples and the sum S of f over the samples.

This function is differentiable by StochasticAD, via stochastic derivatives.
"""
function mh(get_logpdf, rand_proposal, x0; iters = 200, burn_in = 50, f, init=0.0)
    # TODO: fix below monstrosity to find correct type 
    sampletype = [x0][0 + rand(Bernoulli(one(get_logpdf(x0))))] |> typeof
    samples = sampletype[]
    x = x0
    S = init
    for i in 1:iters
        x_proposed = rand_proposal(x)
        α = min(0.0, get_logpdf(x_proposed) - get_logpdf(x))
        coin = rand(Bernoulli(exp(α)))
        x = [x, x_proposed][1 + coin]
        fx = f(x)
        if i > burn_in
            S += StochasticAD.structural_map(StochasticAD.smooth_triple, fx)
            push!(samples, x)
        end
    end 
    return (;samples, S)
end

"""
    mh_score(get_logpdf, rand_proposal, x0; iters = 200, burn_in = 50, f, init=0.0)

Run Metropolis-Hastings with a given log PDF, proposal distribution, and initial state x0.
The provided f computes quantities for which derivatives are desired.
Return the MH samples and the sum S of f over the samples.

This function is differentiable by ForwardDiff or StochasticAD, via the score method.
"""
function mh_score(get_logpdf, rand_proposal, x0; iters = 200, burn_in = 50, f, init=0.0)
    # TODO: fix below monstrosity to find correct type 
    sampletype = [x0][0 + rand(Bernoulli(one(get_logpdf(x0))))] |> typeof
    samples = sampletype[]
    x = x0
    S = init
    w = 0.0
    scoregrad = float.(init)
    for i in 1:iters
        x_proposed = rand_proposal(x)
        α = min(0.0, get_logpdf(x_proposed) - get_logpdf(x))
        coin = rand(Bernoulli(exp(α)))
        x = [x, x_proposed][1 + coin]
        fx = f(x)
        prob = (StochasticAD.value(coin) == 1) ? exp(α) : 1-exp(α)
        w += StochasticAD.delta(log(prob))
        if i > burn_in
            S += StochasticAD.structural_map(StochasticAD.smooth_triple, fx)
            scoregrad += w * StochasticAD.structural_map(StochasticAD.value, fx)
            push!(samples, x)
        end
    end 
    Sduals = StochasticAD.structural_map((val, delta) -> ForwardDiff.Dual(val, delta), StochasticAD.value.(S), scoregrad)
    return (;samples, S=Sduals)
end