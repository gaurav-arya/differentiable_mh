struct MarkovX{K,KI,F}
    kernel::K
    kernel_init::KI
    f::F 
end

function (X::MarkovX)(p, settings, options=NamedTuple())
    x, n, kernel_params = X.kernel_init(p, settings, options)
    for i in 1:n
        x = X.kernel(x, kernel_params) # TODO: allow f on intermediary values
    end
    return X.f(x, settings, options)
end

function _get_chain_slim(X::MarkovX, p, settings, options)
    x, n, kernel_params = X.kernel_init(p, settings, options)
    samples = [StochasticAD.value.(first(x))]
    seeds = [rand(UInt32) for i in 1:n]
    for i in 1:n
        Random.seed!(seeds[i])
        x = X.kernel(x, kernel_params)
        samples = push!(samples, StochasticAD.value.(first(x)))
    end
    ret = X.f(x, settings, options)
    return (; chain = samples, seeds, ret, kernel_params, n)
end

function get_raw_chain_slim(problem; target, alg_id, options=NamedTuple())
    X::MarkovX = problem.targets[target].X
    p = problem.settings.p
    settings = problem.settings
    discrete_algs = Analysis.get_discrete_algs()
    alg = discrete_algs[alg_id]

    (; chain, seeds, ret, kernel_params, n) = stochastic_triple(p -> _get_chain_slim(X, p, settings, options), p; backend = alg.backend)
    return (; chain, seeds, ret, target, alg_id, kernel_params, n, X, p, settings, options)
end