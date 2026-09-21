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

function _get_chain_full(X::MarkovX, p, settings, options)
    x, n, kernel_params = X.kernel_init(p, settings, options)
    samples = Any[first(x)]
    seeds = [rand(UInt32) for i in 1:n]
    for i in 1:n
        Random.seed!(seeds[i])
        x = X.kernel(x, kernel_params)
        samples = push!(samples, first(x))
    end
    ret = X.f(x, settings, options)
    return (; chain = collect(samples), seeds, ret, kernel_params, n)
end

function get_primal_chain_slim(problem; target, options=NamedTuple(), get_chain = _get_chain_slim)
    X::MarkovX = problem.targets[target].X
    p = problem.settings.p
    settings = problem.settings

    raw, duration = let start = time_ns()
        raw = get_chain(X, p, settings, options)
        raw, (time_ns() - start) / 1e9
    end
    (; chain, seeds, ret, kernel_params, n) = raw
    return (; chain, seeds, ret, target, alg_id = "primal", kernel_params, n, X, p, settings, options, duration)
end

function get_raw_chain_slim(problem; target, alg_id, options=NamedTuple(), get_chain = _get_chain_slim)
    X::MarkovX = problem.targets[target].X
    p = problem.settings.p
    settings = problem.settings
    discrete_algs = Analysis.get_discrete_algs()
    alg = discrete_algs[alg_id]

    raw, duration = let start = time_ns()
        raw = stochastic_triple(p -> get_chain(X, p, settings, options), p; backend = alg.backend)
        raw, (time_ns() - start) / 1e9
    end
    (; chain, seeds, ret, kernel_params, n) = raw
    return (; chain, seeds, ret, target, alg_id, kernel_params, n, X, p, settings, options, duration)
end

function get_primal_timing(problem; target, options=NamedTuple())
    settings = problem.settings
    n = settings.n

    X::MarkovX = problem.targets[target].X
    p = problem.settings.p
    primal_run = () -> X(p, settings, options)
    benchmark = BenchmarkTools.@benchmark $primal_run()
    estimate = median(benchmark)
    duration = estimate.time / 1e9
    return (; duration, ns = estimate.time / n, n, benchmark)
end

function get_derivative_timing(problem; target, backend, options=NamedTuple())
    settings = problem.settings
    n = settings.n

    X::MarkovX = problem.targets[target].X
    p = problem.settings.p
    derivative_X = q -> X(q, settings, options)
    derivative_run = () -> stochastic_triple(
        derivative_X, p; backend = backend)
    benchmark = BenchmarkTools.@benchmark $derivative_run()
    estimate = median(benchmark)
    duration = estimate.time / 1e9
    return (; duration, ns = estimate.time / n, n, benchmark)
end
