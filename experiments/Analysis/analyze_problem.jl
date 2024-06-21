function get_discrete_algs(; alg_flags = nothing)
    isnothing(alg_flags) && (alg_flags = ["pruning", "reweighting", "mvd", "smoothing", "no_pruning", "enumeration", "reverse", "uniform"])
    pruning_alg = (; backend=PrunedFIsBackend(), name="Pruning", flags = ["pruning"])
    # pruning_alg = (; backend=PrunedCustomFIsBackend(Val(:weights)), name="Pruning", flags = ["pruning"])
    st_alg = (; backend=StrategyWrapperFIsBackend(SmoothedFIsBackend(), StochasticAD.SmoothedStraightThroughStrategy()), name="Straight Through", flags = ["smoothing"])
    ss_alg = (; backend=StrategyWrapperFIsBackend(SmoothedFIsBackend(), StochasticAD.SingleSidedStrategy()), name="Straight Through Single-sided", flags = ["smoothing"])
    mvd_alg = (; backend=StrategyWrapperFIsBackend(PrunedFIsBackend(), StochasticAD.StraightThroughStrategy()), name="Pruning MVD", flags = ["pruning", "mvd"])
    mvd_uniform_alg = (; backend=StrategyWrapperFIsBackend(PrunedFIsBackend(Val(:wins)), StochasticAD.StraightThroughStrategy()), name="Pruning Uniformly MVD", flags = ["pruning", "mvd", "uniform"])
    pruning_uniform_alg = (; backend=PrunedFIsBackend(Val(:wins)), name="Pruning Uniformly", flags = ["pruning", "uniform"])
    pruning_uniform_rev_alg = (; backend=PrunedFIsBackend(Val(:wins)), name="Pruning uniformly reverse", flags = ["pruning", "reverse", "uniform"], mode = "reverse")
    all_discrete_algs = Dict(
        "pruning" => pruning_alg,
        "straight_through" => st_alg,
        "straight_through_ss" => ss_alg,
        "pruning_mvd" => mvd_alg,
        "pruning_uniform" => pruning_uniform_alg,
        "pruning_uniform_rev" => pruning_uniform_rev_alg, 
        "pruning_uniform_mvd" => mvd_uniform_alg, 
    )

    discrete_algs = filter((alg_name, alg)::Pair -> all(flag -> flag in alg_flags, alg.flags), all_discrete_algs)
    return discrete_algs
end

function get_continuous_algs()
    continuous_alg = (; backend = StrategyWrapperFIsBackend(SmoothedFIsBackend(), StochasticAD.IgnoreDiscreteStrategy()), name = "Derivative")
    continuous_algs = Dict("derivative" => continuous_alg) # TODO: add reverse mode here
    return continuous_algs
end

function take_samples(problem; nsims = 1000, discrete_alg_flags = nothing, target_flags = nothing, store_samples = false, options = (;))
    (; targets, settings) = problem
    isnothing(target_flags) && (target_flags = ["continuous", "deterministic", "enumerate"])
    # exclude enumeration and no pruning in default discrete_alg_flags for sampling 
    isnothing(discrete_alg_flags) && (discrete_alg_flags = ["pruning", "reweighting", "mvd", "smoothing"])

    # Hack to get out a single specific algorithm here
    if discrete_alg_flags isa String
        discrete_algs = [discrete_alg_flags => get_discrete_algs()[discrete_alg_flags]]
    else
        discrete_algs = get_discrete_algs(alg_flags = discrete_alg_flags)
    end
    continuous_algs = get_continuous_algs()

    data = DataFrame(alg_name = String[], target_name = String[], mean = Float64[], std = Float64[], stderr = Float64[], alg_id = String[], target_id = String[], alg = Any[], target = Any[], samples = Any[]) 

    primal_seeds = [Int(rand(UInt32)) for i in 1:nsims]

    primal = targets["primal"]
    row = let
        samples = @showprogress [begin Random.seed!(seed); primal.X(settings.p, settings, options) end for seed in primal_seeds]
        _mean = mean(samples)
        _std = std(samples)
        alg_id, alg = "primal", (; name = "Primal", flags = [])
        (; alg_name = alg.name, target_name = primal.name, mean = _mean, std = _std, stderr = _std / sqrt(nsims), alg_id, target_id = "primal", alg, target=primal,
        samples = store_samples ? samples : missing)
    end
    avg = row.mean 
    push!(data, row)

    if "enumerate" in keys(targets)
        enumerate = targets["enumerate"]
        if all(flag -> flag in target_flags, enumerate.flags)
            row = let
                _mean = enumerate.X(settings.p, settings, options) 
                _std = 0 
                alg_id, alg = "primal", (; name = "Primal", flags = [])
                (; alg_name = alg.name, target_name = enumerate.name, mean = _mean, std = _std, stderr = _std / sqrt(nsims), alg_id, target_id = "enumerate", alg, target=enumerate,
                samples = missing)
            end
            avg = row.mean 
            push!(data, row)
        end
    end

    options = merge((; avg), options)

    for (target_id, target) in targets
        !all(flag -> flag in target_flags, target.flags) && continue
        algs = "continuous" in target.flags ? continuous_algs : discrete_algs 
        # TODO: check for a primal flag (not name), in which case we add the primal alg to the list as well. 
        for (alg_id, alg) in algs
            @show target.name alg.name
            _mean, _std, _stdstd = if "deterministic" in target.flags
                # use ForwardDiff for now
                ForwardDiff.derivative(p -> target.X(p, settings, options), settings.p), 0.0, 0.0
                # derivative_estimate(p -> target.X(p, settings, options), settings.p; backend = alg.backend), 0.0
            else
                samples = if "continuous" in target.flags
                    # use ForwardDiff for now
                    @showprogress [begin Random.seed!(seed); ForwardDiff.derivative(p -> target.X(p, settings, options), settings.p) end for seed in primal_seeds]
                else
                    if !haskey(alg, :mode) || alg.mode == "forward"
                        ad_alg = StochasticAD.ForwardAlgorithm(alg.backend)
                    elseif alg.mode =="reverse"
                        ad_alg = StochasticAD.EnzymeReverseAlgorithm(alg.backend)
                    else
                        error("Unrecognized mode $(alg.mode) for algorithm $alg.")
                    end
                    to_diff = let settings = settings, options = options
                        p -> target.X(p, settings, options)
                    end
                    @showprogress [begin Random.seed!(seed); derivative_estimate(to_diff, settings.p, ad_alg) end for seed in primal_seeds]
                end
                mean(samples), std(samples), std((samples .- mean(samples)).^2)
            end
            row = (; alg_name = alg.name, target_name = target.name, mean = _mean, std = _std, stderr = _std / sqrt(nsims), alg_id, target_id, alg, target,
                    samples = store_samples ? samples : missing)
            push!(data, row)
        end
    end

    return sort(data, :std)
end

function get_asymptotics(ns, make_problem; nsims = 1000, discrete_alg_flags = nothing, target_flags = nothing, options = (;))

    asymptotics_data = DataFrame()

    for n in ns
        @info "Asymptotics for n = $n"
        data = take_samples(make_problem(n); nsims, discrete_alg_flags, target_flags, options)
        insertcols!(data, 1, (:n => n))
        asymptotics_data = vcat(asymptotics_data, data)
    end

    return asymptotics_data
end