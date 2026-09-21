module Analysis
    using StochasticAD
    using ForwardDiff
    using DataFrames
    using ProgressMeter
    using Statistics
    using LinearAlgebra
    using BenchmarkTools
    import Random

    include("analyze_problem.jl")
    export take_samples, get_asymptotics
    
    include("analyze_markov_problem.jl")
    export MarkovX, get_primal_chain_slim, get_raw_chain_slim,
        get_primal_timing, get_derivative_timing,
        _get_chain_slim, _get_chain_full
end
