module Analysis
    using StochasticAD
    using ForwardDiff
    using DataFrames
    using ProgressMeter
    using Statistics
    using LinearAlgebra
    import Random

    include("analyze_problem.jl")
    export take_samples, get_asymptotics
    
    include("analyze_markov_problem.jl")
    export MarkovX, get_raw_chain_slim
end