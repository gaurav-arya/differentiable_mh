module DataContaminationProblem 

# FIXME: this is to a large degree a copy of the GaussianMHProblem setup,
# so it feels like we can probably refactor to something general

__precompile__(false)

using Distributions
using StochasticAD
using LinearAlgebra
using DifferentiableMH
import Analysis: MarkovX

function mh_basic_kernel_no_log(x, kernel_params)
    # as a hack, we have replaced get_logpdf with get_pdf 
    (; get_logpdf, proposal, proposal_coupling) = kernel_params
    get_pdf = get_logpdf 
    x_proposed = rand(MHProposalDistribution(x, proposal, proposal_coupling))
    α = min(1.0, get_pdf(x_proposed) / get_pdf(x))
    coin = rand(Bernoulli(α))
    x = x + (x_proposed - x) * coin #[x, x_proposed][1 + coin]
    if 1 < StochasticAD.value.(x)[1] < 3
        error(x)
    end
    return x
end

function X_kernel_init(θ, settings, options = (;))
    (; model_logpdf, n, f, burn_in, init, proposal, proposal_coupling) = settings
    return mh_kernel_init(x -> model_logpdf(x,θ), proposal, init; f, iters=n, burn_in, f_init=zero(f(init)), proposal_coupling)
end
    
function X_kernel(x_aug, kernel_params)
    return mh_kernel(x_aug, kernel_params; mh_basic_kernel = mh_basic_kernel_no_log)
end

function X_f(x_aug, settings, options = (;))
    (; n, burn_in) = settings
    return mh_f(x_aug; iters=n, burn_in)
end

function make_data_contamination_problem(model_pdf, init, n=10000;
        f = identity, burn_in = nothing, theta = 1e-6,
        proposal = RandomWalkMHProposal{typeof(init)}(MvNormal(zero(init), 2)))
    targets = Dict(
        "primal" => (; X = MarkovX(X_kernel, X_kernel_init, X_f), flags = [], name = "Primal")
    )
    burn_in = isnothing(burn_in) ? n ÷ 2 : burn_in
    settings = (; model_logpdf = model_pdf, n, p = theta, f, burn_in, init, proposal, proposal_coupling = MaximumReflectionProposalCoupling())
    return (; targets, settings)
end

export make_data_contamination_problem 

end