"""Stage-III Thin et al. PPCA gradient diagnostic.

The Python source repository is used only to prepare a portable artifact. The
diagnostic itself is a Julia program. Its DMH estimators are explicit, while
DifferentiableMH is retained for validating the MALA parameterization.

Examples:

    JULIA_DEPOT_PATH=/tmp/dmh_julia_depot:/home/rubense/.julia \
    /opt/julia-1.10.2/bin/julia --project=experiments \
        experiments/mcvae/ppca_diagnostic/run_ppca.jl --self-test

    JULIA_DEPOT_PATH=/tmp/dmh_julia_depot:/home/rubense/.julia \
    /opt/julia-1.10.2/bin/julia --project=experiments \
        experiments/mcvae/ppca_diagnostic/run_ppca.jl --artifact=... \
        --quick --no-write
"""

using DelimitedFiles
using DifferentiableMH
using Distributions
using ForwardDiff
using Functors
using LinearAlgebra
using Printf
using Random
using Statistics
using TOML

const OUTPUT_DIR = @__DIR__
const DEFAULT_SEED = 20260818
const SOURCE_EPSILON = 0.003
const JULIA_MALA_GAMMA = 2SOURCE_EPSILON
const TARGET_COORDINATE_ZERO_BASED = 18
const DEFAULT_REPETITIONS = 200
const DEFAULT_PILOT_REPETITIONS = 200
const DEFAULT_DMH_REPETITIONS = 200

"""Mutable accounting shared by the direct and DMH implementations."""
mutable struct CostStats
    transitions::Int
    logdensity_evals::Int
    gradient_evals::Int
    spawned_alternatives::Int
    meeting_opportunities::Int
    meetings::Int
    accepted::Int
    decisions::Int
    primary_accepted::Int
    primary_decisions::Int
    alternative_accepted::Int
    alternative_decisions::Int
end

CostStats() = CostStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

function Base.:+(left::CostStats, right::CostStats)
    return CostStats(
        left.transitions + right.transitions,
        left.logdensity_evals + right.logdensity_evals,
        left.gradient_evals + right.gradient_evals,
        left.spawned_alternatives + right.spawned_alternatives,
        left.meeting_opportunities + right.meeting_opportunities,
        left.meetings + right.meetings,
        left.accepted + right.accepted,
        left.decisions + right.decisions,
        left.primary_accepted + right.primary_accepted,
        left.primary_decisions + right.primary_decisions,
        left.alternative_accepted + right.alternative_accepted,
        left.alternative_decisions + right.alternative_decisions,
    )
end

function add_cost!(target::CostStats, source::CostStats)
    target.transitions += source.transitions
    target.logdensity_evals += source.logdensity_evals
    target.gradient_evals += source.gradient_evals
    target.spawned_alternatives += source.spawned_alternatives
    target.meeting_opportunities += source.meeting_opportunities
    target.meetings += source.meetings
    target.accepted += source.accepted
    target.decisions += source.decisions
    target.primary_accepted += source.primary_accepted
    target.primary_decisions += source.primary_decisions
    target.alternative_accepted += source.alternative_accepted
    target.alternative_decisions += source.alternative_decisions
    return target
end

const ACTIVE_COST = Ref{Union{Nothing,CostStats}}(nothing)

function count_logdensity!()
    cost = ACTIVE_COST[]
    isnothing(cost) || (cost.logdensity_evals += 1)
    return nothing
end

function count_gradient!()
    cost = ACTIVE_COST[]
    isnothing(cost) || (cost.gradient_evals += 1)
    return nothing
end

"""Portable model/data bundle written by prepare_ppca.py."""
struct PPCAArtifact
    W::Matrix{Float64}
    b::Vector{Float64}
    encoder_W::Matrix{Float64}
    encoder_b::Vector{Float64}
    x::Matrix{Float64}
    labels::Vector{Float64}
    q_mu::Matrix{Float64}
    q_logvar::Matrix{Float64}
    z_n1::Matrix{Float64}
    z_n10::Matrix{Float64}
    sigma::Float64
    metadata::Dict{String,Any}
end

struct PPCAConfig
    K::Int
    gamma::Float64
    coordinate_zero_based::Int
end

PPCAConfig(K::Int; gamma=JULIA_MALA_GAMMA,
           coordinate_zero_based=TARGET_COORDINATE_ZERO_BASED) =
    PPCAConfig(K, float(gamma), coordinate_zero_based)

struct PPCAModel{W,B,S}
    W::W
    b::B
    sigma::S
end

struct ParameterizedPPCA{M,T}
    base::M
    coordinate::Int
    theta::T
end

Functors.@functor ParameterizedPPCA

@inline model_bias(model::PPCAModel, index) = model.b[index]
@inline function model_bias(model::ParameterizedPPCA, index)
    return index == model.coordinate ? model.theta : model.base.b[index]
end

@inline model_weight(model::PPCAModel) = model.W
@inline model_weight(model::ParameterizedPPCA) = model.base.W
@inline model_sigma(model::PPCAModel) = model.sigma
@inline model_sigma(model::ParameterizedPPCA) = model.base.sigma

function read_matrix(path::AbstractString)
    data = readdlm(path, ',', Float64)
    return Array{Float64}(data)
end

function read_vector(path::AbstractString)
    return vec(read_matrix(path))
end

function load_artifact(directory::AbstractString)
    metadata = TOML.parsefile(joinpath(directory, "metadata.toml"))
    W = read_matrix(joinpath(directory, "decoder_weight.csv"))
    b = read_vector(joinpath(directory, "decoder_bias.csv"))
    encoder_W = read_matrix(joinpath(directory, "encoder_weight.csv"))
    encoder_b = read_vector(joinpath(directory, "encoder_bias.csv"))
    x = read_matrix(joinpath(directory, "x.csv"))
    labels = read_vector(joinpath(directory, "labels.csv"))
    q_mu = read_matrix(joinpath(directory, "q_mu.csv"))
    q_logvar = read_matrix(joinpath(directory, "q_logvar.csv"))
    z_n1 = read_matrix(joinpath(directory, "z_n1.csv"))
    z_n10 = read_matrix(joinpath(directory, "z_n10.csv"))
    sigma = Float64(metadata["sigma"])

    artifact = PPCAArtifact(
        W, b, encoder_W, encoder_b, x, labels, q_mu, q_logvar,
        z_n1, z_n10, sigma, metadata,
    )
    validate_artifact(artifact; strict=true)
    return artifact
end

"""Make a small full-dimensional view for the --quick smoke run."""
function quick_artifact(artifact::PPCAArtifact, limit::Int)
    n = size(artifact.x, 1)
    limit >= n && return artifact
    @assert limit > 0
    z_rows = vcat([
        artifact.z_n10[(particle - 1) * n + 1:(particle - 1) * n + limit, :]
        for particle in 1:10
    ]...)
    metadata = copy(artifact.metadata)
    delete!(metadata, "batch_size")
    return PPCAArtifact(
        artifact.W, artifact.b, artifact.encoder_W, artifact.encoder_b,
        artifact.x[1:limit, :], artifact.labels[1:limit],
        artifact.q_mu[1:limit, :], artifact.q_logvar[1:limit, :],
        artifact.z_n1[1:limit, :], z_rows, artifact.sigma, metadata,
    )
end

function validate_artifact(artifact::PPCAArtifact; strict=false)
    p, d = size(artifact.W)
    strict && @assert (p, d) == (784, 100)
    @assert (p, d) == (size(artifact.x, 2), size(artifact.q_mu, 2))
    @assert length(artifact.b) == p
    @assert size(artifact.x, 1) == size(artifact.q_mu, 1)
    @assert size(artifact.q_mu) == size(artifact.q_logvar)
    @assert size(artifact.z_n1) == (size(artifact.x, 1), d)
    @assert size(artifact.z_n10) == (10 * size(artifact.x, 1), d)
    @assert size(artifact.encoder_W) == (2d, p)
    @assert length(artifact.encoder_b) == 2d
    @assert artifact.sigma > 0
    if haskey(artifact.metadata, "latent_dim")
        @assert Int(artifact.metadata["latent_dim"]) == d
    end
    if haskey(artifact.metadata, "observation_dim")
        @assert Int(artifact.metadata["observation_dim"]) == p
    end
    if haskey(artifact.metadata, "batch_size")
        @assert Int(artifact.metadata["batch_size"]) == size(artifact.x, 1)
    end
    return artifact
end

function synthetic_artifact(; seed=DEFAULT_SEED, batch_size=2, p=5, d=3)
    rng = Xoshiro(seed)
    W = randn(rng, p, d) / sqrt(d)
    b = randn(rng, p) / 3
    encoder_W = randn(rng, 2d, p) / sqrt(p)
    encoder_b = randn(rng, 2d) / 5
    x = randn(rng, batch_size, p)
    labels = zeros(batch_size)
    q_mu = randn(rng, batch_size, d) / 3
    q_logvar = fill(-0.4, batch_size, d)
    z_n1 = q_mu + exp.(q_logvar ./ 2) .* randn(rng, batch_size, d)
    z_n10 = vcat([
        q_mu + exp.(q_logvar ./ 2) .* randn(rng, batch_size, d)
        for _ in 1:10
    ]...)
    metadata = Dict{String,Any}("synthetic" => true, "sigma" => 0.4)
    return PPCAArtifact(
        W, b, encoder_W, encoder_b, x, labels, q_mu, q_logvar,
        z_n1, z_n10, 0.4, metadata,
    )
end

@inline function dot_generic(left, right)
    return sum(eachindex(left)) do index
        left[index] * right[index]
    end
end

"""Fast matrix-vector products for ordinary and ForwardDiff values."""
function matrix_vector(W::Matrix{Float64}, z::AbstractVector{Float64})
    return W * z
end

function matrix_vector(W::Matrix{Float64}, z)
    return [
        sum(1:size(W, 2)) do column
            W[row, column] * z[column]
        end
        for row in 1:size(W, 1)
    ]
end

function transpose_matrix_vector(W::Matrix{Float64}, x::AbstractVector{Float64})
    return transpose(W) * x
end

function transpose_matrix_vector(W::Matrix{Float64}, x)
    return [
        sum(1:size(W, 1)) do row
            W[row, column] * x[row]
        end
        for column in 1:size(W, 2)
    ]
end

"""Reusable Float64 scratch space for explicit counterfactual transitions."""
mutable struct PPCAFloatWorkspace
    decoded::Vector{Float64}
    residual::Vector{Float64}
    score::Vector{Float64}
    proposal_score::Vector{Float64}
    diff::Vector{Float64}
    primal_jump::Vector{Float64}
    reflected::Vector{Float64}
    inv_variance::Vector{Float64}
    logvar_sum::Float64
end

function PPCAFloatWorkspace(model::PPCAModel, logvar::Vector{Float64})
    p, d = size(model.W)
    return PPCAFloatWorkspace(
        zeros(p), zeros(p), zeros(d), zeros(d), zeros(d), zeros(d), zeros(d),
        exp.(-logvar), sum(logvar),
    )
end

function workspace_log_q(workspace::PPCAFloatWorkspace, z, mu)
    quadratic = 0.0
    @inbounds for index in eachindex(z, mu, workspace.inv_variance)
        difference = z[index] - mu[index]
        quadratic += difference * difference * workspace.inv_variance[index]
    end
    return -0.5 * (length(z) * log(2π) + workspace.logvar_sum + quadratic)
end

function workspace_log_prior(z)
    return -0.5 * (dot_generic(z, z) + length(z) * log(2π))
end

function workspace_log_likelihood(
    workspace::PPCAFloatWorkspace, model::PPCAModel, z, x,
)
    count_logdensity!()
    mul!(workspace.decoded, model.W, z)
    @. workspace.decoded += model.b
    @. workspace.residual = x - workspace.decoded
    sigma = model.sigma
    return -0.5 * (
        length(x) * log(2π * sigma^2) +
        dot_generic(workspace.residual, workspace.residual) / sigma^2)
end

function workspace_log_density_ratio(
    workspace::PPCAFloatWorkspace, model::PPCAModel, z, x, mu,
)
    return workspace_log_prior(z) +
           workspace_log_likelihood(workspace, model, z, x) -
           workspace_log_q(workspace, z, mu)
end

function workspace_bridge_score!(
    output::Vector{Float64}, workspace::PPCAFloatWorkspace,
    model::PPCAModel, z, x, mu, beta,
)
    count_gradient!()
    mul!(workspace.decoded, model.W, z)
    @. workspace.decoded += model.b
    @. workspace.residual = x - workspace.decoded
    mul!(output, transpose(model.W), workspace.residual)
    sigma2 = model.sigma^2
    @inbounds for index in eachindex(output, z, mu, workspace.inv_variance)
        q_score = -(z[index] - mu[index]) * workspace.inv_variance[index]
        likelihood_score = -z[index] + output[index] / sigma2
        output[index] = (1 - beta) * q_score + beta * likelihood_score
    end
    return output
end

function workspace_bridge_logdensity(
    workspace::PPCAFloatWorkspace, model::PPCAModel, z, x, mu, beta,
)
    # bridge_logdensity calls log_joint, whose likelihood is a second counted
    # density evaluation in the original implementation.
    count_logdensity!()
    q = workspace_log_q(workspace, z, mu)
    joint = workspace_log_prior(z) + workspace_log_likelihood(workspace, model, z, x)
    return (1 - beta) * q + beta * joint
end

function decode_plain(model::PPCAModel, z)
    return model.b .+ matrix_vector(model.W, z)
end

function decode_plain(model::PPCAModel, coordinate, theta, z)
    linear = matrix_vector(model.W, z)
    return [
        model.b[row] + linear[row] +
        (row == coordinate ? theta - model.b[row] : zero(theta))
        for row in axes(model.W, 1)
    ]
end

function decode(model::PPCAModel, z)
    return decode_plain(model, z)
end

function decode(model::ParameterizedPPCA, z)
    return decode_plain(model.base, model.coordinate, model.theta, z)
end

function log_prior(z, d=length(z))
    return -0.5 * (dot_generic(z, z) + d * log(2π))
end

function log_likelihood(model, z, x)
    count_logdensity!()
    residual = x .- decode(model, z)
    sigma = model_sigma(model)
    return -0.5 * (length(x) * log(2π * sigma^2) + dot_generic(residual, residual) / sigma^2)
end

function log_joint(model, z, x)
    return log_prior(z, size(model_weight(model), 2)) + log_likelihood(model, z, x)
end

function log_q(z, mu, logvar)
    inv_variance = exp.(-logvar)
    residual = z .- mu
    return -0.5 * (length(z) * log(2π) + sum(logvar) +
                   dot_generic(residual .* inv_variance, residual))
end

function log_joint_plain(model::PPCAModel, coordinate, theta, z, x)
    residual = x .- decode_plain(model, coordinate, theta, z)
    sigma = model.sigma
    return log_prior(z, size(model.W, 2)) -
           0.5 * (length(x) * log(2π * sigma^2) +
                  dot_generic(residual, residual) / sigma^2)
end

function log_density_ratio_plain(model::PPCAModel, coordinate, theta,
                                 z, x, mu, logvar)
    return log_joint_plain(model, coordinate, theta, z, x) - log_q(z, mu, logvar)
end

function log_density_ratio(model, z, x, mu, logvar)
    return log_joint(model, z, x) - log_q(z, mu, logvar)
end

function log_density_ratio(model::ParameterizedPPCA, z, x, mu, logvar)
    count_logdensity!()
    return log_density_ratio_plain(model.base, model.coordinate, model.theta,
                                   z, x, mu, logvar)
end

function bridge_logdensity(model, z, x, mu, logvar, beta)
    count_logdensity!()
    return (1 - beta) * log_q(z, mu, logvar) + beta * log_joint(model, z, x)
end

function bridge_logdensity(model::ParameterizedPPCA, z, x, mu, logvar, beta)
    count_logdensity!()
    return (1 - beta) * log_q(z, mu, logvar) +
           beta * log_joint_plain(model.base, model.coordinate, model.theta, z, x)
end

function score_prior_likelihood(model, z, x)
    W = model_weight(model)
    residual = x .- decode(model, z)
    sigma2 = model_sigma(model)^2
    return [
        -z[column] + sum(1:size(W, 1)) do row
            W[row, column] * residual[row] / sigma2
        end
        for column in 1:size(W, 2)
    ]
end

function score_q(z, mu, logvar)
    return -(z .- mu) .* exp.(-logvar)
end

function bridge_score(model, z, x, mu, logvar, beta)
    count_gradient!()
    return (1 - beta) .* score_q(z, mu, logvar) .+
           beta .* score_prior_likelihood(model, z, x)
end

function bridge_score_plain(model::PPCAModel, coordinate, theta, z,
                            x, mu, logvar, beta)
    inv_variance = exp.(-logvar)
    residual = x .- decode_plain(model, coordinate, theta, z)
    likelihood_score = -z .+
                       transpose_matrix_vector(model.W, residual) ./ model.sigma^2
    return (1 - beta) .* (-(z .- mu) .* inv_variance) .+
           beta .* likelihood_score
end

function bridge_score(model::ParameterizedPPCA, z, x, mu, logvar, beta)
    count_gradient!()
    return bridge_score_plain(model.base, model.coordinate, model.theta, z,
                              x, mu, logvar, beta)
end

function mala_logratio(model, z, y, x, mu, logvar, beta, gamma)
    gz = bridge_score(model, z, x, mu, logvar, beta)
    return mala_logratio(model, z, y, x, mu, logvar, beta, gamma, gz)
end

function mala_logratio(model, z, y, x, mu, logvar, beta, gamma, gz)
    gy = bridge_score(model, y, x, mu, logvar, beta)
    proposal_ratio = 0.5 * dot_generic(z .- y, gy .+ gz) -
                     gamma / 8 * (dot_generic(gy, gy) - dot_generic(gz, gz))
    return bridge_logdensity(model, y, x, mu, logvar, beta) -
           bridge_logdensity(model, z, x, mu, logvar, beta) + proposal_ratio
end

function workspace_mala_logratio(
    workspace::PPCAFloatWorkspace, model::PPCAModel, z, y, x, mu,
    beta, gamma,
)
    gz = workspace.score
    gy = workspace_bridge_score!(
        workspace.proposal_score, workspace, model, y, x, mu, beta)
    proposal_ratio = 0.5 * dot_generic(z .- y, gy .+ gz) -
                     gamma / 8 * (dot_generic(gy, gy) - dot_generic(gz, gz))
    return workspace_bridge_logdensity(
               workspace, model, y, x, mu, beta) -
           workspace_bridge_logdensity(
               workspace, model, z, x, mu, beta) + proposal_ratio
end

function mala_proposal_step(model, z, x, mu, logvar, beta, gamma, noise)
    gz = bridge_score(model, z, x, mu, logvar, beta)
    mean = z .+ gamma / 2 .* gz
    y = mean .+ sqrt(gamma) .* noise
    gy = bridge_score(model, y, x, mu, logvar, beta)
    proposal_ratio = 0.5 * dot_generic(z .- y, gy .+ gz) -
                     gamma / 8 * (dot_generic(gy, gy) - dot_generic(gz, gz))
    logratio = bridge_logdensity(model, y, x, mu, logvar, beta) -
               bridge_logdensity(model, z, x, mu, logvar, beta) + proposal_ratio
    alpha = min(1.0, exp(min(0.0, logratio)))
    return (; y, mean, logratio, alpha)
end

function log_rejection_probability(logratio)
    # The dual replay knows the primal accept/reject history.  Compare only
    # the primal value here; ForwardDiff duals do not define an ordering.
    dual_value(logratio) >= 0 && return zero(logratio) - Inf
    return log1p(-exp(logratio))
end

struct SampledPath
    weight::Float64
    pathwise_gradient::Float64
    scores::Vector{Float64}
    future_returns::Vector{Float64}
    accepted::BitVector
    alphas::Vector{Float64}
    increments::Vector{Float64}
    states::Union{Nothing,NamedTuple}
    cost::CostStats
end

struct RandomTape
    noises::Vector{Vector{Float64}}
    uniforms::Vector{Float64}
end

function random_tape(rng, K, d)
    return RandomTape([randn(rng, d) for _ in 1:K], rand(rng, K))
end

function dual_seed(value::Float64, derivative::Float64)
    return ForwardDiff.Dual{PPCAForwardTag}(value, derivative)
end

struct PPCAForwardTag end

function dual_value(x)
    return x isa ForwardDiff.Dual ? ForwardDiff.value(x) : x
end

function dual_derivative(x)
    return x isa ForwardDiff.Dual ? ForwardDiff.partials(x)[1] : 0.0
end

"""Replay a fixed accept/reject history with Float64 or ForwardDiff values."""
function replay_fixed(model::PPCAModel, coordinate, theta, x, mu, logvar, z0,
                      tape::RandomTape, accepted, K, gamma)
    parameterized = ParameterizedPPCA(model, coordinate, theta)
    z = [zero(theta) + z0[index] for index in eachindex(z0)]
    increments = Vector{typeof(theta)}(undef, K + 1)
    logprobs = Vector{typeof(theta)}(undef, K)
    betas = collect(range(0.0, 1.0; length=K + 2))

    increments[1] = (betas[2] - betas[1]) *
                    log_density_ratio(parameterized, z, x, mu, logvar)
    for k in 1:K
        beta = betas[k + 1]
        gz = bridge_score(parameterized, z, x, mu, logvar, beta)
        y = z .+ gamma / 2 .* gz .+ sqrt(gamma) .* tape.noises[k]
        gy = bridge_score(parameterized, y, x, mu, logvar, beta)
        proposal_ratio = 0.5 * dot_generic(z .- y, gy .+ gz) -
                         gamma / 8 * (dot_generic(gy, gy) - dot_generic(gz, gz))
        logratio = bridge_logdensity(parameterized, y, x, mu, logvar, beta) -
                   bridge_logdensity(parameterized, z, x, mu, logvar, beta) + proposal_ratio

        if accepted[k]
            logprobs[k] = dual_value(logratio) >= 0 ? zero(logratio) : logratio
            z = y
        else
            logprobs[k] = log_rejection_probability(logratio)
        end
        increments[k + 1] = (betas[k + 2] - betas[k + 1]) *
                            log_density_ratio(parameterized, z, x, mu, logvar)
    end
    return sum(increments), logprobs, increments
end

function sample_path(model::PPCAModel, coordinate, x, mu, logvar, z0,
                     tape::RandomTape, K, gamma; store_states=false)
    cost = CostStats()
    old_cost = ACTIVE_COST[]
    ACTIVE_COST[] = cost
    try
        z = copy(z0)
        accepted = falses(K)
        alphas = zeros(K)
        increments = zeros(K + 1)
        betas = collect(range(0.0, 1.0; length=K + 2))
        states = store_states ? (
            pre=Vector{Vector{Float64}}(undef, K),
            proposals=Vector{Vector{Float64}}(undef, K),
            post=Vector{Vector{Float64}}(undef, K),
            means=Vector{Vector{Float64}}(undef, K),
        ) : nothing
        increments[1] = (betas[2] - betas[1]) *
                        log_density_ratio(model, z, x, mu, logvar)
        for k in 1:K
            beta = betas[k + 1]
            # The sampler never mutates a state vector in place, so retaining
            # these references avoids two 100-dimensional copies per step.
            store_states && (states.pre[k] = z)
            step = mala_proposal_step(model, z, x, mu, logvar, beta, gamma, tape.noises[k])
            if store_states
                states.means[k] = step.mean
                states.proposals[k] = step.y
            end
            alphas[k] = step.alpha
            accepted[k] = tape.uniforms[k] < step.alpha
            z = accepted[k] ? step.y : z
            store_states && (states.post[k] = z)
            cost.transitions += 1
            cost.decisions += 1
            cost.accepted += accepted[k]
            cost.primary_decisions += 1
            cost.primary_accepted += accepted[k]
            logprob = accepted[k] ? log(step.alpha) : log_rejection_probability(step.logratio)
            increments[k + 1] = (betas[k + 2] - betas[k + 1]) *
                                log_density_ratio(model, z, x, mu, logvar)
        end

        theta = dual_seed(model.b[coordinate], 1.0)
        dual_weight, dual_logprobs, _ = replay_fixed(
            model, coordinate, theta, x, mu, logvar, z0, tape, accepted, K, gamma)
        future = [sum(increments[(k + 1):end]) for k in 1:K]
        return SampledPath(
            sum(increments), dual_derivative(dual_weight),
            dual_derivative.(dual_logprobs), future, accepted, alphas,
            increments, states, cost,
        )
    finally
        ACTIVE_COST[] = old_cost
    end
end

@inline standard_normal_logdensity(z) =
    -0.5 * (dot_generic(z, z) + length(z) * log(2π))

"""One maximal-reflection MALA proposal, with the random draw made explicit."""
function hand_maximum_reflection_mala(
    model::PPCAModel, alternative, primal_mean, primal_proposal,
    x, mu, logvar, beta, gamma, rng, workspace::PPCAFloatWorkspace,
)
    workspace_bridge_score!(workspace.score, workspace, model, alternative,
                            x, mu, beta)
    scale = sqrt(gamma)
    @inbounds for index in eachindex(
        workspace.diff, alternative, primal_mean, workspace.score,
    )
        workspace.diff[index] = (
            alternative[index] + gamma / 2 * workspace.score[index] -
            primal_mean[index]) / scale
        workspace.primal_jump[index] = (
            primal_proposal[index] - primal_mean[index]) / scale
    end
    @. workspace.reflected = workspace.primal_jump - workspace.diff
    if log(rand(rng)) + standard_normal_logdensity(workspace.primal_jump) <=
       standard_normal_logdensity(workspace.reflected)
        return (; proposal=primal_proposal, met=true)
    end
    diff_norm = dot_generic(workspace.diff, workspace.diff)
    diff_norm == 0.0 &&
        return (; proposal=primal_proposal, met=true)
    coefficient = 1 - 2 * dot_generic(workspace.diff, workspace.primal_jump) / diff_norm
    @. workspace.reflected = coefficient * workspace.diff
    proposal = similar(primal_proposal)
    @. proposal = primal_proposal + scale * workspace.reflected
    return (; proposal, met=false)
end

"""Use the inversion coupling specified for the MH acceptance coin."""
function hand_coupled_bernoulli(
    primal_accept, primal_alpha, alternative_alpha, rng,
)
    low, high = primal_accept ?
        (1 - primal_alpha, 1.0) : (0.0, 1 - primal_alpha)
    uniform = low + (high - low) * rand(rng)
    return uniform > 1 - alternative_alpha
end

function dmh_acceptance_derivative(path::SampledPath, k)
    return path.accepted[k] ?
        path.alphas[k] * path.scores[k] :
        -(1 - path.alphas[k]) * path.scores[k]
end

function dmh_branch_weight(path::SampledPath, k)
    # W = d(alpha) * (1 - 2 * accepted), as in the paper's Eq. (25).
    return dmh_acceptance_derivative(path, k) *
           (path.accepted[k] ? -1.0 : 1.0)
end

"""Advance one counterfactual branch through primal transition `j`."""
function hand_counterfactual_step(
    model::PPCAModel, alternative, states, path::SampledPath, j,
    x, mu, logvar, gamma, rng, workspace::PPCAFloatWorkspace,
)
    K = length(path.accepted)
    beta = j / (K + 1)
    proposal = hand_maximum_reflection_mala(
        model, alternative, states.means[j], states.proposals[j],
        x, mu, logvar, beta, gamma, rng, workspace)
    alternative_proposal = proposal.proposal
    alternative_logratio = workspace_mala_logratio(
        workspace, model, alternative, alternative_proposal,
        x, mu, beta, gamma)
    alternative_alpha = min(1.0, exp(min(0.0, alternative_logratio)))
    alternative_accept = hand_coupled_bernoulli(
        path.accepted[j], path.alphas[j], alternative_alpha, rng)
    next_state = alternative_accept ? alternative_proposal : alternative
    met = proposal.met && same_state(next_state, states.post[j])
    increment = met ? path.increments[j + 1] :
        (1 / (K + 1)) * workspace_log_density_ratio(
            workspace, model, next_state, x, mu)

    cost = ACTIVE_COST[]
    if !isnothing(cost)
        cost.transitions += 1
        cost.decisions += 1
        cost.accepted += alternative_accept
        cost.alternative_decisions += 1
        cost.alternative_accepted += alternative_accept
        cost.meeting_opportunities += 1
        cost.meetings += met
    end
    return (; state=next_state, increment, met)
end

"""Start the branch obtained by taking the opposite decision at transition k.

The opposite post-decision state contributes immediately to the AIS objective.
Keeping this as a separate operation is important: it is a branch birth, not a
new MH transition.  The old implementation accidentally skipped this reward
and advanced the newborn branch through transition k+1 before using it.
"""
function hand_counterfactual_split(
    model::PPCAModel, x, mu, logvar, path::SampledPath, states, k,
    workspace::PPCAFloatWorkspace,
)
    alternative = path.accepted[k] ? states.pre[k] : states.proposals[k]
    cost = ACTIVE_COST[]
    if !isnothing(cost)
        cost.spawned_alternatives += 1
    end
    increment = (1 / (length(path.accepted) + 1)) *
                workspace_log_density_ratio(workspace, model, alternative, x, mu)
    return (; state=alternative, increment,
            met=same_state(alternative, states.post[k]))
end

function hand_counterfactual_future(
    model::PPCAModel, x, mu, logvar, path::SampledPath, states, k,
    gamma, rng, workspace::PPCAFloatWorkspace,
)
    K = length(path.accepted)
    split = hand_counterfactual_split(
        model, x, mu, logvar, path, states, k, workspace)
    alternative = split.state
    future = split.increment
    met = split.met
    met && return path.future_returns[k], true
    for j in (k + 1):K
        step = hand_counterfactual_step(
            model, alternative, states, path, j, x, mu, logvar, gamma, rng,
            workspace)
        alternative = step.state
        future += step.increment
        if step.met
            met = true
            j < K && (future += path.future_returns[j + 1])
            break
        end
    end
    return future, met
end

"""Explicit DMH-all audit using only Float64 and ForwardDiff."""
function hand_dmh_all_path(
    model::PPCAModel, coordinate, x, mu, logvar, z0, tape::RandomTape,
    K, gamma, rng,
)
    cost = CostStats()
    old_cost = ACTIVE_COST[]
    ACTIVE_COST[] = cost
    try
        path = sample_path(model, coordinate, x, mu, logvar, z0, tape, K, gamma;
                           store_states=true)
        states = path.states
        states === nothing && error("explicit DMH path did not retain primal states")
        workspace = PPCAFloatWorkspace(model, logvar)
        add_cost!(cost, path.cost)
        ar = 0.0
        # Every accept/reject decision contributes to the finite AIS
        # objective, including the final transition.  Its reward is the
        # final post-transition increment, so the roots run through K.
        for k in 1:K
            alternative_future, _ = hand_counterfactual_future(
                model, x, mu, logvar, path, states, k, gamma, rng, workspace)
            branch_difference = path.accepted[k] ?
                path.future_returns[k] - alternative_future :
                alternative_future - path.future_returns[k]
            ar += dmh_acceptance_derivative(path, k) * branch_difference
        end
        return (; estimate=path.pathwise_gradient + ar,
                smooth=path.pathwise_gradient, ar, cost,
                meetings=cost.meetings,
                meeting_opportunities=cost.meeting_opportunities)
    finally
        ACTIVE_COST[] = old_cost
    end
end

"""Explicit DMH-one using the paper's weighted sequential pruning rule."""
function hand_dmh_one_path(
    model::PPCAModel, coordinate, x, mu, logvar, z0, tape::RandomTape,
    K, gamma, coupling_rng, pruning_rng,
)
    cost = CostStats()
    old_cost = ACTIVE_COST[]
    ACTIVE_COST[] = cost
    try
        path = sample_path(model, coordinate, x, mu, logvar, z0, tape, K, gamma;
                           store_states=true)
        states = path.states
        states === nothing && error("explicit DMH path did not retain primal states")
        workspace = PPCAFloatWorkspace(model, logvar)
        add_cost!(cost, path.cost)

        ar = 0.0
        active_state = nothing
        active_increment = 0.0
        active_root = 0
        active_met = false
        cumulative_weight = 0.0

        # At reward time j, the new branch is split at transition j and its
        # opposite post-decision state is already the state contributing this
        # time point.  The incumbent, if still alive, is advanced through the
        # same transition and then competes with that newborn branch.
        for j in 1:K
            new_root = j
            new_step = hand_counterfactual_split(
                model, x, mu, logvar, path, states, j, workspace)
            new_weight = dmh_branch_weight(path, new_root)
            new_mass = abs(new_weight)

            if active_root == 0 || active_met
                active_root = new_root
                active_state = new_step.state
                active_increment = new_step.increment
                active_met = new_step.met
                cumulative_weight = new_mass
            else
                old_step = hand_counterfactual_step(
                    model, active_state, states, path, j,
                    x, mu, logvar, gamma, coupling_rng, workspace)
                if old_step.met
                    # A recoupling resets the pruning mass; the newborn
                    # branch is the only representative after this time.
                    active_root = new_root
                    active_state = new_step.state
                    active_increment = new_step.increment
                    active_met = new_step.met
                    cumulative_weight = new_mass
                else
                    combined_mass = cumulative_weight + new_mass
                    keep_old = combined_mass > 0.0 &&
                                rand(pruning_rng) < cumulative_weight / combined_mass
                    if keep_old
                        active_state = old_step.state
                        active_increment = old_step.increment
                        active_met = old_step.met
                    else
                        active_root = new_root
                        active_state = new_step.state
                        active_increment = new_step.increment
                        active_met = new_step.met
                    end
                    cumulative_weight = combined_mass
                end
            end

            primal_increment = path.increments[j + 1]
            ar += cumulative_weight * sign(dmh_branch_weight(path, active_root)) *
                  (active_increment - primal_increment)
        end
        return (; estimate=path.pathwise_gradient + ar,
                smooth=path.pathwise_gradient, ar, cost,
                meetings=cost.meetings,
                meeting_opportunities=cost.meeting_opportunities)
    finally
        ACTIVE_COST[] = old_cost
    end
end

function hand_dmh_all_one(artifact, config::PPCAConfig, seed; observation=1)
    rng = Xoshiro(seed)
    tape = random_tape(rng, config.K, size(artifact.W, 2))
    model = PPCAModel(artifact.W, artifact.b, artifact.sigma)
    x = vec(artifact.x[observation, :])
    mu = vec(artifact.q_mu[observation, :])
    logvar = vec(artifact.q_logvar[observation, :])
    z0 = collect(view(artifact.z_n1, observation, :))
    return hand_dmh_all_path(
        model, config.coordinate_zero_based + 1, x, mu, logvar, z0, tape,
        config.K, config.gamma, Xoshiro(seed + 1))
end

function hand_dmh_one_one(artifact, config::PPCAConfig, seed; observation=1)
    rng = Xoshiro(seed)
    tape = random_tape(rng, config.K, size(artifact.W, 2))
    model = PPCAModel(artifact.W, artifact.b, artifact.sigma)
    x = vec(artifact.x[observation, :])
    mu = vec(artifact.q_mu[observation, :])
    logvar = vec(artifact.q_logvar[observation, :])
    z0 = collect(view(artifact.z_n1, observation, :))
    return hand_dmh_one_path(
        model, config.coordinate_zero_based + 1, x, mu, logvar, z0, tape,
        config.K, config.gamma, Xoshiro(seed + 1), Xoshiro(seed + 2))
end

function particle_row(artifact, particle, observation, n_particles)
    batch_size = size(artifact.x, 1)
    row = (particle - 1) * batch_size + observation
    z = n_particles == 1 ?
        view(artifact.z_n1, observation, :) : view(artifact.z_n10, row, :)
    return z
end

function sample_batch(artifact, K, n_particles, rng, coordinate;
                      gamma=JULIA_MALA_GAMMA)
    batch_size = size(artifact.x, 1)
    paths = Matrix{SampledPath}(undef, n_particles, batch_size)
    total_cost = CostStats()
    for particle in 1:n_particles, observation in 1:batch_size
        tape = random_tape(rng, K, size(artifact.W, 2))
        path = sample_path(
            PPCAModel(artifact.W, artifact.b, artifact.sigma), coordinate,
            vec(artifact.x[observation, :]), vec(artifact.q_mu[observation, :]),
            vec(artifact.q_logvar[observation, :]),
            collect(particle_row(artifact, particle, observation, n_particles)), tape, K,
            gamma,
        )
        paths[particle, observation] = path
        total_cost += path.cost
    end
    return paths, total_cost
end

function estimate_baselines(artifact, K, repetitions, seed, coordinate;
                            gamma=JULIA_MALA_GAMMA)
    rng = Xoshiro(seed)
    values = zeros(repetitions, K)
    total_cost = CostStats()
    elapsed = @elapsed for repetition in 1:repetitions
        paths, cost = sample_batch(artifact, K, 1, rng, coordinate; gamma)
        total_cost += cost
        values[repetition, :] = vec(mean(hcat([path.future_returns for path in paths]...), dims=2))
    end
    return (
        baselines=vec(mean(values, dims=1)),
        cost=total_cost,
        elapsed,
    )
end

function reinforce_estimates(paths, baselines)
    n_particles, batch_size = size(paths)
    ordinary = zeros(batch_size)
    causal = zeros(batch_size)
    baseline = zeros(batch_size)
    for observation in 1:batch_size
        for particle in 1:n_particles
            path = paths[particle, observation]
            score_sum = sum(path.scores)
            ordinary[observation] += path.pathwise_gradient + path.weight * score_sum
            causal[observation] += path.pathwise_gradient +
                                   dot_generic(path.scores, path.future_returns)
            baseline[observation] += path.pathwise_gradient +
                                     dot_generic(path.scores, path.future_returns .- baselines)
        end
    end
    # Return one estimate per observation.  `run_reinforce` averages this
    # vector over the batch; dividing by batch_size here would normalize the
    # same batch a second time.
    scale = 1 / n_particles
    return scale .* ordinary, scale .* causal, scale .* baseline
end

function loo_estimate(paths)
    n_particles, batch_size = size(paths)
    @assert n_particles > 1
    estimate = 0.0
    for observation in 1:batch_size
        weights = [paths[particle, observation].weight for particle in 1:n_particles]
        for particle in 1:n_particles
            path = paths[particle, observation]
            leave_one_out = (sum(weights) - path.weight) / (n_particles - 1)
            estimate += path.pathwise_gradient +
                        (path.weight - leave_one_out) * sum(path.scores)
        end
    end
    return estimate / (n_particles * batch_size)
end

struct PPCAAnnealedScore{W,B,S,C,T,X,U,L,V}
    W::W
    base_b::B
    sigma::S
    coordinate::C
    theta::T
    x::X
    mu::U
    logvar::L
    beta::V
end

Functors.@functor PPCAAnnealedScore
Functors.@functor PPCAModel

function PPCAAnnealedScore(model::ParameterizedPPCA, x, mu, logvar, beta)
    return PPCAAnnealedScore(
        model.base.W, model.base.b, model.base.sigma, model.coordinate,
        model.theta, x, mu, logvar, beta,
    )
end

function PPCAAnnealedScore(model::PPCAModel, x, mu, logvar, beta)
    # Coordinate zero means that the callable uses the unparameterized bias.
    return PPCAAnnealedScore(
        model.W, model.b, model.sigma, 0, 0.0, x, mu, logvar, beta,
    )
end

function (score::PPCAAnnealedScore)(z)
    base_model = PPCAModel(score.W, score.base_b, score.sigma)
    model = ParameterizedPPCA(base_model, score.coordinate, score.theta)
    return bridge_score(model, z, score.x, score.mu, score.logvar, score.beta)
end

function same_state(left, right)
    return length(left) == length(right) && all(isequal.(left, right))
end

# ---------------------------------------------------------------------------
# Experiment bookkeeping and analytical reference

struct EstimatorResult
    samples::Vector{Float64}
    costs::Vector{CostStats}
    insertions::Vector{Int}
    elapsed::Float64
    acceptance::Float64
    pilot_cost::CostStats
    pilot_elapsed::Float64
end

function EstimatorResult(
    samples::Vector{Float64}, costs::Vector{CostStats}, insertions::Vector{Int},
    elapsed::Float64, acceptance::Float64;
    pilot_cost=CostStats(), pilot_elapsed=0.0,
)
    return EstimatorResult(
        samples, costs, insertions, elapsed, acceptance,
        pilot_cost, pilot_elapsed,
    )
end

function mean_cost(costs)
    isempty(costs) && return (
        transitions=0.0, logdensity_evals=0.0, gradient_evals=0.0,
        spawned_alternatives=0.0, meeting_opportunities=0.0, meetings=0.0,
        accepted=0.0, decisions=0.0,
        primary_accepted=0.0, primary_decisions=0.0,
        alternative_accepted=0.0, alternative_decisions=0.0,
    )
    total = reduce(+, costs)
    n = length(costs)
    return (
        transitions=total.transitions / n,
        logdensity_evals=total.logdensity_evals / n,
        gradient_evals=total.gradient_evals / n,
        spawned_alternatives=total.spawned_alternatives / n,
        meeting_opportunities=total.meeting_opportunities / n,
        meetings=total.meetings / n,
        accepted=total.accepted / n,
        decisions=total.decisions / n,
        primary_accepted=total.primary_accepted / n,
        primary_decisions=total.primary_decisions / n,
        alternative_accepted=total.alternative_accepted / n,
        alternative_decisions=total.alternative_decisions / n,
    )
end

function run_reinforce(artifact, config, repetitions, pilot_repetitions, seed)
    coordinate = config.coordinate_zero_based + 1
    pilot = estimate_baselines(
        artifact, config.K, pilot_repetitions, seed + 1, coordinate;
        gamma=config.gamma)
    baselines = pilot.baselines
    rng = Xoshiro(seed + 2)
    ordinary = zeros(repetitions)
    causal = zeros(repetitions)
    baseline = zeros(repetitions)
    costs = CostStats[]
    acceptance = 0.0
    elapsed = @elapsed for repetition in 1:repetitions
        paths, cost = sample_batch(artifact, config.K, 1, rng, coordinate;
                                   gamma=config.gamma)
        ordinary_vec, causal_vec, baseline_vec = reinforce_estimates(paths, baselines)
        ordinary[repetition] = mean(ordinary_vec)
        causal[repetition] = mean(causal_vec)
        baseline[repetition] = mean(baseline_vec)
        push!(costs, cost)
        acceptance += cost.accepted / cost.decisions
    end
    mean_acceptance = acceptance / repetitions
    return (;
        ordinary=EstimatorResult(ordinary, costs, zeros(Int, repetitions), elapsed, mean_acceptance),
        causal=EstimatorResult(causal, costs, zeros(Int, repetitions), elapsed, mean_acceptance),
        baseline=EstimatorResult(
            baseline, costs, zeros(Int, repetitions),
            elapsed + pilot.elapsed, mean_acceptance;
            pilot_cost=pilot.cost, pilot_elapsed=pilot.elapsed),
        baselines,
        pilot,
    )
end

function run_loo(artifact, config, repetitions, seed)
    rng = Xoshiro(seed)
    samples = zeros(repetitions)
    costs = CostStats[]
    acceptance = 0.0
    elapsed = @elapsed for repetition in 1:repetitions
        paths, cost = sample_batch(artifact, config.K, 10, rng,
                                   config.coordinate_zero_based + 1;
                                   gamma=config.gamma)
        samples[repetition] = loo_estimate(paths)
        push!(costs, cost)
        acceptance += cost.accepted / cost.decisions
    end
    return EstimatorResult(samples, costs, zeros(Int, repetitions), elapsed,
                           acceptance / repetitions)
end

function dmh_batch(artifact, config::PPCAConfig, mode::Symbol, seed)
    mode in (:all, :one) || error("unknown explicit DMH mode: $(mode)")
    tape_rng = Xoshiro(seed)
    coupling_rng = Xoshiro(seed + 1)
    pruning_rng = Xoshiro(seed + 2)
    model = PPCAModel(artifact.W, artifact.b, artifact.sigma)
    coordinate = config.coordinate_zero_based + 1
    batch_size = size(artifact.x, 1)
    total_estimate = 0.0
    total_smooth = 0.0
    total_ar = 0.0
    total_cost = CostStats()
    elapsed = @elapsed for observation in 1:batch_size
        tape = random_tape(tape_rng, config.K, size(artifact.W, 2))
        x = vec(artifact.x[observation, :])
        mu = vec(artifact.q_mu[observation, :])
        logvar = vec(artifact.q_logvar[observation, :])
        z0 = collect(view(artifact.z_n1, observation, :))
        result = if mode == :all
            hand_dmh_all_path(
                model, coordinate, x, mu, logvar, z0, tape,
                config.K, config.gamma, coupling_rng)
        else
            hand_dmh_one_path(
                model, coordinate, x, mu, logvar, z0, tape,
                config.K, config.gamma, coupling_rng, pruning_rng)
        end
        total_estimate += result.estimate
        total_smooth += result.smooth
        total_ar += result.ar
        total_cost += result.cost
    end
    return (; estimate=total_estimate / batch_size,
            smooth=total_smooth / batch_size,
            ar=total_ar / batch_size,
            cost=total_cost,
            insertions=0,
            elapsed)
end

function run_dmh(artifact, config, mode::Symbol, repetitions, seed)
    samples = zeros(repetitions)
    ar_samples = zeros(repetitions)
    smooth_samples = zeros(repetitions)
    costs = CostStats[]
    insertions = zeros(Int, repetitions)
    elapsed = @elapsed for repetition in 1:repetitions
        result = dmh_batch(artifact, config, mode, seed + repetition)
        samples[repetition] = result.estimate
        ar_samples[repetition] = result.ar
        smooth_samples[repetition] = result.smooth
        push!(costs, result.cost)
        insertions[repetition] = result.insertions
    end
    acceptance = mean(cost.accepted / max(cost.decisions, 1) for cost in costs)
    return EstimatorResult(samples, costs, insertions, elapsed, acceptance),
           (; ar_samples, smooth_samples)
end

function log_marginal(model::PPCAModel, x)
    covariance = Symmetric(model.W * model.W' + model.sigma^2 * I)
    factor = cholesky(covariance)
    residual = x .- model.b
    return -0.5 * (length(x) * log(2π) + logdet(factor) +
                    dot(residual, factor \ residual))
end

function analytical_gradient(artifact::PPCAArtifact)
    covariance = Symmetric(artifact.W * artifact.W' + artifact.sigma^2 * I)
    factor = cholesky(covariance)
    gradient = zeros(size(artifact.W, 1))
    for observation in axes(artifact.x, 1)
        gradient += factor \ (vec(artifact.x[observation, :]) .- artifact.b)
    end
    return gradient / size(artifact.x, 1)
end

function summary_row(K, method, result::EstimatorResult, reference)
    samples = result.samples
    cost = mean_cost(result.costs)
    n = max(length(samples), 1)
    # The pilot is a fixed cost of producing the RF time baseline.  Charge it
    # only to that estimator and amortize it over its reported replicates.
    pilot = result.pilot_cost
    cost = (
        transitions=cost.transitions + pilot.transitions / n,
        logdensity_evals=cost.logdensity_evals + pilot.logdensity_evals / n,
        gradient_evals=cost.gradient_evals + pilot.gradient_evals / n,
        spawned_alternatives=cost.spawned_alternatives + pilot.spawned_alternatives / n,
        meeting_opportunities=cost.meeting_opportunities + pilot.meeting_opportunities / n,
        meetings=cost.meetings + pilot.meetings / n,
        accepted=cost.accepted + pilot.accepted / n,
        decisions=cost.decisions + pilot.decisions / n,
        primary_accepted=cost.primary_accepted + pilot.primary_accepted / n,
        primary_decisions=cost.primary_decisions + pilot.primary_decisions / n,
        alternative_accepted=cost.alternative_accepted + pilot.alternative_accepted / n,
        alternative_decisions=cost.alternative_decisions + pilot.alternative_decisions / n,
    )
    variance = length(samples) > 1 ? var(samples) : 0.0
    mean_value = mean(samples)
    mse = mean((samples .- reference) .^ 2)
    # One MALA transition entails several target evaluations.  Keep the
    # legacy CSV field name below, but define it explicitly as the sum of
    # log-density and state-gradient evaluations.
    target_evals = cost.logdensity_evals + cost.gradient_evals
    meeting_frequency = cost.meeting_opportunities == 0 ? 0.0 :
                        cost.meetings / cost.meeting_opportunities
    return (
        K=K,
        method=method,
        mean=mean_value,
        bias=mean_value - reference,
        mse_vs_reference=mse,
        variance=variance,
        stderr=length(samples) > 1 ? sqrt(variance / length(samples)) : 0.0,
        transitions=cost.transitions,
        logdensity_evals=cost.logdensity_evals,
        gradient_evals=cost.gradient_evals,
        expensive_evals=target_evals,
        variance_x_transitions=variance * cost.transitions,
        variance_x_expensive_evals=variance * target_evals,
        mse_x_transitions=mse * cost.transitions,
        mse_x_expensive_evals=mse * target_evals,
        wall_seconds=result.elapsed,
        wall_seconds_per_estimate=result.elapsed / max(length(samples), 1),
        variance_x_wall=variance * result.elapsed / max(length(samples), 1),
        acceptance=result.acceptance,
        primary_acceptance=cost.primary_decisions == 0 ? 0.0 :
                           cost.primary_accepted / cost.primary_decisions,
        alternative_acceptance=cost.alternative_decisions == 0 ? 0.0 :
                               cost.alternative_accepted / cost.alternative_decisions,
        spawned_alternatives=cost.spawned_alternatives,
        meeting_frequency=meeting_frequency,
        insertions=mean(result.insertions),
        pilot_wall_seconds=result.pilot_elapsed,
        pilot_expensive_evals_per_estimate=(
            result.pilot_cost.logdensity_evals + result.pilot_cost.gradient_evals) / n,
    )
end

function reference_row(K, reference)
    return (
        K=K,
        method="Analytical_PPCA_reference",
        mean=reference,
        bias=0.0,
        mse_vs_reference=0.0,
        variance=0.0,
        stderr=0.0,
        transitions=0.0,
        logdensity_evals=0.0,
        gradient_evals=0.0,
        expensive_evals=0.0,
        variance_x_transitions=0.0,
        variance_x_expensive_evals=0.0,
        mse_x_transitions=0.0,
        mse_x_expensive_evals=0.0,
        wall_seconds=0.0,
        wall_seconds_per_estimate=0.0,
        variance_x_wall=0.0,
        acceptance=0.0,
        primary_acceptance=0.0,
        alternative_acceptance=0.0,
        spawned_alternatives=0.0,
        meeting_frequency=0.0,
        insertions=0.0,
        pilot_wall_seconds=0.0,
        pilot_expensive_evals_per_estimate=0.0,
    )
end

function write_rows(path, rows)
    isempty(rows) && return
    fields = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(fields, ','))
        for row in rows
            values = map(fields) do field
                text = string(getproperty(row, field))
                occursin(',', text) ? "\"$(replace(text, '"' => "\"\""))\"" : text
            end
            println(io, join(values, ','))
        end
    end
end

const SAMPLE_METHOD_SPECS = [
    (key=:ordinary, method="REINFORCE_ordinary"),
    (key=:causal, method="REINFORCE_causal_return_to_go"),
    (key=:baseline, method="REINFORCE_causal_time_baseline"),
    (key=:loo, method="Thin_leave_one_out_N10"),
    (key=:dmh_all, method="DMH_all_MaximumReflection"),
    (key=:dmh_one, method="DMH_one_MaximumReflection_Pruned"),
]

function write_sample_rows(path, case_data)
    rows = NamedTuple[]
    for K in sort(collect(keys(case_data)))
        for spec in SAMPLE_METHOD_SPECS
            haskey(case_data[K], spec.key) || continue
            for (repetition, estimate) in enumerate(case_data[K][spec.key])
                push!(rows, (
                    K=K,
                    method=spec.method,
                    repetition=repetition,
                    estimate=estimate,
                    gradient=-estimate,
                ))
            end
        end
    end
    write_rows(path, rows)
end

function finite_difference_coordinate(artifact::PPCAArtifact, coordinate; h=1e-5)
    model = PPCAModel(artifact.W, artifact.b, artifact.sigma)
    plus_bias = copy(model.b)
    minus_bias = copy(model.b)
    plus_bias[coordinate] += h
    minus_bias[coordinate] -= h
    plus_model = PPCAModel(model.W, plus_bias, model.sigma)
    minus_model = PPCAModel(model.W, minus_bias, model.sigma)
    plus = sum(log_marginal(plus_model, vec(row)) for row in eachrow(artifact.x))
    minus = sum(log_marginal(minus_model, vec(row)) for row in eachrow(artifact.x))
    return (plus - minus) / (2h * size(artifact.x, 1))
end

function validate_configuration(artifact::PPCAArtifact, config::PPCAConfig)
    coordinate = config.coordinate_zero_based + 1
    @assert 1 <= coordinate <= length(artifact.b)
    @assert config.K > 0
    @assert isapprox(config.gamma, 2 * SOURCE_EPSILON; atol=0, rtol=0)
    if haskey(artifact.metadata, "source_epsilon")
        @assert isapprox(Float64(artifact.metadata["source_epsilon"]), SOURCE_EPSILON;
                         atol=1e-12)
    end
    if haskey(artifact.metadata, "julia_mala_gamma")
        @assert isapprox(Float64(artifact.metadata["julia_mala_gamma"]), config.gamma;
                         atol=1e-12)
    end
    betas = collect(range(0.0, 1.0; length=config.K + 2))
    @assert first(betas) == 0.0 && last(betas) == 1.0
    @assert length(betas) == config.K + 2
    @assert all(diff(betas) .> 0)
    return true
end

function fixed_tape_validation(artifact::PPCAArtifact, config::PPCAConfig, seed)
    coordinate = config.coordinate_zero_based + 1
    model = PPCAModel(artifact.W, artifact.b, artifact.sigma)
    rng = Xoshiro(seed)
    observation = 1
    z0 = collect(view(artifact.z_n1, observation, :))
    tape = random_tape(rng, config.K, size(artifact.W, 2))
    x = vec(artifact.x[observation, :])
    mu = vec(artifact.q_mu[observation, :])
    logvar = vec(artifact.q_logvar[observation, :])
    path = sample_path(model, coordinate, x, mu, logvar, z0, tape,
                       config.K, config.gamma)
    replay_weight, _, _ = replay_fixed(
        model, coordinate, model.b[coordinate], x, mu, logvar, z0, tape,
        path.accepted, config.K, config.gamma)
    @assert isapprox(path.weight, replay_weight; atol=1e-10, rtol=1e-10)

    h = 1e-5
    plus, _, _ = replay_fixed(
        model, coordinate, model.b[coordinate] + h, x, mu, logvar, z0,
        tape, path.accepted, config.K, config.gamma)
    minus, _, _ = replay_fixed(
        model, coordinate, model.b[coordinate] - h, x, mu, logvar, z0,
        tape, path.accepted, config.K, config.gamma)
    finite_difference = (plus - minus) / (2h)
    @assert isapprox(path.pathwise_gradient, finite_difference;
                     atol=2e-5, rtol=2e-5)
    return (; direct_weight=path.weight, replay_weight,
            pathwise_gradient=path.pathwise_gradient, finite_difference)
end

function validate_mala_parameterization(artifact::PPCAArtifact, config::PPCAConfig)
    model = PPCAModel(artifact.W, artifact.b, artifact.sigma)
    observation = 1
    z = collect(view(artifact.z_n1, observation, :))
    x = vec(artifact.x[observation, :])
    mu = vec(artifact.q_mu[observation, :])
    logvar = vec(artifact.q_logvar[observation, :])
    beta = 0.5
    score = PPCAAnnealedScore(model, x, mu, logvar, beta)
    proposal = MALAProposal{Vector{Float64}}(
        MvNormal(zeros(length(z)), I), config.gamma, score)
    expected_mean = z .+ config.gamma / 2 .* bridge_score(
        model, z, x, mu, logvar, beta)
    actual_mean = DifferentiableMH._mean(proposal, z)
    covariance = proposal.inner_proposal.step_distribution.Σ
    @assert isapprox(actual_mean, expected_mean; atol=1e-10, rtol=1e-10)
    @assert isapprox(Matrix(covariance), config.gamma * I;
                     atol=1e-10, rtol=1e-10)
    return (; gamma=config.gamma, source_epsilon=SOURCE_EPSILON,
            mean_error=maximum(abs.(actual_mean .- expected_mean)))
end

function self_test()
    artifact = synthetic_artifact(seed=DEFAULT_SEED + 1, batch_size=2, p=5, d=3)
    validate_artifact(artifact)
    config = PPCAConfig(2; gamma=JULIA_MALA_GAMMA, coordinate_zero_based=1)
    validate_configuration(artifact, config)

    coordinate = config.coordinate_zero_based + 1
    reference = analytical_gradient(artifact)[coordinate]
    finite_difference = finite_difference_coordinate(artifact, coordinate)
    @assert isapprox(reference, finite_difference; atol=2e-5, rtol=2e-5)
    model = PPCAModel(artifact.W, artifact.b, artifact.sigma)
    x = vec(artifact.x[1, :])
    mu = vec(artifact.q_mu[1, :])
    logvar = vec(artifact.q_logvar[1, :])
    z = collect(view(artifact.z_n1, 1, :))
    y = z .+ 0.1
    workspace = PPCAFloatWorkspace(model, logvar)
    beta = 0.37
    workspace_bridge_score!(workspace.score, workspace, model, z, x, mu, beta)
    @assert isapprox(workspace.score, bridge_score(model, z, x, mu, logvar, beta);
                     atol=1e-10, rtol=1e-10)
    @assert isapprox(
        workspace_log_density_ratio(workspace, model, z, x, mu),
        log_density_ratio(model, z, x, mu, logvar);
        atol=1e-10, rtol=1e-10)
    @assert isapprox(
        workspace_bridge_logdensity(workspace, model, z, x, mu, beta),
        bridge_logdensity(model, z, x, mu, logvar, beta);
        atol=1e-10, rtol=1e-10)
    @assert isapprox(
        workspace_mala_logratio(workspace, model, z, y, x, mu, beta, config.gamma),
        mala_logratio(model, z, y, x, mu, logvar, beta, config.gamma);
        atol=1e-10, rtol=1e-10)
    fixed_tape = fixed_tape_validation(artifact, config, DEFAULT_SEED + 2)
    mala = validate_mala_parameterization(artifact, config)

    # A one-transition DMH path has no continuation: its A/R term must still
    # compare the opposite post-decision AIS increment with the primal final
    # increment.  This catches both a missing final root and the historical
    # one-transition shift in the counterfactual replay.
    one_config = PPCAConfig(1; gamma=JULIA_MALA_GAMMA, coordinate_zero_based=1)
    one_model = PPCAModel(artifact.W, artifact.b, artifact.sigma)
    one_x = vec(artifact.x[1, :])
    one_mu = vec(artifact.q_mu[1, :])
    one_logvar = vec(artifact.q_logvar[1, :])
    one_z = collect(view(artifact.z_n1, 1, :))
    one_tape = random_tape(Xoshiro(DEFAULT_SEED + 9), 1, size(artifact.W, 2))
    one_path = sample_path(
        one_model, one_config.coordinate_zero_based + 1, one_x, one_mu,
        one_logvar, one_z, one_tape, 1, one_config.gamma; store_states=true)
    one_states = one_path.states
    one_states === nothing && error("one-step self-test did not retain states")
    one_workspace = PPCAFloatWorkspace(one_model, one_logvar)
    one_split = hand_counterfactual_split(
        one_model, one_x, one_mu, one_logvar, one_path, one_states, 1,
        one_workspace)
    one_difference = one_path.accepted[1] ?
        one_path.future_returns[1] - one_split.increment :
        one_split.increment - one_path.future_returns[1]
    one_expected = one_path.pathwise_gradient +
                   dmh_acceptance_derivative(one_path, 1) * one_difference
    one_actual = hand_dmh_all_path(
        one_model, one_config.coordinate_zero_based + 1, one_x, one_mu,
        one_logvar, one_z, one_tape, 1, one_config.gamma, Xoshiro(DEFAULT_SEED + 10))
    @assert isapprox(one_actual.estimate, one_expected; atol=1e-10, rtol=1e-10)

    # `reinforce_estimates` returns one value per observation.  Its batch
    # average must agree with a direct path average, guarding against a
    # second division by batch size in the caller.
    normalization_paths, _ = sample_batch(
        artifact, config.K, 2, Xoshiro(DEFAULT_SEED + 11), coordinate;
        gamma=config.gamma)
    normalization_manual = mean(
        path.pathwise_gradient + path.weight * sum(path.scores)
        for path in normalization_paths)
    normalization_estimate = mean(
        first(reinforce_estimates(normalization_paths, zeros(config.K))))
    @assert isapprox(normalization_estimate, normalization_manual;
                     atol=1e-10, rtol=1e-10)

    rf = run_reinforce(artifact, config, 2, 2, DEFAULT_SEED + 3)
    loo = run_loo(artifact, config, 2, DEFAULT_SEED + 4)
    dmh_all, _ = run_dmh(artifact, config, :all, 1, DEFAULT_SEED + 5)
    dmh_one, _ = run_dmh(artifact, config, :one, 1, DEFAULT_SEED + 6)
    pruning_config = PPCAConfig(4; gamma=JULIA_MALA_GAMMA, coordinate_zero_based=1)
    pruning_run, _ = run_dmh(artifact, pruning_config, :one, 1, DEFAULT_SEED + 8)
    hand = hand_dmh_all_one(artifact, config, DEFAULT_SEED + 7)
    @assert all(isfinite, rf.ordinary.samples)
    @assert all(isfinite, rf.causal.samples)
    @assert all(isfinite, rf.baseline.samples)
    @assert all(isfinite, loo.samples)
    @assert all(isfinite, dmh_all.samples)
    @assert all(isfinite, dmh_one.samples)
    @assert all(isfinite, pruning_run.samples)
    @assert rf.ordinary.pilot_cost.logdensity_evals == 0
    @assert rf.ordinary.pilot_cost.gradient_evals == 0
    @assert rf.baseline.pilot_cost.logdensity_evals +
            rf.baseline.pilot_cost.gradient_evals > 0
    @assert rf.baseline.elapsed >= rf.ordinary.elapsed
    @assert isfinite(hand.estimate)
    @assert isfinite(hand.smooth)
    @assert isfinite(hand.ar)
    @assert 0 <= hand.meetings <= hand.meeting_opportunities
    @assert dmh_all.costs[1].transitions >= size(artifact.x, 1) * config.K
    @assert dmh_one.costs[1].transitions >= size(artifact.x, 1) * config.K
    @assert dmh_all.costs[1].decisions == dmh_all.costs[1].transitions
    @assert dmh_one.costs[1].decisions == dmh_one.costs[1].transitions
    @assert dmh_all.costs[1].spawned_alternatives > 0
    @assert dmh_one.costs[1].spawned_alternatives > 0
    @assert dmh_all.costs[1].meeting_opportunities <=
            dmh_all.costs[1].spawned_alternatives * config.K
    @assert dmh_one.costs[1].meeting_opportunities <=
            dmh_one.costs[1].spawned_alternatives

    println("PPCA self-test passed")
    println("  analytical/finite-difference = $(reference) / $(finite_difference)")
    println("  fixed-tape direct/replay      = $(fixed_tape.direct_weight) / $(fixed_tape.replay_weight)")
    println("  MALA gamma                    = $(mala.gamma)")
    return true
end


function run_case(artifact, K, repetitions, pilot_repetitions, dmh_repetitions, seed)
    config = PPCAConfig(K)
    validate_configuration(artifact, config)
    coordinate = config.coordinate_zero_based + 1
    reference = analytical_gradient(artifact)[coordinate]
    println("  K=$(K): estimating RF ordinary/causal/baseline"); flush(stdout)
    rf = run_reinforce(artifact, config, repetitions, pilot_repetitions, seed + 1)
    println("  K=$(K): RF complete (main=$(round(rf.ordinary.elapsed; digits=2)) s, " *
            "baseline total=$(round(rf.baseline.elapsed; digits=2)) s)"); flush(stdout)
    println("  K=$(K): estimating leave-one-out"); flush(stdout)
    loo = run_loo(artifact, config, repetitions, seed + 2)
    println("  K=$(K): LOO complete ($(round(loo.elapsed; digits=2)) s)"); flush(stdout)
    println("  K=$(K): estimating DMH-all"); flush(stdout)
    dmh_all, dmh_all_parts = run_dmh(
        artifact, config, :all, dmh_repetitions, seed + 3)
    println("  K=$(K): DMH-all complete ($(round(dmh_all.elapsed; digits=2)) s)"); flush(stdout)
    println("  K=$(K): estimating DMH-one"); flush(stdout)
    dmh_one, dmh_one_parts = run_dmh(
        artifact, config, :one, dmh_repetitions, seed + 4)
    println("  K=$(K): DMH-one complete ($(round(dmh_one.elapsed; digits=2)) s)"); flush(stdout)
    rows = [
        summary_row(K, "REINFORCE_ordinary", rf.ordinary, reference),
        summary_row(K, "REINFORCE_causal_return_to_go", rf.causal, reference),
        summary_row(K, "REINFORCE_causal_time_baseline", rf.baseline, reference),
        summary_row(K, "Thin_leave_one_out_N10", loo, reference),
        summary_row(K, "DMH_all_MaximumReflection", dmh_all, reference),
        summary_row(K, "DMH_one_MaximumReflection_Pruned", dmh_one, reference),
    ]
    data = Dict{Symbol,Any}(
        :ordinary => rf.ordinary.samples,
        :causal => rf.causal.samples,
        :baseline => rf.baseline.samples,
        :loo => loo.samples,
        :dmh_all => dmh_all.samples,
        :dmh_one => dmh_one.samples,
        :rows => Dict(
            :ordinary => rows[1], :causal => rows[2], :baseline => rows[3],
            :loo => rows[4], :dmh_all => rows[5], :dmh_one => rows[6],
        ),
        :dmh_all_ar => dmh_all_parts.ar_samples,
        :dmh_one_ar => dmh_one_parts.ar_samples,
        :reference => reference,
    )
    return (; rows, data, reference)
end

function option_value(argument, prefix, default)
    startswith(argument, prefix) || return default
    return parse(Int, argument[length(prefix) + 1:end])
end

function main(arguments=ARGS)
    artifact_directory = joinpath(OUTPUT_DIR, "artifact")
    output_directory = OUTPUT_DIR
    repetitions = DEFAULT_REPETITIONS
    pilot_repetitions = DEFAULT_PILOT_REPETITIONS
    dmh_repetitions = DEFAULT_DMH_REPETITIONS
    seed = DEFAULT_SEED
    quick = false
    only_k = nothing
    no_write = false
    selftest = false
    quick_batch_limit = 0
    for argument in arguments
        if argument == "--quick"
            quick = true
        elseif argument == "--no-write"
            no_write = true
        elseif argument == "--self-test"
            selftest = true
        elseif startswith(argument, "--artifact=")
            artifact_directory = argument[length("--artifact=") + 1:end]
        elseif startswith(argument, "--output=")
            output_directory = argument[length("--output=") + 1:end]
        elseif startswith(argument, "--reps=")
            repetitions = parse(Int, argument[length("--reps=") + 1:end])
        elseif startswith(argument, "--pilot=")
            pilot_repetitions = parse(Int, argument[length("--pilot=") + 1:end])
        elseif startswith(argument, "--dmh-reps=")
            dmh_repetitions = parse(Int, argument[length("--dmh-reps=") + 1:end])
        elseif startswith(argument, "--seed=")
            seed = parse(Int, argument[length("--seed=") + 1:end])
        elseif startswith(argument, "--only-k=")
            only_k = parse(Int, argument[length("--only-k=") + 1:end])
        elseif argument == "--help"
            println("Options: --artifact=DIR --output=DIR --quick --no-write --self-test")
            println("         --only-k=K")
            println("         --reps=N --pilot=N --dmh-reps=N --seed=N")
            return nothing
        else
            error("Unknown option: $(argument)")
        end
    end
    selftest && return self_test()
    if quick
        repetitions = min(repetitions, 1)
        pilot_repetitions = min(pilot_repetitions, 1)
        dmh_repetitions = min(dmh_repetitions, 1)
        quick_batch_limit = 1
    end
    artifact = load_artifact(artifact_directory)
    validate_artifact(artifact)
    if quick_batch_limit > 0
        artifact = quick_artifact(artifact, quick_batch_limit)
        validate_artifact(artifact)
        println("Quick smoke run uses the first $(size(artifact.x, 1)) observations of the fixed artifact batch")
    end
    coordinate = TARGET_COORDINATE_ZERO_BASED + 1
    reference = analytical_gradient(artifact)[coordinate]
    finite_difference = finite_difference_coordinate(artifact, coordinate)
    @assert isapprox(reference, finite_difference; atol=2e-5, rtol=2e-5)
    fixed_tape_validation(artifact, PPCAConfig(5), seed + 10)
    validate_mala_parameterization(artifact, PPCAConfig(5))

    println("PPCA diagnostic: latent=$(size(artifact.W, 2)), observations=$(size(artifact.W, 1)), batch=$(size(artifact.x, 1))")
    println("Julia MALA gamma=$(JULIA_MALA_GAMMA), source epsilon=$(SOURCE_EPSILON), coordinate=$(TARGET_COORDINATE_ZERO_BASED) zero-based")
    println("Analytical/reference gradient = $(reference); finite difference = $(finite_difference)")

    all_rows = Any[]
    case_data = Dict{Int,Any}()
    horizons = only_k === nothing ? (3, 5, 10) : (only_k,)
    for K in horizons
        println("Running K=$(K), repetitions=$(repetitions), pilot=$(pilot_repetitions), DMH repetitions=$(dmh_repetitions)")
        result = run_case(artifact, K, repetitions, pilot_repetitions,
                          dmh_repetitions, seed + K * 100)
        append!(all_rows, result.rows)
        push!(all_rows, reference_row(K, result.reference))
        case_data[K] = result.data
        for row in result.rows
            @printf("  %-38s mean=% .6e var=% .3e target_evals=%g acc=% .3f primary=% .3f alternative=% .3f\n",
                    row.method, row.mean, row.variance,
                    row.expensive_evals, row.acceptance,
                    row.primary_acceptance, row.alternative_acceptance)
        end
    end
    if !no_write
        mkpath(output_directory)
        write_rows(joinpath(output_directory, "ppca_results.csv"), all_rows)
        write_sample_rows(joinpath(output_directory, "ppca_samples.csv"), case_data)
        write_rows(joinpath(output_directory, "ppca_validation.csv"), [
            (name="analytical_vs_finite_difference", analytical=reference,
             finite_difference=finite_difference,
             absolute_error=abs(reference - finite_difference)),
            (name="source_epsilon", analytical=SOURCE_EPSILON,
             finite_difference=JULIA_MALA_GAMMA,
             absolute_error=0.0),
        ])
    end
    return all_rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
