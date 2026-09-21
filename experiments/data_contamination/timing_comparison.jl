# Handwritten StochasticAD comparison for the data-contamination MH example.
#
# See Algorithms S1 and S2 for the corresponding unpruned and pruned estimators.
# This tracks the same straight-through MVD estimator as the
# StrategyWrapperFIsBackend(PrunedFIsBackend(...), StraightThroughStrategy())
# setup in analyze_data_contamination.jl, but stores finite perturbations as
# ordinary (delta, weight) pairs. In `pruned=true` mode, collisions are
# importance-pruned by |weight|, matching PrunedFIsBackend(Val(:weights)).
#
# Run from the experiments environment:
#   julia --project=experiments experiments/data_contamination/scratch_handwritten_comparison.jl

using Distributions
using LinearAlgebra
using Random
using Statistics

const DIM = 2
const N = 1_000_000
const N_REPLICATES_PRUNED = 100
const N_REPLICATES_UNPRUNED = 100
const THETA = 1e-6
const STEP = MvNormal(zeros(DIM), 2.0)

struct WeightedPerturbation
    dx1::Float64
    dx2::Float64
    weight::Float64
end

@inline function target_pdf_and_dtheta(x1, x2, theta)
    r2 = x1 * x1 + x2 * x2
    core = r2 < 1.0
    annulus = 9.0 < r2 < 25.0
    pdf = (1.0 - theta) * core + theta * annulus
    dpdf = annulus - core
    return pdf, dpdf
end

@inline function target_pdf(x1, x2, theta)
    r2 = x1 * x1 + x2 * x2
    return (1.0 - theta) * (r2 < 1.0) + theta * (9.0 < r2 < 25.0)
end

@inline function mh_alpha(x1, x2, y1, y2, theta)
    return min(1.0, target_pdf(y1, y2, theta) / target_pdf(x1, x2, theta))
end

@inline function mh_alpha_and_dtheta(x1, x2, y1, y2, theta)
    px, dpx = target_pdf_and_dtheta(x1, x2, theta)
    py, dpy = target_pdf_and_dtheta(y1, y2, theta)
    ratio = py / px
    if ratio < 1.0
        alpha = ratio
        dalpha = (dpy * px - py * dpx) / (px * px)
        return alpha, dalpha
    else
        return 1.0, 0.0
    end
end

# Maximal reflection coupling for two random-walk proposals with covariance 2I.
@inline function reflected_proposal(x1, x2, y1, y2, xp1, xp2, rng)
    d1 = y1 - x1
    d2 = y2 - x2
    j1 = xp1 - x1
    j2 = xp2 - x2
    d2norm = d1 * d1 + d2 * d2
    logu = log(rand(rng))
    if logu - (j1 * j1 + j2 * j2) / 4.0 <=
       -((j1 - d1)^2 + (j2 - d2)^2) / 4.0
        return xp1, xp2
    end
    factor = 1.0 - 2.0 * (d1 * j1 + d2 * j2) / d2norm
    return xp1 + d1 * factor, xp2 + d2 * factor
end

@inline function append_if_active!(events, dx1, dx2, weight)
    if (!iszero(weight)) && (!iszero(dx1) || !iszero(dx2))
        push!(events, WeightedPerturbation(dx1, dx2, weight))
    end
    return nothing
end

@inline function straight_through_event_weight(dalpha, alpha, coin)
    if dalpha > 0.0
        return coin == 0 ? dalpha : -dalpha
    elseif dalpha < 0.0
        return coin == 0 ? dalpha * alpha / (1.0 - alpha) :
                           -dalpha * (1.0 - alpha) / alpha
    else
        return 0.0
    end
end

function handwritten_dmh(n, burn_in, theta; pruned,
                         primal_rng, coupling_rng, accumulate_statistic=false)
    x1, x2 = 0.0, 0.0
    current = WeightedPerturbation[]
    next = WeightedPerturbation[]
    scratch_capacity = min(n + 1, max(256, n ÷ 100))
    sizehint!(current, scratch_capacity)
    sizehint!(next, scratch_capacity)

    alt_x1 = Vector{Float64}(undef, scratch_capacity)
    alt_x2 = similar(alt_x1)
    alt_prop1 = similar(alt_x1)
    alt_prop2 = similar(alt_x1)
    alt_alpha = similar(alt_x1)

    statistic_sum = 0.0
    n_statistic = 0

    for i in 1:n
        proposal_noise = rand(primal_rng, STEP)
        xp1, xp2 = x1 + proposal_noise[1], x2 + proposal_noise[2]
        alpha, dalpha = mh_alpha_and_dtheta(x1, x2, xp1, xp2, theta)

        # First propagate each existing perturbation through the coupled
        # proposal. The acceptance coupling below uses the same separate RNG.
        if length(current) > length(alt_x1)
            resize!(alt_x1, length(current))
            resize!(alt_x2, length(current))
            resize!(alt_prop1, length(current))
            resize!(alt_prop2, length(current))
            resize!(alt_alpha, length(current))
        end
        for j in eachindex(current)
            event = current[j]
            y1, y2 = x1 + event.dx1, x2 + event.dx2
            yp1, yp2 = reflected_proposal(x1, x2, y1, y2, xp1, xp2, coupling_rng)
            alpha_y, _ = mh_alpha_and_dtheta(y1, y2, yp1, yp2, theta)
            alt_x1[j], alt_x2[j] = y1, y2
            alt_prop1[j], alt_prop2[j] = yp1, yp2
            alt_alpha[j] = alpha_y
        end

        coin = rand(primal_rng, Bernoulli{Float64}(alpha))
        xnew1 = coin == 1 ? xp1 : x1
        xnew2 = coin == 1 ? xp2 : x2
        empty!(next)

        # Inversion coupling of each old Bernoulli event, conditional on the
        # primal accept/reject outcome.
        for j in eachindex(current)
            u = coin == 1 ? (1.0 - alpha) + rand(coupling_rng) * alpha :
                             rand(coupling_rng) * (1.0 - alpha)
            coin_y = u > 1.0 - alt_alpha[j] ? 1 : 0
            ynew1 = coin_y == 1 ? alt_prop1[j] : alt_x1[j]
            ynew2 = coin_y == 1 ? alt_prop2[j] : alt_x2[j]
            event = current[j]
            append_if_active!(next, ynew1 - xnew1, ynew2 - xnew2, event.weight)
        end

        # Straight-through Bernoulli MVD: the alternate coin is the opposite
        # outcome. Its conditional weight is the one used by StochasticAD's
        # StraightThroughStrategy, including the negative-dalpha case.
        if !iszero(dalpha)
            event_weight = straight_through_event_weight(dalpha, alpha, coin)
            alt_coin = 1 - coin
            ynew1 = alt_coin == 1 ? xp1 : x1
            ynew2 = alt_coin == 1 ? xp2 : x2
            append_if_active!(next, ynew1 - xnew1, ynew2 - xnew2, event_weight)
        end

        if pruned && length(next) > 1
            # With weighted pruning the live set contains at most one event;
            # the two candidates here are its continuation and this step's
            # fresh MVD event. The surviving weight is rescaled by its
            # selection probability.
            old = next[1]
            fresh = next[2]
            old_strength, fresh_strength = abs(old.weight), abs(fresh.weight)
            total_strength = old_strength + fresh_strength
            if iszero(total_strength)
                empty!(next)
            elseif rand(coupling_rng) < fresh_strength / total_strength
                next[1] = WeightedPerturbation(
                    fresh.dx1, fresh.dx2, copysign(total_strength, fresh.weight))
                resize!(next, 1)
            else
                next[1] = WeightedPerturbation(
                    old.dx1, old.dx2, copysign(total_strength, old.weight))
                resize!(next, 1)
            end
        end

        x1, x2 = xnew1, xnew2
        current, next = next, current

        if accumulate_statistic && i > burn_in
            base_f = x1 * x1 + x2 * x2
            for event in current
                alt_f = (x1 + event.dx1)^2 + (x2 + event.dx2)^2
                statistic_sum += event.weight * (alt_f - base_f)
            end
            n_statistic += 1
        end
    end

    estimate = n_statistic == 0 ? NaN : statistic_sum / n_statistic
    return (; estimate, final_state=(x1, x2), n_perturbations=length(current))
end

function primal_mh(n, theta, rng)
    x1, x2 = 0.0, 0.0
    for _ in 1:n
        noise = rand(rng, STEP)
        xp1, xp2 = x1 + noise[1], x2 + noise[2]
        alpha = mh_alpha(x1, x2, xp1, xp2, theta)
        if rand(rng, Bernoulli{Float64}(alpha)) == 1
            x1, x2 = xp1, xp2
        end
    end
    return x1, x2
end

function primal_mh_estimator(n, burn_in, theta, rng)
    x1, x2 = 0.0, 0.0
    statistic_sum = 0.0
    n_statistic = 0
    for i in 1:n
        noise = rand(rng, STEP)
        xp1, xp2 = x1 + noise[1], x2 + noise[2]
        alpha = mh_alpha(x1, x2, xp1, xp2, theta)
        if rand(rng, Bernoulli{Float64}(alpha)) == 1
            x1, x2 = xp1, xp2
        end
        if i > burn_in
            statistic_sum += x1 * x1 + x2 * x2
            n_statistic += 1
        end
    end
    return statistic_sum / n_statistic
end

function median_ns_per_transition(f, n; samples=5)
    f() # compile and warm up
    elapsed = Vector{Float64}(undef, samples)
    for i in eachindex(elapsed)
        elapsed[i] = @elapsed f()
    end
    return median(elapsed) * 1e9 / n
end

function main()
    # Keep RNG construction out of every timed run by closing over each stream.
    primal_rng = Random.Xoshiro(1234)
    primal_ns = median_ns_per_transition(() -> primal_mh(N, THETA, primal_rng), N)
    primal_rng_pruned = Random.Xoshiro(1234)
    coupling_rng_pruned = Random.Xoshiro(4321)
    pruned_ns = median_ns_per_transition(() -> handwritten_dmh(
        N, N, THETA; pruned=true,
        primal_rng=primal_rng_pruned, coupling_rng=coupling_rng_pruned), N)

    primal_rng_unpruned = Random.Xoshiro(1234)
    coupling_rng_unpruned = Random.Xoshiro(4321)
    unpruned_ns = median_ns_per_transition(() -> handwritten_dmh(
        N, N, THETA; pruned=false,
        primal_rng=primal_rng_unpruned, coupling_rng=coupling_rng_unpruned), N)

    # Also run the finite-sample statistic used in the tutorial, f(x)=||x||^2.
    # These runs include statistic accumulation; the timing benchmark above
    # measures transitions only.
    estimates_pruned = Vector{Float64}(undef, N_REPLICATES_PRUNED)
    estimates_unpruned = Vector{Float64}(undef, N_REPLICATES_UNPRUNED)
    seconds_pruned = similar(estimates_pruned)
    seconds_unpruned = similar(estimates_unpruned)
    seconds_mh_estimator = similar(estimates_unpruned)

    # Measure the reference in matched full-estimator units: same chain length,
    # burn-in, and statistic accumulation, but no derivative propagation.
    primal_mh_estimator(10_000, 5_000, THETA, Random.Xoshiro(90_000))
    for replicate in 1:N_REPLICATES_UNPRUNED
        seed = 1234 + replicate
        seconds_mh_estimator[replicate] = @elapsed primal_mh_estimator(
            N, N ÷ 2, THETA, Random.Xoshiro(seed))
    end

    # Warm both full-estimator paths before recording run times.
    handwritten_dmh(10_000, 5_000, THETA; pruned=true,
        primal_rng=Random.Xoshiro(90_001), coupling_rng=Random.Xoshiro(90_002),
        accumulate_statistic=true)
    handwritten_dmh(10_000, 5_000, THETA; pruned=false,
        primal_rng=Random.Xoshiro(90_001), coupling_rng=Random.Xoshiro(90_002),
        accumulate_statistic=true)

    for replicate in 1:N_REPLICATES_PRUNED
        seed = 1234 + replicate
        primal_rng_pruned = Random.Xoshiro(seed)
        coupling_rng_pruned = Random.Xoshiro(seed + 10_000)
        result_pruned = nothing
        seconds_pruned[replicate] = @elapsed result_pruned = handwritten_dmh(
            N, N ÷ 2, THETA; pruned=true,
            primal_rng=primal_rng_pruned, coupling_rng=coupling_rng_pruned,
            accumulate_statistic=true)
        estimates_pruned[replicate] = result_pruned.estimate
    end

    for replicate in 1:N_REPLICATES_UNPRUNED
        seed = 1234 + replicate
        primal_rng_unpruned = Random.Xoshiro(seed)
        coupling_rng_unpruned = Random.Xoshiro(seed + 10_000)
        result_unpruned = nothing
        seconds_unpruned[replicate] = @elapsed result_unpruned = handwritten_dmh(
            N, N ÷ 2, THETA; pruned=false,
            primal_rng=primal_rng_unpruned, coupling_rng=coupling_rng_unpruned,
            accumulate_statistic=true)
        estimates_unpruned[replicate] = result_unpruned.estimate
    end

    pruned_ratio = pruned_ns / primal_ns
    unpruned_ratio = unpruned_ns / primal_ns
    pruned_time = mean(seconds_pruned)
    unpruned_time = mean(seconds_unpruned)
    pruned_total_time = sum(seconds_pruned)
    unpruned_total_time = sum(seconds_unpruned)
    pruned_se = std(estimates_pruned) / sqrt(N_REPLICATES_PRUNED)
    unpruned_se = std(estimates_unpruned) / sqrt(N_REPLICATES_UNPRUNED)
    mh_chain_time = mean(seconds_mh_estimator)
    pruned_relative_cost = pruned_time / mh_chain_time
    unpruned_relative_cost = unpruned_time / mh_chain_time
    pruned_total_mh_units = pruned_total_time / mh_chain_time
    unpruned_total_mh_units = unpruned_total_time / mh_chain_time
    pruned_cost_variance = pruned_total_mh_units * pruned_se^2
    unpruned_cost_variance = unpruned_total_mh_units * unpruned_se^2
    println("Data-contamination hand-coded MVD comparison")
    println("n=$N, theta=$THETA, proposal covariance=2I")
    println("Transition-only median timings (ns/transition):")
    println("  plain MH:             $(round(primal_ns; sigdigits=4))")
    println("  pruned:               $(round(pruned_ns; sigdigits=4))  ($(round(pruned_ratio; sigdigits=4))× plain MH)")
    println("  unpruned:             $(round(unpruned_ns; sigdigits=4))  ($(round(unpruned_ratio; sigdigits=4))× plain MH)")
    println("Pooled finite-sample estimates for f(x)=||x||², burn-in=$(N ÷ 2):")
    println("  pruned:   runs=$N_REPLICATES_PRUNED, mean=$(mean(estimates_pruned)), run stddev=$(std(estimates_pruned)), pooled-mean SE=$pruned_se")
    println("  unpruned:        runs=$N_REPLICATES_UNPRUNED, mean=$(mean(estimates_unpruned)), run stddev=$(std(estimates_unpruned)), pooled-mean SE=$unpruned_se")
    println("Matched plain-MH estimator runtime per chain: $(mh_chain_time)s")
    println("Full-estimator total runtime: weighted-pruned=$(pruned_total_time)s, unpruned=$(unpruned_total_time)s")
    println("Per-chain runtime: pruned=$(pruned_relative_cost) MH chains, unpruned=$(unpruned_relative_cost) MH chains")
    println("Pooled total runtime: pruned=$(pruned_total_mh_units) MH chains, unpruned=$(unpruned_total_mh_units) MH chains")
    println("Total relative MH-chain cost × variance of pooled estimate: pruned=$(pruned_cost_variance), unpruned=$(unpruned_cost_variance)")
    println("Cost × pooled-variance ratio (pruned/unpruned): $(pruned_cost_variance / unpruned_cost_variance)")
end

main()

# 100 independent runs per mode; 1,000,000 transitions and 500,000 burn-in.
# Per-chain cost is relative to the matched plain-MH estimator runtime.
# Cost-variance is per-chain relative cost × variance across single-chain estimates.
#
# Mode     Mean estimate   Single-run SD   Cost (MH chains)   Cost × variance
# Pruned        1054.421         30.0117             1.7743           1598.08
# Unpruned      1051.366         8.04022            32.3348           2090.29
# Ratio (pruned/unpruned)        3.7327              0.05487             0.7645
