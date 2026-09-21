"""Differentiable Metropolis--Hastings replay for the neural A-MCVAE.

The code in this module deliberately keeps the counterfactual paths separate
from the primal path.  Counterfactual states and rewards are detached: they
are values used by the DMH zero-forward-value surrogate, not another source
of pathwise derivatives.

The first implementation is intentionally narrow.  It implements the
standard-MH MALA construction used by the Thin experiments, with a fixed
linear annealing schedule.  ``DMHRandomTape`` makes every random choice
explicit, which is useful both for debugging and for the parent-owned neural
diagnostic.
"""

from dataclasses import dataclass

import torch


def _log_standard_normal(value):
    return -0.5 * value.square().sum(dim=-1)


def _finite_summary(value):
    """Compact tensor summary for a fail-fast numerical error."""

    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_count = int(finite.sum().item())
    if finite_count:
        finite_values = detached[finite]
        value_range = (
            f"min={finite_values.min().item():.6g}, "
            f"max={finite_values.max().item():.6g}, "
            f"max_abs={finite_values.abs().max().item():.6g}"
        )
    else:
        value_range = "no finite values"
    return (
        f"shape={tuple(detached.shape)}, dtype={detached.dtype}, "
        f"device={detached.device}, finite={finite_count}/{detached.numel()}, "
        f"{value_range}"
    )


def _require_finite(name, value, context=""):
    if torch.isfinite(value).all():
        return
    suffix = f" ({context})" if context else ""
    raise FloatingPointError(
        f"non-finite DMH {name}{suffix}: {_finite_summary(value)}"
    )


def _require_finite_parameters(model, name, context=""):
    """Fail at the optimizer boundary that first corrupts model parameters."""

    for parameter_name, parameter in model.named_parameters():
        _require_finite(
            f"{name} parameter {parameter_name}",
            parameter,
            context,
        )


def _require_finite_gradients(model, name, context=""):
    """Fail before an invalid gradient can be passed to the optimizer."""

    for parameter_name, parameter in model.named_parameters():
        if parameter.grad is not None:
            _require_finite(
                f"{name} gradient {parameter_name}",
                parameter.grad,
                context,
            )


def _batch_context(model, extra=""):
    debugger = getattr(model, "_path_debugger", None)
    batch_idx = getattr(debugger, "batch_idx", None)
    parts = []
    if batch_idx is not None:
        parts.append(f"batch={batch_idx}")
    if extra:
        parts.append(extra)
    return ", ".join(parts)


@dataclass
class DMHRandomTape:
    """Random numbers consumed by a primal path and its replays.

    ``proposal_noise`` and ``accept_uniform`` have shape ``[K, P, D]`` and
    ``[K, P]``.  The two coupling arrays have shape ``[K, K, P]``; the first
    index is the branch's birth transition and the second is the future
    transition.  Entries on and below the diagonal are unused but retained
    to keep indexing simple and deterministic.
    """

    proposal_noise: torch.Tensor
    accept_uniform: torch.Tensor
    proposal_coupling_uniform: torch.Tensor
    accept_coupling_uniform: torch.Tensor
    pruning_uniform: torch.Tensor

    @classmethod
    def random_like(cls, z, n_transitions, generator=None):
        n_particles, latent_dim = z.shape
        noise = torch.randn(
            (n_transitions, n_particles, latent_dim),
            device=z.device,
            dtype=z.dtype,
            generator=generator,
        )
        # Open intervals avoid log(0) and make inversion coupling stable at
        # alpha=0 or alpha=1.
        def uniforms(shape):
            return torch.rand(
                shape, device=z.device, dtype=z.dtype, generator=generator
            ).clamp_(min=torch.finfo(z.dtype).eps, max=1.0 - torch.finfo(z.dtype).eps)

        return cls(
            proposal_noise=noise,
            accept_uniform=uniforms((n_transitions, n_particles)),
            proposal_coupling_uniform=uniforms(
                (n_transitions, n_transitions, n_particles)
            ),
            accept_coupling_uniform=uniforms(
                (n_transitions, n_transitions, n_particles)
            ),
            pruning_uniform=uniforms((n_transitions, n_particles)),
        )

    def validate(self, z, n_transitions):
        expected = (n_transitions, z.shape[0])
        if tuple(self.proposal_noise.shape[:2]) != expected:
            raise ValueError(
                "DMH proposal_noise must have shape [K, P, D], "
                f"got {tuple(self.proposal_noise.shape)}"
            )
        for name in (
            "accept_uniform",
            "pruning_uniform",
        ):
            value = getattr(self, name)
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"DMH {name} must have shape [K, P], got {tuple(value.shape)}"
                )
        expected_coupling = (n_transitions, n_transitions, z.shape[0])
        for name in ("proposal_coupling_uniform", "accept_coupling_uniform"):
            value = getattr(self, name)
            if tuple(value.shape) != expected_coupling:
                raise ValueError(
                    f"DMH {name} must have shape [K, K, P], "
                    f"got {tuple(value.shape)}"
                )


@dataclass
class DMHResult:
    """Primal output, surrogate, and diagnostics from one DMH replay."""

    z_transformed: torch.Tensor
    path: "DMHPath"
    sum_log_weights: torch.Tensor
    ar_surrogate: torch.Tensor
    all_acceptance: torch.Tensor
    pathwise_objective: torch.Tensor
    counterfactual_transitions: int
    meeting_count: torch.Tensor
    meeting_opportunities: int
    pruning_choices: int


@dataclass
class DMHPath:
    """Recorded primal path used as the source for all counterfactuals."""

    states: torch.Tensor
    proposals: torch.Tensor
    means: torch.Tensor
    acceptance_probabilities: torch.Tensor
    accepted: torch.Tensor
    increments: torch.Tensor
    initial_increment: torch.Tensor


def maximal_reflection_proposal(
    alternative_state,
    alternative_mean,
    primary_mean,
    primary_proposal,
    coupling_uniform,
    step_size,
    primary_noise=None,
    alternative_grad=None,
    return_noise=False,
    check_finite=True,
):
    """Construct the maximal-reflection coupling of two MALA proposals.

    The primary proposal is already sampled.  On the common part of the two
    Gaussian proposal laws the alternative proposal is exactly the primary
    proposal.  Otherwise the standardized primary noise is reflected across
    the hyperplane halfway between the two proposal means.
    """

    if check_finite:
        _require_finite("alternative state", alternative_state)
        _require_finite("alternative proposal mean", alternative_mean)
        _require_finite("primary proposal mean", primary_mean)
        _require_finite("primary proposal", primary_proposal)
        _require_finite("proposal step size", step_size)

    scale = torch.sqrt(2.0 * step_size)
    difference = (alternative_mean - primary_mean) / scale
    if primary_noise is None:
        # Backward-compatible fallback for callers without a tape.  The
        # primary DMH path supplies the exact draw below; reconstructing it
        # here is precisely the cancellation we must avoid during replay.
        primary_jump = (primary_proposal - primary_mean) / scale
    else:
        primary_jump = primary_noise
    difference_norm = difference.square().sum(dim=-1)

    common_log_test = (
        torch.log(coupling_uniform)
        + _log_standard_normal(primary_jump)
        <= _log_standard_normal(primary_jump - difference)
    )
    common_log_test = common_log_test | (difference_norm <= torch.finfo(difference.dtype).eps)

    safe_norm = difference_norm.clamp_min(torch.finfo(difference.dtype).eps)
    coefficient = 1.0 - 2.0 * (difference * primary_jump).sum(dim=-1) / safe_norm
    reflected_jump = coefficient[..., None] * difference
    alternative_noise_common = primary_jump - difference
    alternative_noise_reflected = primary_jump + reflected_jump - difference
    alternative_noise = torch.where(
        common_log_test[..., None],
        alternative_noise_common,
        alternative_noise_reflected,
    )
    if alternative_grad is None:
        alternative_drift = alternative_mean - alternative_state
    else:
        alternative_drift = step_size * alternative_grad
    alternative_update = scale * alternative_noise + alternative_drift
    reflected_proposal = alternative_state + alternative_update
    proposal = torch.where(
        common_log_test[..., None], primary_proposal, reflected_proposal
    )
    # For a common proposal the actual displacement is the primary proposal
    # minus the alternative state.  For a reflected proposal retain the
    # direct MALA update used to construct it.
    proposal_update = torch.where(
        common_log_test[..., None], proposal - alternative_state, alternative_update
    )
    if check_finite:
        _require_finite("maximal-reflection proposal", proposal)
        _require_finite("maximal-reflection noise", alternative_noise)

    # ``alternative_state`` is accepted as an argument to keep the function's
    # call site self-documenting and to make shape mistakes obvious.  The
    # proposal law itself only depends on the two means and the primary draw.
    if proposal.shape != alternative_state.shape:
        raise ValueError("MALA coupling proposal and alternative state disagree")
    if return_noise:
        return proposal, common_log_test, alternative_noise, proposal_update
    return proposal, common_log_test


def inversion_coupled_accept(
    primary_accept, primary_alpha, alternative_alpha, coupling_uniform
):
    """Couple two Bernoulli accept/reject decisions by inversion coupling."""

    primary_alpha = primary_alpha.detach().clamp(0.0, 1.0)
    alternative_alpha = alternative_alpha.detach().clamp(0.0, 1.0)
    coupling_uniform = coupling_uniform.clamp(
        min=torch.finfo(coupling_uniform.dtype).eps,
        max=1.0 - torch.finfo(coupling_uniform.dtype).eps,
    )
    lower = torch.where(
        primary_accept, 1.0 - primary_alpha, torch.zeros_like(primary_alpha)
    )
    upper = torch.where(
        primary_accept, torch.ones_like(primary_alpha), 1.0 - primary_alpha
    )
    conditioned_uniform = lower + (upper - lower) * coupling_uniform
    return conditioned_uniform > 1.0 - alternative_alpha


def _same_state(left, right):
    return (left == right).all(dim=-1)


def _branch_increment(model, state, x, init_logdensity, beta_increment, use_true_decoder):
    """Evaluate one detached AIS reward increment."""

    with torch.no_grad():
        joint = model.joint_logdensity(use_true_decoder=use_true_decoder)
        return (
            beta_increment * (joint(z=state, x=x) - init_logdensity(z=state))
        ).detach()


def _primary_target(init_logdensity, joint_logdensity, beta):
    return lambda z, x: (1.0 - beta) * init_logdensity(z=z) + beta * joint_logdensity(
        z=z, x=x
    )


def _counterfactual_step(
    model,
    alternative_state,
    primary_step,
    target,
    x,
    proposal_uniform,
    accept_uniform,
    primary_post,
    primary_increment,
    already_met=None,
    branch_root=None,
):
    """Advance one detached alternative through a recorded primal transition."""

    transition = model.transitions[primary_step["transition_index"]]
    step_size = primary_step["step_size"].detach()
    alternative_state = alternative_state.detach()
    debugger = getattr(model, "_path_debugger", None)
    fail_fast = bool(getattr(debugger, "fail_fast", False))
    context = ""
    if fail_fast:
        context = _batch_context(model, (
            f"transition={primary_step['transition_index']}, "
            f"step_size={_finite_summary(step_size)}, "
            f"input_state={_finite_summary(alternative_state)}"
        ))
    if debugger is not None and debugger.active:
        debugger.record(
            "dmh_counterfactual_input",
            values={
                "branch_root": branch_root,
                "transition": primary_step["transition_index"],
                "live_particles": (
                    int((~already_met).sum().item())
                    if already_met is not None else None
                ),
            },
            tensors={
                "alternative_state": alternative_state,
                "primary_post": primary_post,
                "step_size": step_size,
            },
        )
    if fail_fast:
        _require_finite("counterfactual input state", alternative_state, context)
    with torch.no_grad():
        alternative_grad = transition.get_grad(
            z=alternative_state,
            target=target,
            x=x,
            create_graph=False,
        )
        if fail_fast:
            _require_finite("counterfactual target gradient", alternative_grad, context)
        alternative_mean = alternative_state + step_size * alternative_grad
        if fail_fast:
            _require_finite("counterfactual proposal mean", alternative_mean, context)
        alternative_proposal, proposal_meets, alternative_noise, alternative_update = maximal_reflection_proposal(
            alternative_state=alternative_state,
            alternative_mean=alternative_mean,
            primary_mean=primary_step["mean"].detach(),
            primary_proposal=primary_step["proposal"].detach(),
            coupling_uniform=proposal_uniform,
            step_size=step_size,
            primary_noise=primary_step["eps"].detach(),
            alternative_grad=alternative_grad,
            return_noise=True,
            check_finite=fail_fast,
        )
        alternative_details = transition.evaluate_proposal(
            z=alternative_state,
            proposal=alternative_proposal,
            target=target,
            x=x,
            forward_grad=alternative_grad,
            create_graph=False,
            step_size=step_size,
            forward_noise=alternative_noise,
            forward_update=alternative_update,
        )
        if fail_fast:
            for name, value in (
                ("counterfactual proposal noise", alternative_noise),
                ("counterfactual proposal update", alternative_update),
                ("counterfactual reverse noise", alternative_details["reverse_noise"]),
                ("counterfactual log ratio", alternative_details["log_ratio"]),
                ("counterfactual alpha", alternative_details["alpha"]),
            ):
                _require_finite(name, value, context)
        alternative_accept = inversion_coupled_accept(
            primary_step["accepted"].detach(),
            primary_step["alpha"].detach(),
            alternative_details["alpha"],
            accept_uniform,
        )
        alternative_next_raw = torch.where(
            alternative_accept[..., None], alternative_proposal, alternative_state
        )
        met_raw = proposal_meets & _same_state(
            alternative_next_raw, primary_post.detach()
        )
        if already_met is None:
            already_met = torch.zeros_like(met_raw)
        alternative_next = torch.where(
            already_met[..., None], primary_post.detach(), alternative_next_raw
        )
        met = already_met | met_raw
        increment_raw = _branch_increment(
            model,
            alternative_next_raw,
            x,
            target.init_logdensity,
            target.beta_increment,
            target.use_true_decoder,
        )
        increment = torch.where(
            met,
            primary_increment.detach(),
            increment_raw,
        )
    if debugger is not None and debugger.active:
        debugger.record(
            "dmh_counterfactual_transition",
            values={
                "branch_root": branch_root,
                "transition": primary_step["transition_index"],
                "meeting_rate": float(met.float().mean().item()),
            },
            tensors={
                "alternative_grad": alternative_grad,
                "alternative_mean": alternative_mean,
                "alternative_proposal": alternative_proposal,
                "alternative_alpha": alternative_details["alpha"],
                "alternative_accept": alternative_accept,
                "alternative_next": alternative_next,
                "increment": increment,
            },
        )
    return alternative_next.detach(), increment.detach(), met.detach()


class _TargetInfo:
    """Small callable carrying the reward metadata needed by replay."""

    def __init__(self, fn, init_logdensity, beta_increment, use_true_decoder):
        self.fn = fn
        self.init_logdensity = init_logdensity
        self.beta_increment = beta_increment
        self.use_true_decoder = use_true_decoder

    def __call__(self, z, x):
        return self.fn(z=z, x=x)


def _counterfactual_split(
    model, primary_step, x, init_logdensity, beta_increment, use_true_decoder, primary_increment
):
    """Take the opposite decision at a branch root without a new transition."""

    alternative_state = torch.where(
        primary_step["accepted"][..., None],
        primary_step["z_old"].detach(),
        primary_step["proposal"].detach(),
    )
    debugger = getattr(model, "_path_debugger", None)
    fail_fast = bool(getattr(debugger, "fail_fast", False))
    if fail_fast:
        _require_finite(
            "counterfactual root state",
            alternative_state,
            _batch_context(model, f"transition={primary_step['transition_index']}"),
        )
    met = _same_state(alternative_state, primary_step["z_new"].detach())
    increment = _branch_increment(
        model,
        alternative_state,
        x,
        init_logdensity,
        beta_increment,
        use_true_decoder,
    )
    # At a meeting, the branch's reward is literally the primal reward.  This
    # assignment also removes harmless floating-point discrepancies from the
    # zero-forward DMH signal.
    increment = torch.where(met, primary_increment.detach(), increment)
    if debugger is not None and debugger.active:
        debugger.record(
            "dmh_counterfactual_root",
            values={"transition": primary_step["transition_index"]},
            tensors={
                "primary_state_before": primary_step["z_old"],
                "primary_proposal": primary_step["proposal"],
                "primary_state_after": primary_step["z_new"],
                "alternative_state": alternative_state,
                "met": met,
                "increment": increment,
            },
        )
    return alternative_state.detach(), increment.detach(), met.detach()


def _root_uniforms(tape, roots, transition_index):
    """Gather root-specific coupling uniforms for a vectorized DMH-one path."""

    values = tape.proposal_coupling_uniform[:, transition_index, :]
    return torch.gather(values, 0, roots[None, :]).squeeze(0)


def _root_accept_uniforms(tape, roots, transition_index):
    values = tape.accept_coupling_uniform[:, transition_index, :]
    return torch.gather(values, 0, roots[None, :]).squeeze(0)


def run_dmh_transitions(model, z, x, mu, logvar, mode, tape=None):
    """Run an A-MCVAE path and return a DMH-all or DMH-one surrogate.

    The returned ``ar_surrogate`` has zero forward value when used as
    ``raw - raw.detach()``.  Its backward pass is the accept/reject DMH
    contribution.  All branch rewards are detached before entering ``raw``.
    """

    if mode not in {"dmh_one", "dmh_all"}:
        raise ValueError(f"unknown DMH mode: {mode}")
    if any(transition.use_barker for transition in model.transitions):
        raise ValueError("DMH replay currently supports standard MH only")
    if model.annealing_scheme != "linear":
        raise ValueError("DMH replay currently supports the linear schedule only")
    n_transitions = model.K
    if tape is None:
        tape = DMHRandomTape.random_like(z, n_transitions)
    tape.validate(z, n_transitions)
    debugger = getattr(model, "_path_debugger", None)
    fail_fast = bool(getattr(debugger, "fail_fast", False))

    if fail_fast:
        context = _batch_context(model)
        _require_finite_parameters(model, "DMH pre-batch", context)
        _require_finite("primary initial state", z, context)

    beta = model.get_betas()
    init_logdensity = lambda z: torch.distributions.Normal(
        loc=mu, scale=torch.exp(0.5 * logvar)
    ).log_prob(z).sum(-1)
    joint_logdensity = model.joint_logdensity()

    initial_increment = (beta[1] - beta[0]) * (
        joint_logdensity(z=z, x=x) - init_logdensity(z)
    )
    state = z
    primary_steps = []
    primary_increments = []
    acceptances = []

    if debugger is not None and debugger.active:
        debugger.record("dmh_initial_state", tensors={"state": state})

    for transition_index in range(n_transitions):
        beta_index = transition_index + 1
        use_true_decoder = transition_index == n_transitions - 1
        target = _TargetInfo(
            _primary_target(init_logdensity, joint_logdensity, beta[beta_index]),
            init_logdensity,
            beta[beta_index + 1] - beta[beta_index],
            use_true_decoder,
        )
        transition = model.transitions[transition_index]
        state_before = state
        context = _batch_context(model, f"transition={transition_index}")
        if fail_fast:
            _require_finite("primary input state", state, context)
        details = transition.transition_details(
            z=state,
            target=target,
            x=x,
            eps=tape.proposal_noise[transition_index],
            log_uniform=torch.log(tape.accept_uniform[transition_index]),
            create_graph=True,
        )
        if fail_fast:
            for name in (
                "target_log_density_proposal",
                "target_log_density_current",
                "forward_noise",
                "reverse_noise",
                "forward_energy",
                "reverse_energy",
                "forward_grad",
                "mean",
                "proposal",
                "reverse_grad",
                "log_ratio",
                "log_alpha",
                "alpha",
            ):
                _require_finite(f"primary {name}", details[name], context)
        details["transition_index"] = transition_index
        details["target"] = target
        details["z_old"] = state
        state = details["z_new"]
        if fail_fast:
            _require_finite("primary output state", state, context)
        increment = target.beta_increment * (
            model.joint_logdensity(use_true_decoder=use_true_decoder)(z=state, x=x)
            - init_logdensity(z=state)
        )
        if fail_fast:
            _require_finite("primary increment", increment, context)
        primary_steps.append(details)
        primary_increments.append(increment)
        acceptances.append(details["accepted"].to(torch.float32))

        if debugger is not None and debugger.active:
            debugger.record(
                "dmh_primary_transition",
                values={
                    "transition": transition_index,
                    "beta": float(beta[beta_index].detach().item()),
                    "acceptance_rate": float(details["accepted"].float().mean().item()),
                },
                tensors={
                    "state_before": state_before,
                    "state_after": state,
                    "proposal": details["proposal"],
                    "mean": details["mean"],
                    "forward_grad": details["forward_grad"],
                    "log_ratio": details["log_ratio"],
                    "log_alpha": details["log_alpha"],
                    "alpha": details["alpha"],
                    "accepted": details["accepted"],
                    "increment": increment,
                    "step_size_used": details["step_size"],
                },
            )

        if model.variance_sensitive_step:
            model.update_stepsize(
                accept_rate=details["accepted"].to(torch.float32),
                current_tran_id=transition_index,
                current_gradient_batch=details["forward_grad"],
            )
            if debugger is not None and debugger.active:
                debugger.record(
                    "dmh_stepsize_update",
                    values={"transition": transition_index},
                    tensors={"step_size_after": transition.step_size},
                )
            if fail_fast:
                _require_finite("primary updated step size", transition.step_size, context)

    if not model.variance_sensitive_step:
        model.update_stepsize(accept_rate=torch.stack(acceptances).mean(dim=1))

    sum_log_weights = initial_increment + torch.stack(primary_increments).sum(dim=0)
    if fail_fast:
        _require_finite("primary log weights", sum_log_weights, _batch_context(model))
    primary_future = []
    for root in range(n_transitions):
        # The return attached to an accept/reject decision starts with the
        # AIS reward immediately after that decision, i.e. the increment at
        # the root transition itself, and continues through the final step.
        primary_future.append(torch.stack(primary_increments[root:]).sum(dim=0))

    meeting_count = torch.zeros_like(sum_log_weights)
    meeting_opportunities = torch.zeros((), device=z.device, dtype=torch.long)
    counterfactual_transitions = torch.zeros((), device=z.device, dtype=torch.long)
    pruning_choices = torch.zeros((), device=z.device, dtype=torch.long)

    if mode == "dmh_all":
        ar_raw = torch.zeros_like(sum_log_weights)
        for root in range(n_transitions):
            root_step = primary_steps[root]
            alternative_state, alternative_increment, met = _counterfactual_split(
                model,
                root_step,
                x,
                init_logdensity,
                root_step["target"].beta_increment,
                root_step["target"].use_true_decoder,
                primary_increments[root],
            )
            alternative_future = alternative_increment
            met_any = met
            meeting_count = meeting_count + met.to(sum_log_weights.dtype)
            for transition_index in range(root + 1, n_transitions):
                if bool(met_any.all()):
                    break
                live_count = (~met_any).sum()
                transition_target = primary_steps[transition_index]["target"]
                alternative_state, increment, met_now = _counterfactual_step(
                    model,
                    alternative_state,
                    primary_steps[transition_index],
                    transition_target,
                    x,
                    tape.proposal_coupling_uniform[root, transition_index],
                    tape.accept_coupling_uniform[root, transition_index],
                    primary_steps[transition_index]["z_new"],
                    primary_increments[transition_index],
                    already_met=met_any,
                    branch_root=root,
                )
                alternative_future = alternative_future + increment
                newly_met = met_now & (~met_any)
                met_any = met_any | met_now
                meeting_count = meeting_count + newly_met.to(sum_log_weights.dtype)
                meeting_opportunities = meeting_opportunities + live_count
                counterfactual_transitions = counterfactual_transitions + live_count
                if bool(met_any.all()):
                    break
            branch_difference = torch.where(
                root_step["accepted"].detach(),
                primary_future[root] - alternative_future,
                alternative_future - primary_future[root],
            ).detach()
            ar_raw = ar_raw + root_step["alpha"] * branch_difference
    else:
        # Uniform winner pruning.  The active candidate has one root index and
        # one inverse-probability-rescaled coefficient; the coefficient is
        # detached before it enters the reverse-mode surrogate.
        active_state = None
        active_increment = None
        active_met = None
        active_root = None
        active_wins = None
        active_coefficient = None
        ar_raw = torch.zeros_like(sum_log_weights)
        primary_alphas = torch.stack(
            [primary_steps[root]["alpha"] for root in range(n_transitions)],
            dim=1,
        )

        for transition_index in range(n_transitions):
            root_step = primary_steps[transition_index]
            new_state, new_increment, new_met = _counterfactual_split(
                model,
                root_step,
                x,
                init_logdensity,
                root_step["target"].beta_increment,
                root_step["target"].use_true_decoder,
                primary_increments[transition_index],
            )
            orientation = torch.where(
                root_step["accepted"],
                -torch.ones_like(sum_log_weights),
                torch.ones_like(sum_log_weights),
            )
            new_root = torch.full_like(
                root_step["accepted"], transition_index, dtype=torch.long
            )

            if active_state is None:
                active_state = new_state
                active_increment = new_increment
                active_met = new_met
                active_root = new_root
                active_wins = torch.ones_like(sum_log_weights)
                active_coefficient = orientation
            else:
                reset = active_met
                live_count = (~reset).sum()
                old_state, old_increment, old_met = _counterfactual_step(
                    model,
                    active_state,
                    primary_steps[transition_index],
                    root_step["target"],
                    x,
                    _root_uniforms(tape, active_root, transition_index),
                    _root_accept_uniforms(tape, active_root, transition_index),
                    root_step["z_new"],
                    primary_increments[transition_index],
                    branch_root="active",
                )
                newly_met = old_met & (~reset)
                meeting_count = meeting_count + newly_met.to(sum_log_weights.dtype)
                meeting_opportunities = meeting_opportunities + live_count
                counterfactual_transitions = counterfactual_transitions + live_count

                reset_after_meet = (~reset) & old_met
                combine = (~reset) & (~old_met)
                combined_wins = active_wins + 1.0
                probability_new = 1.0 / combined_wins
                choose_new = tape.pruning_uniform[transition_index] < probability_new
                choose_new = choose_new & combine
                keep_old = combine & (~choose_new)
                replace_active = reset | reset_after_meet | choose_new

                # Meeting or a previously empty population discards the old
                # candidate.  A live population uses inverse-probability
                # rescaling so its expected vector contribution is preserved.
                active_state = torch.where(
                    replace_active[..., None],
                    new_state,
                    torch.where(keep_old[..., None], old_state, active_state),
                )
                active_increment = torch.where(
                    replace_active,
                    new_increment,
                    torch.where(keep_old, old_increment, active_increment),
                )
                # Torch 1.7 cannot use torch.where with a Bool output on CPU.
                active_met = (
                    (replace_active & new_met)
                    | ((~replace_active) & keep_old & old_met)
                    | ((~replace_active) & (~keep_old) & active_met)
                )
                active_root = torch.where(
                    replace_active,
                    new_root,
                    active_root,
                )
                new_scale = probability_new.clamp_min(torch.finfo(z.dtype).eps).reciprocal()
                old_scale = (1.0 - probability_new).clamp_min(
                    torch.finfo(z.dtype).eps
                ).reciprocal()
                active_coefficient = torch.where(
                    reset | reset_after_meet,
                    orientation,
                    torch.where(
                        choose_new,
                        orientation * new_scale,
                        active_coefficient * old_scale,
                    ),
                )
                active_wins = torch.where(
                    combine, combined_wins, torch.ones_like(active_wins)
                )
                pruning_choices = pruning_choices + combine.sum()

            primary_increment = primary_increments[transition_index]
            branch_difference = (active_increment - primary_increment).detach()
            # alpha has shape [P]; coefficient columns are root-specific and
            # are multiplied by the corresponding root acceptance gradient.
            active_alpha = primary_alphas.gather(
                1, active_root[..., None]
            ).squeeze(1)
            ar_raw = ar_raw + (
                active_alpha * active_coefficient.detach() * branch_difference
            )

    return DMHResult(
        z_transformed=state,
        path=DMHPath(
            states=torch.stack([z] + [step["z_new"] for step in primary_steps]),
            proposals=torch.stack([step["proposal"] for step in primary_steps]),
            means=torch.stack([step["mean"] for step in primary_steps]),
            acceptance_probabilities=torch.stack(
                [step["alpha"] for step in primary_steps]
            ),
            accepted=torch.stack([step["accepted"] for step in primary_steps]),
            increments=torch.stack(primary_increments),
            initial_increment=initial_increment,
        ),
        sum_log_weights=sum_log_weights,
        ar_surrogate=ar_raw,
        all_acceptance=torch.stack(acceptances),
        pathwise_objective=sum_log_weights,
        counterfactual_transitions=int(counterfactual_transitions.item()),
        meeting_count=meeting_count,
        meeting_opportunities=int(meeting_opportunities.item()),
        pruning_choices=int(pruning_choices.item()),
    )
