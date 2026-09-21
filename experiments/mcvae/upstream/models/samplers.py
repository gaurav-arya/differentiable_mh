import math

import torch
import torch.nn as nn


def _require_finite_sampler(name, value):
    if torch.isfinite(value).all():
        return
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_count = int(finite.sum().item())
    if finite_count:
        finite_values = detached[finite]
        summary = (
            f"finite={finite_count}/{detached.numel()}, "
            f"min={finite_values.min().item():.6g}, "
            f"max={finite_values.max().item():.6g}"
        )
    else:
        summary = f"finite=0/{detached.numel()}"
    raise FloatingPointError(f"non-finite REINFORCE {name}: {summary}")


def _standard_normal_log_prob(value, zero, one):
    """Match the legacy Normal.log_prob arithmetic without object creation."""

    var = one ** 2
    log_scale = one.log()
    return (
        -((value - zero) ** 2) / (2 * var)
        - log_scale
        - math.log(math.sqrt(2 * math.pi))
    ).sum(1)


def acceptance_ratio(log_t, log_1_t, use_barker, return_pre_alphas=False,
                     log_probs=None):
    if use_barker:
        current_log_alphas_pre = log_t - log_1_t
    else:
        current_log_alphas_pre = torch.min(log_t, torch.zeros_like(log_t))

    if log_probs is None:
        log_probs = torch.log(torch.rand_like(log_t))
    a = log_probs <= current_log_alphas_pre

    if use_barker:
        current_log_alphas = torch.where(
            a, current_log_alphas_pre, -log_1_t
        )
    else:
        expression = torch.ones_like(current_log_alphas_pre) - torch.exp(current_log_alphas_pre)
        corr_expression = torch.log(expression + 1e-8)
        current_log_alphas = torch.where(
            a, current_log_alphas_pre, corr_expression
        )

    if not return_pre_alphas:
        return a, current_log_alphas
    else:
        return a, current_log_alphas, current_log_alphas_pre


def compute_grad(z, target, x, create_graph=True):
    flag = z.requires_grad  # True, if requires grad (means that we propagate gradients to some parameters)
    if not flag:
        z_ = z.detach().requires_grad_(True)
    else:
        z_ = z.requires_grad_(True)  ##  Do I need to clone it?
    with torch.enable_grad():
        grad = _get_grad(z=z_, target=target, x=x, create_graph=create_graph)
        if not flag:
            if not create_graph:
                grad = grad.detach()
            z_.requires_grad_(False)
        return grad


def _get_grad(z, target, x=None, create_graph=True):
    s = target(x=x, z=z)
    grad = torch.autograd.grad(s.sum(), z, create_graph=create_graph, only_inputs=True)[0]
    return grad


def run_chain(kernel, z_init, target, x=None, n_steps=100, return_trace=False, burnin=0):
    samples = z_init
    if not return_trace:
        for _ in range(burnin + n_steps):
            samples = kernel.make_transition(z=samples, target=target, x=x)[0].detach()
        return samples
    else:
        final = torch.tensor([], device=z_init.device, dtype=torch.float32)
        for i in range(burnin + n_steps):
            samples = kernel.make_transition(z=samples, target=target, x=x)[0].detach()
            if i >= burnin:
                final = torch.cat([final, samples])
        return final


class HMC(nn.Module):
    def __init__(self, n_leapfrogs, step_size, use_barker=False, partial_ref=False, learnable=False):
        '''
        :param n_leapfrogs: number of leapfrog iterations
        :param step_size: stepsize for leapfrog
        :param use_barker: If True -- Barker ratios applied. MH otherwise
        :param partial_ref: whether use partial refresh or not
        :param learnable: whether learnable (usage for Met model) or not
        '''
        super().__init__()
        self.n_leapfrogs = n_leapfrogs
        self.use_barker = use_barker
        self.partial_ref = partial_ref
        self.learnable = learnable
        self.register_buffer('zero', torch.tensor(0., dtype=torch.float32))
        self.register_buffer('one', torch.tensor(1., dtype=torch.float32))
        self.alpha_logit = nn.Parameter(self.zero, requires_grad=learnable)
        self.log_stepsize = nn.Parameter(torch.log(torch.tensor(step_size, dtype=torch.float32)),
                                         requires_grad=learnable)

    @property
    def step_size(self):
        return torch.exp(self.log_stepsize)

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def _forward_step(self, z_old, x=None, target=None, p_old=None):
        p_ = p_old + self.step_size / 2. * self.get_grad(z=z_old, target=target,
                                                         x=x)
        z_ = z_old
        for l in range(self.n_leapfrogs):
            z_ = z_ + self.step_size * p_
            if (l != self.n_leapfrogs - 1):
                p_ = p_ + self.step_size * self.get_grad(z=z_, target=target,
                                                         x=x)
        p_ = p_ + self.step_size / 2. * self.get_grad(z=z_, target=target,
                                                      x=x)
        return z_, p_

    def _make_transition(self, z_old, target, p_old=None, x=None):
        std_normal = torch.distributions.Normal(loc=self.zero, scale=self.one)

        ############ Then we compute new points and densities ############
        z_upd, p_upd = self.forward_step(z_old=z_old, p_old=p_old, target=target, x=x)

        target_log_density_f = target(z=z_upd, x=x) + std_normal.log_prob(p_upd).sum(-1)
        target_log_density_old = target(z=z_old, x=x) + std_normal.log_prob(p_old).sum(-1)

        log_t = target_log_density_f - target_log_density_old
        log_1_t = torch.logsumexp(torch.cat([torch.zeros_like(log_t).view(-1, 1),
                                             log_t.view(-1, 1)], dim=-1), dim=-1)  # log(1+t)

        a, current_log_alphas = acceptance_ratio(log_t=log_t, log_1_t=log_1_t, use_barker=self.use_barker)

        z_new = z_upd
        z_new[~a] = z_old[~a]

        p_new = -p_upd
        p_new[~a] = -p_old[~a]

        return z_new, p_new, a.to(torch.float32), current_log_alphas

    def make_transition(self, z, target, x=None, p=None):
        if p is None:
            p = torch.randn_like(z)
        if self.partial_ref:
            p = p * self.alpha + torch.sqrt(self.one - self.alpha ** 2) * torch.randn_like(p)
        z_new, p_new, a, current_log_alphas = self._make_transition(z_old=z,
                                                                    target=target, p_old=p, x=x)
        return z_new, p_new, a, current_log_alphas

    def forward_step(self, z_old, x=None, target=None, p_old=None):
        z_, p_ = self._forward_step(z_old=z_old, x=x, target=target, p_old=p_old)
        return z_, p_

    def get_grad(self, z, target, x=None):
        grad = compute_grad(z, target, x)
        return grad


class MALA(nn.Module):
    def __init__(self, step_size, use_barker, learnable):
        '''
        :param step_size: stepsize for leapfrog
        :param use_barker: If True -- Barker ratios applied. MH otherwise
        :param learnable: whether learnable (usage for Met model) or not
        '''
        super().__init__()
        self.use_barker = use_barker  # if use barker ratio
        self.learnable = learnable  # if stepsize are learnable
        self.capture_debug_details = False
        self.fail_fast = False
        self.register_buffer('zero', torch.tensor(0., dtype=torch.float32))
        self.register_buffer('one', torch.tensor(1., dtype=torch.float32))
        self.log_stepsize = nn.Parameter(torch.log(torch.tensor(step_size, dtype=torch.float32)),
                                         requires_grad=learnable)

    @property
    def step_size(self):
        return torch.exp(self.log_stepsize)

    def _forward_step(self, z_old, x=None, target=None):
        eps = torch.randn_like(z_old)
        forward_grad = self.get_grad(z=z_old,
                                     target=target,
                                     x=x)
        update = torch.sqrt(2 * self.step_size) * eps + self.step_size * forward_grad
        return z_old + update, update, eps, forward_grad

    def _transition_from_draws(self, z, target, x=None, eps=None,
                               log_uniform=None, create_graph=True,
                               step_size=None):
        """Run Thin's MALA arithmetic with optional recorded random draws.

        The replay implementation must not have a second version of the
        primary MALA calculation.  In particular, using a separately derived
        log-ratio or a different acceptance/state-selection expression can
        change the floating-point path even when the random tape is fixed.
        This helper is the one implementation used by both APIs; the normal
        API simply lets it draw ``eps`` and the accept coin itself.
        """
        if step_size is None:
            step_size = self.step_size
        if eps is None:
            eps = torch.randn_like(z)

        forward_grad = self.get_grad(
            z=z, target=target, x=x, create_graph=create_graph)
        update = torch.sqrt(2 * step_size) * eps + step_size * forward_grad
        z_upd = z + update

        std_normal = torch.distributions.Normal(loc=self.zero, scale=self.one)
        target_log_density_upd = target(z=z_upd, x=x)
        target_log_density_old = target(z=z, x=x)
        reverse_grad = self.get_grad(
            z=z_upd, target=target, x=x, create_graph=create_graph)
        eps_reverse = (-update - step_size * reverse_grad) / torch.sqrt(
            2 * step_size)
        proposal_density_numerator = std_normal.log_prob(eps_reverse).sum(1)
        proposal_density_denominator = std_normal.log_prob(eps).sum(1)

        log_t = (target_log_density_upd - target_log_density_old
                 - proposal_density_denominator + proposal_density_numerator)
        log_1_t = torch.logsumexp(torch.cat([
            torch.zeros_like(log_t).view(-1, 1),
            log_t.view(-1, 1),
        ], dim=-1), dim=-1)

        if self.fail_fast:
            for name, value in (
                ("proposal", z_upd),
                ("forward gradient", forward_grad),
                ("reverse gradient", reverse_grad),
                ("reverse noise", eps_reverse),
                ("proposal log density", proposal_density_numerator),
                ("current log density", proposal_density_denominator),
                ("log ratio", log_t),
                ("log normalizer", log_1_t),
            ):
                _require_finite_sampler(name, value)

        accepted, current_log_alphas, current_log_alphas_pre = acceptance_ratio(
            log_t, log_1_t, use_barker=self.use_barker,
            return_pre_alphas=True, log_probs=log_uniform)

        z_new = torch.empty_like(z_upd)
        z_new[accepted] = z_upd[accepted]
        z_new[~accepted] = z[~accepted]

        return {
            "z_new": z_new,
            "proposal": z_upd,
            "update": update,
            "mean": z + step_size * forward_grad,
            "forward_grad": forward_grad,
            "reverse_grad": reverse_grad,
            "eps": eps,
            "eps_reverse": eps_reverse,
            "target_log_density_proposal": target_log_density_upd,
            "target_log_density_current": target_log_density_old,
            "proposal_log_density": proposal_density_numerator,
            "current_log_density": proposal_density_denominator,
            "log_ratio": log_t,
            "log_alpha": current_log_alphas_pre,
            "alpha": torch.exp(current_log_alphas_pre),
            "accepted": accepted,
            "log_acceptance": current_log_alphas,
            "forward_energy": 0.5 * eps.square().sum(dim=1),
            "reverse_energy": 0.5 * eps_reverse.square().sum(dim=1),
            "step_size": step_size,
        }

    def transition_details(self, z, target, x=None, eps=None, log_uniform=None,
                           create_graph=True, step_size=None):
        """Return a replayable, differentiable MALA transition.

        This is an opt-in companion to :meth:`make_transition`.  In
        particular, callers can provide ``eps`` and ``log_uniform`` to replay
        a transition exactly, while retaining the proposal mean and acceptance
        probability needed by differentiable-MH estimators.

        ``create_graph=False`` is intended for counterfactual replay: it still
        computes the MALA score with respect to the state, but does not retain
        parameter derivatives for the detached branch reward.
        """
        if self.use_barker:
            raise ValueError("DMH replay currently supports standard MH only; set use_barker=False")

        details = self._transition_from_draws(
            z=z, target=target, x=x, eps=eps, log_uniform=log_uniform,
            create_graph=create_graph, step_size=step_size)
        return {
            "z_old": z,
            "z_new": details["z_new"],
            "proposal": details["proposal"],
            "update": details["update"],
            "mean": details["mean"],
            "eps": details["eps"],
            "forward_grad": details["forward_grad"],
            "reverse_grad": details["reverse_grad"],
            "log_ratio": details["log_ratio"],
            "log_alpha": details["log_alpha"],
            "alpha": details["alpha"],
            "target_log_density_proposal": details["target_log_density_proposal"],
            "target_log_density_current": details["target_log_density_current"],
            "forward_noise": details["eps"],
            "reverse_noise": details["eps_reverse"],
            "forward_energy": details["forward_energy"],
            "reverse_energy": details["reverse_energy"],
            "accepted": details["accepted"],
            "log_acceptance": details["log_acceptance"],
            "step_size": details["step_size"],
        }

    def evaluate_proposal(self, z, proposal, target, x=None, forward_grad=None,
                          create_graph=True, step_size=None, forward_noise=None,
                          forward_update=None):
        """Evaluate the MH ratio for an already constructed proposal."""
        if self.use_barker:
            raise ValueError("DMH replay currently supports standard MH only; set use_barker=False")

        if step_size is None:
            step_size = self.step_size
        if forward_grad is None:
            forward_grad = self.get_grad(
                z=z, target=target, x=x, create_graph=create_graph)
        reverse_grad = self.get_grad(
            z=proposal, target=target, x=x, create_graph=create_graph)
        if forward_update is None:
            update = proposal - z
        else:
            update = forward_update
        scale = torch.sqrt(2. * step_size)
        # When the proposal was generated from a taped draw, retain that draw
        # exactly.  Reconstructing it from ``proposal - z`` introduces a
        # subtraction of the forward drift and can become numerically unstable
        # once the learned target gradient is large.
        if forward_noise is None:
            eps = update / scale - torch.sqrt(step_size / 2.) * forward_grad
        else:
            eps = forward_noise
        eps_reverse = (-update - step_size * reverse_grad) / scale

        target_log_density_proposal = target(z=proposal, x=x)
        target_log_density_current = target(z=z, x=x)
        # The normalizing constants cancel in the MH ratio.  This replay-only
        # evaluator therefore uses the quadratic form directly, avoiding two
        # Normal distribution objects and their elementwise log_prob calls.
        proposal_density_numerator = _standard_normal_log_prob(
            eps_reverse, self.zero, self.one
        )
        proposal_density_denominator = _standard_normal_log_prob(
            eps, self.zero, self.one
        )
        log_ratio = (
            target_log_density_proposal - target_log_density_current
            - proposal_density_denominator
            + proposal_density_numerator
        )
        log_alpha = torch.min(log_ratio, torch.zeros_like(log_ratio))
        alpha = torch.exp(log_alpha)
        return {
            "reverse_grad": reverse_grad,
            "log_ratio": log_ratio,
            "log_alpha": log_alpha,
            "alpha": alpha,
            "target_log_density_proposal": target_log_density_proposal,
            "target_log_density_current": target_log_density_current,
            "forward_noise": eps,
            "reverse_noise": eps_reverse,
            "forward_energy": 0.5 * eps.square().sum(dim=1),
            "reverse_energy": 0.5 * eps_reverse.square().sum(dim=1),
        }

    def make_transition(self, z, target, x=None, eps=None, log_uniform=None):
        """
        Input:
        z_old - current position
        target - target distribution
        x - data object (optional)
        Output:
        z_new - new position
        current_log_alphas - current log_alphas, corresponding to sampled decision variables
        a - decision variables (0 or +1)
        """
        details = self._transition_from_draws(
            z=z, target=target, x=x, eps=eps, log_uniform=log_uniform,
            create_graph=True)

        if self.capture_debug_details:
            self._last_debug_details = {
                "proposal": details["proposal"].detach(),
                "mean": details["mean"].detach(),
                "forward_grad": details["forward_grad"].detach(),
                "log_ratio": details["log_ratio"].detach(),
                "log_alpha": details["log_alpha"].detach(),
                "accepted": details["accepted"].detach(),
                "log_acceptance": details["log_acceptance"].detach(),
            }
        return (details["z_new"], details["accepted"].to(torch.float32),
                details["log_acceptance"], details["forward_grad"])

    def get_grad(self, z, target, x=None, create_graph=True):
        grad = compute_grad(z, target, x, create_graph=create_graph)
        return grad


class ULA(nn.Module):
    def __init__(self, step_size, learnable=False, transforms=None, ula_skip_threshold=0.0):
        '''
        :param step_size: stepsize for leapfrog
        :param learnable: whether learnable (usage for Met model) or not
        '''
        super().__init__()
        self.learnable = learnable
        self.ula_skip_threshold = ula_skip_threshold
        self.register_buffer('zero', torch.tensor(0., dtype=torch.float32))
        self.register_buffer('one', torch.tensor(1., dtype=torch.float32))
        self.log_stepsize = nn.Parameter(torch.log(torch.tensor(step_size, dtype=torch.float32)),
                                         requires_grad=learnable)
        self.transforms = False
        self.add_nn = None
        self.scale_nn = None
        self.score_matching = False
        if transforms is not None:
            self.transforms = True
            self.add_nn = transforms()
            self.scale_nn = transforms()  ###just test with step size at the moment
            # self.scale_nn = lambda z, sign: 1.
            self.score_matching = True

    @property
    def step_size(self):
        return torch.exp(self.log_stepsize)

    def _forward_step(self, z_old, x=None, target=None):
        eps = torch.randn_like(z_old)
        self.log_jac = torch.zeros_like(z_old[:, 0])
        if not self.transforms:
            add = torch.zeros_like(z_old)
            forward_grad = self.get_grad(
                z=z_old,
                target=target,
                x=x)
            update = torch.sqrt(2 * self.step_size) * eps + self.step_size * forward_grad
            z_new = z_old + update
            eps_reverse = (z_old - z_new - self.step_size * self.get_grad(z=z_new, target=target, x=x)) / torch.sqrt(
                2 * self.step_size)
            score_match_cur = add
        else:
            add = self.add_nn(z=z_old, x=x)
            z_new = z_old + self.step_size * add + torch.sqrt(2 * self.step_size) * eps
            eps_reverse = (z_old - z_new - self.step_size * self.add_nn(z=z_new, x=x)) / torch.sqrt(2 * self.step_size)
            score_match_cur = (add - self.get_grad(z=z_old, target=target, x=x)) ** 2
            forward_grad = add
        return z_new, eps, eps_reverse, score_match_cur, forward_grad

    def scale_transform(self, z, sign='+'):
        S = torch.sigmoid(self.scale_nn(z))
        sign = {"+": 1., "-": -1.}[sign]
        self.log_jac += torch.sum(torch.log(S), dim=1) * sign
        return S

    def make_transition(self, z, target, x=None, reverse_kernel=None, mu_amortize=None):
        """
        Input:
        z_old - current position
        target - target distribution
        x - data object (optional)
        Output:
        z_new - new position
        current_log_alphas - current log_alphas, corresponding to sampled decision variables
        a - decision variables (0 or +1)
        """

        ############ Then we compute new points and densities ############
        std_normal = torch.distributions.Normal(loc=self.zero, scale=self.one)

        z_upd, eps, eps_reverse, score_match_cur, forward_grad = self._forward_step(z_old=z, x=x, target=target)

        if reverse_kernel is None:
            proposal_density_numerator = std_normal.log_prob(eps_reverse).sum(1)
        else:
            mu, logvar = reverse_kernel(torch.cat([z_upd, mu_amortize], dim=1))
            proposal_density_numerator = torch.distributions.Normal(loc=mu + z_upd, scale=torch.exp(0.5 * logvar)).log_prob(
                z).sum(1)

        proposal_density_denominator = std_normal.log_prob(eps).sum(1)

        z_new = z_upd

        ###
        with torch.no_grad():
            target_log_density_upd = target(z=z_upd, x=x)
            target_log_density_old = target(z=z, x=x)
            log_t = target_log_density_upd + proposal_density_numerator - target_log_density_old - proposal_density_denominator + self.log_jac
            log_1_t = torch.logsumexp(torch.cat([torch.zeros_like(log_t).view(-1, 1),
                                                 log_t.view(-1, 1)], dim=-1), dim=-1)  # log(1+t)
            if self.ula_skip_threshold > 0.:
                a, _, current_log_alphas_pre = acceptance_ratio(log_t, log_1_t, use_barker=False,
                                                                return_pre_alphas=True)
                acceptance_probs = torch.exp(current_log_alphas_pre)
                reject_mask = acceptance_probs <= self.ula_skip_threshold
            else:
                a, _ = acceptance_ratio(log_t, log_1_t, use_barker=False, return_pre_alphas=False)
                reject_mask = torch.zeros_like(a) < -1.
        ###
        if reject_mask.sum():
            # z_new[reject_mask] = z[reject_mask]
            z_new = torch.where(reject_mask[..., None], z, z_new)
            proposal_density_numerator[reject_mask] = torch.zeros_like(proposal_density_numerator[reject_mask])
            proposal_density_denominator[reject_mask] = torch.zeros_like(proposal_density_denominator[reject_mask])
            score_match_cur[reject_mask] = torch.zeros_like(score_match_cur[reject_mask])

        return z_new, proposal_density_numerator - proposal_density_denominator + self.log_jac, a.to(
            torch.float32), score_match_cur, forward_grad

    def get_grad(self, z, target, x=None):
        grad = compute_grad(z, target, x)
        return grad
