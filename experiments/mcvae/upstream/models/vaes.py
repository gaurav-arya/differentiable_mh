import copy

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from metlibvi.vi import NormFlow

from models.aux import ULA_nn_sm
from models.decoders import get_decoder
from models.debug import PathDebugger
from models.encoders import get_encoder, backward_kernel_mnist
# from models.normflows import NormFlow
from models.dmh import (
    _require_finite_gradients,
    _require_finite_parameters,
    run_dmh_transitions,
)
from models.samplers import HMC, MALA, ULA


def binary_crossentropy_logits_stable(x, y):
    """
    Special binary crossentropy where y can be a single data batch, while
    x has several repeats
    """
    return torch.clamp(x, 0) - x * y + torch.log(1 + torch.exp(-torch.abs(x)))


def repeat_data(x, n_samples):
    '''
    Repeats data n_samples times, taking dimensionality into account
    '''
    if len(x.shape) == 4:
        x = x.repeat(n_samples, 1, 1, 1)
    else:
        x = x.repeat(n_samples, 1)
    return x


def gradient_norm(parameters):
    """Compute an unclipped gradient norm for skip-only manual updates."""

    squared = [
        parameter.grad.detach().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squared:
        return 0.
    return torch.sqrt(torch.stack(squared).sum()).item()


class Base(pl.LightningModule):
    '''
    Base class for all VAEs
    '''

    def __init__(self, num_samples, act_func, shape, hidden_dim, net_type,
                 dataset, specific_likelihood=None, sigma=1., name="VAE", **kwargs):
        """

        :param num_samples: how many latent samples per object to use
        :param act_func: activation function
        :param shape: image shape
        :param hidden_dim: hidden dim
        :param name: model name
        :param net_type: network type (conv or fc)
        :param dataset: dataset name
        :param kwargs: specific args
        """
        super().__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.hidden_dim = hidden_dim
        # Define Encoder part
        self.encoder_net = get_encoder(net_type, act_func, hidden_dim, dataset, shape=shape)
        # # Define Decoder part
        self.decoder_net = get_decoder(net_type, act_func, hidden_dim, dataset, shape=shape)
        # Number of latent samples per object
        self.num_samples = num_samples
        # Fixed random vector, which we recover each epoch
        self.random_z = torch.randn((64, hidden_dim), dtype=torch.float32)
        # Name, which is used for logging
        self.name = name
        # Transitions, which are going to be used for NLL estimation
        self.transitions_nll = nn.ModuleList(
            [HMC(n_leapfrogs=3, step_size=0.05, use_barker=False)
             for _ in range(5 - 1)])
        # We dont need these transitions to have trainable parameters
        for p in self.transitions_nll.parameters():
            p.requires_grad_(False)

        self.sigma = sigma  ## Noise for gaussian likelihood
        self.specific_likelihood = specific_likelihood

    def encode(self, x):
        # We treat the first half of output as mu, and the rest as logvar
        h = self.encoder_net(x)
        return h[:, :h.shape[1] // 2], h[:, h.shape[1] // 2:]

    def reparameterize(self, mu, logvar):
        # Reparametrization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def enc_rep(self, x, n_samples=1):
        # Encode and reparametrize
        mu, logvar = self.encode(x)
        mu = mu.repeat(n_samples, 1)
        logvar = logvar.repeat(n_samples, 1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder_net(z)

    def forward(self, z):
        return self.decode(z)

    def one_transition(self, current_num, z, x, annealing_logdens, mu=None, nll=False):
        # nll flag is to use those transitions, which are specifically defined for nll.
        z_new = self.transitions_nll[current_num].make_transition(z=z, x=x,
                                                                  target=annealing_logdens)
        return z_new

    def joint_logdensity(self, use_true_decoder=None):
        """
        Defines joint p(x, z). Needed for transitions-based models (AIS, ULA).
        :param use_true_decoder: When true, we use true decoder. Otherwise, could be auxilary model, or true one (if no auxilary).
        :return: joint logdensity of batchsize shape.
        """

        def density(z, x):
            if (use_true_decoder is not None) and use_true_decoder:
                x_reconst = self(z)
            elif hasattr(self, 'use_cloned_decoder') and self.use_cloned_decoder:
                x_reconst = self.cloned_decoder(z)
            else:
                x_reconst = self(z)

            log_Pr = torch.distributions.Normal(loc=torch.tensor(0., device=x.device, dtype=torch.float32),
                                                scale=torch.tensor(1., device=x.device, dtype=torch.float32)).log_prob(
                z).sum(-1)
            likelihood = self.get_likelihood(x_reconst, x)
            return likelihood + log_Pr

        return density

    def get_likelihood(self, x_reconst, x):
        if self.specific_likelihood is None:
            if self.dataset in ['mnist', 'fashionmnist']:
                likelihood = -binary_crossentropy_logits_stable(x_reconst.view(x_reconst.shape[0], -1),
                                                                x.view(x_reconst.shape[0], -1)).sum(-1)
            else:
                x_reconst = x_reconst.view(x_reconst.shape[0], -1)
                likelihood = torch.distributions.Normal(loc=x_reconst,
                                                        scale=self.sigma * torch.ones_like(x_reconst)).log_prob(
                    x.view(*x_reconst.shape)).sum(-1)
        elif self.specific_likelihood == 'bernoulli':
            likelihood = -binary_crossentropy_logits_stable(x_reconst.view(x_reconst.shape[0], -1),
                                                            x.view(x_reconst.shape[0], -1)).sum(-1)
        else:
            x_reconst = x_reconst.view(x_reconst.shape[0], -1)
            likelihood = torch.distributions.Normal(loc=x_reconst,
                                                    scale=self.sigma * torch.ones_like(x_reconst)).log_prob(
                x.view(*x_reconst.shape)).sum(-1)

        return likelihood

    def validation_epoch_end(self, outputs):
        # Tensorboard logging
        if "val_loss" in outputs[0].keys():  # if we have single loss
            val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/avg_val_loss', val_loss, self.current_epoch)
        else:  # if we have separate losses for encoder and decoder
            val_loss = torch.stack([x['val_loss_enc'] for x in outputs]).mean()
            val_loss_dec = torch.stack([x['val_loss_dec'] for x in outputs]).mean()
            self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/avg_val_loss_enc', val_loss,
                                              self.current_epoch)
            self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/avg_val_loss_dec', val_loss_dec,
                                              self.current_epoch)
            if "val_loss_score_match" in outputs[0].keys():  # if we have score matching loss
                val_loss_score_match = torch.stack([x['val_loss_score_match'] for x in outputs]).mean()
                self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/avg_val_loss_score_match',
                                                  val_loss_score_match,
                                                  self.current_epoch)

        if "acceptance_rate" in outputs[0].keys():  # if we have acceptance, we plot the mean after each transition
            acceptance = torch.stack([x['acceptance_rate'] for x in outputs]).mean(0)
            for i in range(len(acceptance)):
                self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/acceptance_rate_{i}',
                                                  acceptance[i].item(),
                                                  self.current_epoch)

        if "nll" in outputs[0].keys():  # if we have nll (typically at epoch 49), we plot it
            nll = torch.stack([x['nll'] for x in outputs]).mean(0)
            self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/nll', nll,
                                              self.current_epoch)

        if hasattr(self, 'transitions'):  # if we have transitions, we show median stepsize
            for i in range(len(self.transitions)):
                self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/median_step_size_{i}',
                                                  np.median(self.transitions[i].step_size.cpu().detach().numpy()),
                                                  self.current_epoch)

        if self.dataset.lower() in ['cifar', 'mnist', 'omniglot', 'celeba']:
            if self.dataset.lower().find('cifar') > -1:
                x_hat = self(self.random_z.to(val_loss.device)).view((-1, 3, 32, 32)).clamp(0., 1.)
                # x_hat = torch.sigmoid(self(self.random_z.to(val_loss.device))).view((-1, 3, 32, 32))
            elif self.dataset.lower().find('mnist') > -1:
                x_hat = torch.sigmoid(self(self.random_z.to(val_loss.device))).view((-1, 1, 28, 28))
            elif self.dataset.lower().find('omniglot') > -1:
                x_hat = torch.sigmoid(self(self.random_z.to(val_loss.device))).view((-1, 1, 105, 105))
            elif self.dataset.lower().find('celeba') > -1:
                x_hat = self(self.random_z.to(val_loss.device)).view((-1, 3, 64, 64)).clamp(0., 1.)
            grid = torchvision.utils.make_grid(x_hat)
            self.logger.experiment.add_image(f'{self.dataset}/{self.name}/image', grid, self.current_epoch)
        else:
            pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        output = self.step(batch)
        return {"loss": output[0]}

    def validation_step(self, batch, batch_idx):
        output = self.step(batch)
        d = {"val_loss": output[0]}
        # TODO: Bypass self.current_epoch here or 'dataset'
        if self.current_epoch % 10 == 9:
            nll = self.evaluate_nll(batch=batch,
                                    beta=torch.linspace(0., 1., 5, device=batch[0].device, dtype=torch.float32))
            d.update({"nll": nll})
        return d

    def evaluate_nll(self, batch, beta):
        # the function evaluates NLL using AIS
        x, _ = batch
        with torch.no_grad():
            n_samples = 10  # num samples for NLL estimation per data object
            z, mu, logvar = self.enc_rep(x, n_samples=n_samples)
            x = repeat_data(x, n_samples)
            if self.name == "VAE_with_flows":
                z_0 = z.clone()
                output = self.Flow(z_0)
                z = output["z_new"]

                def init_logdensity(z):
                    output = self.Flow.inverse(z)
                    z_0, log_jac = output["z_new"], output["aggregated_log_jac"]
                    return torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar)).log_prob(z_0).sum(
                        -1) + log_jac
            else:
                init_logdensity = lambda z: torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar)).log_prob(
                    z).sum(-1)
            annealing_logdens = lambda beta: lambda z, x: (1. - beta) * init_logdensity(
                z=z) + beta * self.joint_logdensity()(
                z=z,
                x=x)
            sum_log_weights = (beta[1] - beta[0]) * (self.joint_logdensity()(z=z, x=x) - init_logdensity(z))

            for i in range(1, len(beta) - 1):
                z = self.one_transition(current_num=i - 1, z=z, x=x,
                                        annealing_logdens=annealing_logdens(beta=beta[i]), nll=True)[0]

                sum_log_weights += (beta[i + 1] - beta[i]) * (self.joint_logdensity()(z=z, x=x) - init_logdensity(z=z))
            sum_log_weights = sum_log_weights.view(n_samples, batch[0].shape[0])
            batch_nll_estimator = -torch.logsumexp(sum_log_weights,
                                                   dim=0) + torch.log(torch.tensor(n_samples, dtype=torch.float32,
                                                                                   device=x.device))  # Should be a vector of batchsize containing nll estimator for each term of the batch

            return torch.mean(batch_nll_estimator)


class VAE(Base):
    def loss_function(self, recon_x, x, mu, logvar):
        batch_size = mu.shape[0] // self.num_samples

        # BCE = binary_crossentropy_logits_stable(recon_x.view(mu.shape[0], -1), x.view(mu.shape[0], -1)).view(
        #     (self.num_samples, batch_size, -1)).mean(0).sum(-1).mean()
        likelihood = self.get_likelihood(recon_x, x)
        likelihood = likelihood.view(self.num_samples, batch_size).mean(0).mean()
        KLD = -0.5 * torch.mean((1 + logvar - mu.pow(2) - logvar.exp()).view(
            (self.num_samples, -1, self.hidden_dim)).mean(0).sum(-1))
        loss = -likelihood + KLD
        return loss

    def step(self, batch):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x_hat = self(z)
        x = repeat_data(x, self.num_samples)
        loss = self.loss_function(x_hat, x, mu, logvar)
        return loss, x_hat, z


class IWAE(Base):
    def loss_function(self, recon_x, x, mu, logvar, z):
        batch_size = mu.shape[0] // self.num_samples
        log_Q = torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar)).log_prob(z).view(
            (self.num_samples, -1, self.hidden_dim)).sum(-1)

        log_Pr = torch.distributions.Normal(loc=torch.tensor(0., device=x.device, dtype=torch.float32),
                                            scale=torch.tensor(1., device=x.device, dtype=torch.float32)).log_prob(
            z).view((self.num_samples, -1, self.hidden_dim)).sum(-1)

        # BCE = binary_crossentropy_logits_stable(recon_x.view(mu.shape[0], -1), x.view(mu.shape[0], -1)).view(
        #     (self.num_samples, batch_size, -1)).sum(-1)
        likelihood = self.get_likelihood(recon_x, x)
        likelihood = likelihood.view(self.num_samples, batch_size)
        log_weight = log_Pr + likelihood - log_Q
        log_weight = log_weight - torch.max(log_weight, 0)[0]  # for stability
        weight = torch.exp(log_weight)
        weight = weight / torch.sum(weight, 0)
        weight = weight.detach()
        loss = torch.mean(torch.sum(weight * (-log_Pr - likelihood + log_Q), 0)) + np.log(1. * self.num_samples)

        return loss

    def step(self, batch):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x_hat = self(z)
        x = repeat_data(x, self.num_samples)
        loss = self.loss_function(x_hat, x, mu, logvar, z)
        return loss, x_hat, z


class VAE_with_flows(Base):
    def __init__(self, flow_type, num_flows, **kwargs):
        super().__init__(**kwargs)
        self.Flow = NormFlow(flow_type=flow_type, num_flows=num_flows, hidden_dim=self.hidden_dim, need_permute=True,
                             hidden_dims=[self.hidden_dim, self.hidden_dim])
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def loss_function(self, recon_x, x, mu, logvar, z, z_transformed, log_jac):
        batch_size = mu.shape[0] // self.num_samples
        likelihood = self.get_likelihood(recon_x, x)
        likelihood = likelihood.view(self.num_samples, batch_size).mean(0).mean()
        log_Q = torch.mean(torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar)).log_prob(z).view(
            (self.num_samples, batch_size, -1)).sum(-1) - log_jac.view((self.num_samples, -1)), dim=0)
        log_Pr = (-0.5 * z_transformed ** 2).view(
            (self.num_samples, batch_size, -1)).sum(-1).mean(0)
        KLD = torch.mean(log_Q - log_Pr)
        loss = -likelihood + KLD
        return loss

    def step(self, batch):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x = repeat_data(x, self.num_samples)
        output = self.Flow(z)
        z_transformed, log_jac = output["z_new"], output["aggregated_log_jac"]
        x_hat = self(z_transformed)
        loss = self.loss_function(recon_x=x_hat, x=x, mu=mu, logvar=logvar, z=z, z_transformed=z_transformed,
                                  log_jac=log_jac)
        return loss, x_hat, z_transformed


class BaseMCMC(Base):
    '''
    Base class for MCMC transition-based VAEs
    '''

    def __init__(self, step_size, K, grad_skip_val, grad_clip_val, use_cloned_decoder,
                 learnable_transitions, variance_sensitive_step, acceptance_rate_target,
                 bound_variance_sensitive_step=False, adam_lr=1e-3,
                 annealing_scheme='linear',
                 **kwargs):
        '''

        :param step_size: stepsize for transitions
        :param K: number of transitions
        :param grad_skip_val: threshold for gradient norm to skip gradient update. Set to 0 if dont need it
        :param grad_clip_val: clip gradients to be no more than this value. Set to 0 if dont need it
        :param use_cloned_decoder: Flag, if true we use cloned version of decoder to simplify grads for decoder through alphas
        :param learnable_transitions: Flag, if true we train stepsize of transitions
        :param variance_sensitive_step: Flag, if true we adapt stepsize in a variance awared manner. If false, adapt it to match target acceptance rate
        :param acceptance_rate_target: Target acceptance rate. Stepsize will be adjusted to fit it.
        :param annealing_scheme: 'linear', 'sigmoidal', 'all_learnable'
        :param kwargs: specific arguments
        '''
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.learnable_transitions = learnable_transitions
        self.K = K
        self.variance_sensitive_step = variance_sensitive_step
        if self.variance_sensitive_step:
            self.gamma_0 = [0.1 for _ in range(self.K)]  # an auxilary multipliers to adjust stepsize
        self.epsilons = [step_size for _ in range(self.K)]
        self.acceptance_rate_target = acceptance_rate_target  # target acceptance rate
        self.epsilon_decrease_alpha = 0.998
        self.epsilon_increase_alpha = 1.002
        self.epsilon_min = 0.001  # minimal possible stepsize
        self.epsilon_max = 0.5  # maximal possible stepsize
        self.bound_variance_sensitive_step = bound_variance_sensitive_step
        self.annealing_scheme = annealing_scheme
        linear_beta = torch.tensor(np.linspace(0., 1., self.K + 2), dtype=torch.float32)
        self.register_buffer('linear_beta', linear_beta)
        if self.annealing_scheme == 'linear':
            pass
        elif self.annealing_scheme == 'sigmoidal':
            self.tempering_logalpha = nn.Parameter(torch.tensor(-3., dtype=torch.float32))
        elif self.annealing_scheme == 'all_learnable':
            self.tempering_beta_logits = nn.Parameter(torch.zeros(self.K, dtype=torch.float32))
        else:
            raise ValueError('Please, select temrering scheme, which is one of [linear, sigmoidal, all_learnable].')
        self.grad_skip_val = grad_skip_val
        self.grad_clip_val = grad_clip_val
        self.adam_lr = adam_lr
        self.use_cloned_decoder = use_cloned_decoder
        self.use_stepsize_update = True
        if self.use_cloned_decoder:
            # In case we use cloned decoder, we create an exact copy of a true one in the beginning. After that, it will be trained in its own manner
            self.cloned_decoder = copy.deepcopy(self.decoder_net)

    def configure_optimizers(self):
        # all_params = list(self.decoder_net.parameters()) + list(self.encoder_net.parameters()) + list(
        #     self.transitions.parameters())
        # ## If we are using cloned decoder to approximate the true one, we add its params to inference optimizer
        # if self.use_cloned_decoder:
        #     all_params += list(self.cloned_decoder.parameters())
        optimizer = torch.optim.Adam(self.parameters(), lr=self.adam_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.25)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=10, epochs=5)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=20,
        #                                               mode="exp_range", gamma=0.85)
        return [optimizer], [scheduler]

    def run_transitions(self, z, x, mu, logvar):
        '''
        The function maintains the logic of transitions. It defines target, run specific transition, returns output
        :param z: current state
        :param x: data batch
        :param mu: parameter mu, returned by encoder
        :param logvar: parameter logvar, returned by encoder
        :return: List with specific content
        '''
        init_logdensity = lambda z: torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar)).log_prob(
            z).sum(-1)

        annealing_logdens = lambda beta: lambda z, x: (1. - beta) * init_logdensity(
            z=z) + beta * self.joint_logdensity()(
            z=z,
            x=x)

        output = self.specific_transitions(
            z=z,
            x=x,
            init_logdensity=init_logdensity,
            annealing_logdens=annealing_logdens,
            mu=mu
        )

        return output

    def update_stepsize(self, accept_rate=None, current_tran_id=None, current_gradient_batch=None):
        '''
        Stepsize update machinery.
        :param accept_rate: List of mean acceptance rates after each transitions.
        :param current_tran_id: Current transition id
        :param current_gradient_batch: Current batch of gradients of target logdensity wrt inputs
        :return:
        '''
        if self.use_stepsize_update:
            if not self.variance_sensitive_step:
                accept_rate = list(accept_rate.cpu().data.numpy())
                for l in range(0, self.K):
                    if accept_rate[l] < self.acceptance_rate_target:
                        self.epsilons[l] *= self.epsilon_decrease_alpha
                    else:
                        self.epsilons[l] *= self.epsilon_increase_alpha

                    if self.epsilons[l] < self.epsilon_min:
                        self.epsilons[l] = self.epsilon_min
                    if self.epsilons[l] > self.epsilon_max:
                        self.epsilons[l] = self.epsilon_max
                    self.transitions[l].log_stepsize.data = torch.tensor(np.log(self.epsilons[l]), dtype=torch.float32,
                                                                         device=self.transitions[l].log_stepsize.device)
            else:
                with torch.no_grad():
                    gradient_std = torch.std(current_gradient_batch, dim=0)
                    epsilon_target = (
                        self.gamma_0[current_tran_id] / (gradient_std + 1.)
                    )
                    updated_epsilon = (
                        0.9 * self.epsilons[current_tran_id]
                        + 0.1 * epsilon_target
                    )
                    if self.bound_variance_sensitive_step:
                        # The scalar acceptance-rate updater enforces these
                        # same bounds.  Bounding is opt-in so the historical
                        # REINFORCE baseline remains exactly reproducible.
                        updated_epsilon = torch.clamp(
                            updated_epsilon,
                            min=self.epsilon_min,
                            max=self.epsilon_max,
                        )
                    self.epsilons[current_tran_id] = updated_epsilon
                    self.transitions[current_tran_id].log_stepsize.data = torch.log(self.epsilons[current_tran_id])
                    observed_acceptance = float(
                        accept_rate.detach().float().mean().item()
                    )
                    if observed_acceptance < self.acceptance_rate_target:
                        self.gamma_0[current_tran_id] *= 0.99
                    else:
                        self.gamma_0[current_tran_id] *= 1.02
        else:
            pass

    def get_betas(self, ):
        if self.annealing_scheme == 'linear':
            betas = self.linear_beta
        elif self.annealing_scheme == 'sigmoidal':
            beta_0 = torch.sigmoid(-torch.exp(self.tempering_logalpha))
            beta_K_1 = torch.sigmoid(torch.exp(self.tempering_logalpha))
            betas_unnormed = torch.sigmoid(torch.exp(self.tempering_logalpha) * (2 * self.linear_beta - 1))
            betas = (betas_unnormed - beta_0) / (beta_K_1 - beta_0)

        elif self.annealing_scheme == 'all_learnable':
            betas = torch.flip(torch.cat([torch.zeros(1, dtype=torch.float32, device=self.tempering_beta_logits.device),
                                          torch.sigmoid(self.tempering_beta_logits),
                                          torch.ones(1, dtype=torch.float32,
                                                     device=self.tempering_beta_logits.device)]), dims=(0,))
            betas = torch.flip(torch.cumprod(betas, dim=0), dims=(0,))
        else:
            raise ValueError('Please, select temrering scheme, which is one of [linear, sigmoidal, all_learnable].')
        return betas

    def step(self, batch):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x = repeat_data(x, self.num_samples)
        z_transformed, sum_log_weights, _, all_acceptance = self.run_transitions(z=z, x=x, mu=mu, logvar=logvar)
        x_hat = self(z_transformed)
        loss = self.loss_function(z_transformed, x_hat, x, mu, logvar, sum_log_weights)
        return loss, all_acceptance, z_transformed

    def validation_step(self, batch, batch_idx):
        output = self.step(batch)
        d = {"val_loss": output[0], "acceptance_rate": output[1].mean(1)}
        if self.current_epoch % 10 == 9:
            nll = self.evaluate_nll(batch=batch,
                                    beta=torch.linspace(0., 1., 5, device=batch[0].device, dtype=torch.float32))
            d.update({"nll": nll})
        return d


class AMCVAE(BaseMCMC):
    '''
    Class for A-MCVAE
    '''

    def __init__(self, use_barker, use_alpha_annealing, gradient_estimator="reinforce", **kwargs):
        '''

        :param use_barker: Flag, whether to use Barker ratio or not
        :param kwargs: Args for BaseMCMC class
        '''
        super().__init__(**kwargs)
        if gradient_estimator not in {"reinforce", "dmh_one", "dmh_all"}:
            raise ValueError(
                "gradient_estimator must be one of reinforce, dmh_one, dmh_all"
            )
        if gradient_estimator != "reinforce" and use_barker:
            raise ValueError("DMH estimators require standard Metropolis acceptance")
        if gradient_estimator != "reinforce" and self.annealing_scheme != "linear":
            raise ValueError("DMH estimators currently require linear annealing")
        self.transitions = nn.ModuleList(
            [MALA(step_size=self.epsilons[0], use_barker=use_barker, learnable=self.learnable_transitions)
             for _ in range(self.K)])

        self.use_alpha_annealing = use_alpha_annealing
        self.gradient_estimator = gradient_estimator
        self._path_debugger = PathDebugger(gradient_estimator)
        for transition in self.transitions:
            transition.capture_debug_details = self._path_debugger.enabled
            transition.fail_fast = self._path_debugger.fail_fast
        self.save_hyperparameters()

    def one_transition(
        self,
        current_num,
        z,
        x,
        annealing_logdens,
        mu=None,
        nll=False,
        eps=None,
        log_uniform=None,
    ):
        """

        :param current_num: current transition number
        :param z: current state
        :param x: current batch
        :param annealing_logdens: current target (annealed combination of variational family and induced by decoder)
        :param nll: Whether to compute nll (we use another set of transitions) or not
        :return:
        """
        if nll:
            with torch.no_grad():
                z_new = self.transitions_nll[current_num].make_transition(z=z, x=x, target=annealing_logdens)
            return z_new
        else:
            z_new, directions, current_log_alphas, forward_grad = self.transitions[current_num].make_transition(z=z,
                                                                                                                x=x,
                                                                                                                target=annealing_logdens,
                                                                                                                eps=eps,
                                                                                                                log_uniform=log_uniform)
            return z_new, current_log_alphas, directions, forward_grad

    def specific_transitions(self, z, x, init_logdensity, annealing_logdens, mu, tape=None):
        """
        Specific transition for AIS VAE
        :param z: current state
        :param x: current batch
        :param init_logdensity: Initial log density (induced by encoder)
        :param annealing_logdens: current target (annealed combination of variational family and induced by decoder)
        :return: Specific output: final states, summed low weights, summed log alphas and acceptances
        """
        all_acceptance = torch.tensor([], dtype=torch.float32, device=x.device)
        beta = self.get_betas()
        sum_log_weights = (beta[1] - beta[0]) * (self.joint_logdensity()(z=z, x=x) - init_logdensity(z))
        sum_log_alphas = torch.zeros_like(z[:, 0])
        z_transformed = z
        debugger = self._path_debugger
        debugger.record("reinforce_initial_state", tensors={"state": z_transformed})
        for i in range(1, self.K + 1):
            state_before = z_transformed
            tape_kwargs = {}
            if tape is not None:
                tape_kwargs = {
                    "eps": tape.proposal_noise[i - 1],
                    "log_uniform": torch.log(tape.accept_uniform[i - 1]),
                }
            z_transformed, current_log_alphas, directions, forward_grad = self.one_transition(current_num=i - 1,
                                                                                              z=z_transformed,
                                                                                              x=x,
                                                                                              annealing_logdens=annealing_logdens(beta=beta[i]),
                                                                                              **tape_kwargs)
            sum_log_alphas += current_log_alphas

            increment = (beta[i + 1] - beta[i]) * (
                    self.joint_logdensity(use_true_decoder=(i == self.K))(z=z_transformed, x=x) - init_logdensity(
                z=z_transformed))
            sum_log_weights += increment
            debug_details = getattr(self.transitions[i - 1], "_last_debug_details", {})

            debugger.record(
                "reinforce_transition",
                values={
                    "transition": i - 1,
                    "beta": float(beta[i].detach().item()),
                    "acceptance_rate": float(directions.mean().detach().item()),
                },
                tensors={
                    "state_before": state_before,
                    "state_after": z_transformed,
                    "proposal": debug_details.get("proposal"),
                    "mean": debug_details.get("mean"),
                    "forward_grad": forward_grad,
                    "log_acceptance": current_log_alphas,
                    "accepted": directions,
                    "increment": increment,
                    "step_size_used": self.transitions[i - 1].step_size,
                },
            )

            all_acceptance = torch.cat([all_acceptance, directions[None]])
            if self.variance_sensitive_step:
                self.update_stepsize(accept_rate=directions, current_tran_id=i - 1, current_gradient_batch=forward_grad)
                debugger.record(
                    "reinforce_stepsize_update",
                    values={"transition": i - 1},
                    tensors={"step_size_after": self.transitions[i - 1].step_size},
                )

        if not self.variance_sensitive_step:
            self.update_stepsize(accept_rate=all_acceptance.mean(1))

        return z_transformed, sum_log_weights, sum_log_alphas, all_acceptance

    def step(self, batch):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x = repeat_data(x, self.num_samples)
        z_transformed, sum_log_weights, sum_log_alphas, all_acceptance = self.run_transitions(z=z,
                                                                                              x=x,
                                                                                              mu=mu,
                                                                                              logvar=logvar)

        loss = self.loss_function(sum_log_alphas=sum_log_alphas, sum_log_weights=sum_log_weights)
        return loss, all_acceptance, z_transformed

    def loss_function(self, sum_log_alphas, sum_log_weights):
        batch_size = sum_log_weights.shape[0] // self.num_samples
        elbo_est = sum_log_weights.view((self.num_samples, batch_size))
        ######
        elbo_est_normalized = 1.

        if self.use_alpha_annealing:
            annealing_coeff = np.minimum(1., 0.1 * self.current_epoch)
        else:
            annealing_coeff = 1.

        if self.num_samples > 1:
            multiplier = (self.num_samples * elbo_est.detach() - elbo_est.detach().sum(0)) / (
                    self.num_samples - 1.)
        else:
            multiplier = elbo_est.detach()

        loss = -torch.mean(elbo_est * elbo_est_normalized) - annealing_coeff * torch.mean(
            multiplier * sum_log_alphas.view((self.num_samples, batch_size)))
        return loss

    def training_step(self, batch, batch_idx):
        debugger = self._path_debugger
        fail_fast = bool(getattr(debugger, "fail_fast", False))
        if fail_fast:
            _require_finite_parameters(
                self,
                "training pre-batch",
                f"batch={batch_idx}, estimator={self.gradient_estimator}",
            )
        if self.gradient_estimator != "reinforce":
            x, _ = batch
            z, mu, logvar = self.enc_rep(x, self.num_samples)
            x = repeat_data(x, self.num_samples)
            debugger.begin(
                batch_idx,
                tensors={"x": x, "z_initial": z, "mu": mu, "logvar": logvar},
            )
            dmh_result = run_dmh_transitions(
                model=self,
                z=z,
                x=x,
                mu=mu,
                logvar=logvar,
                mode=self.gradient_estimator,
            )
            pathwise_loss = -dmh_result.sum_log_weights.mean()
            ar_signal = dmh_result.ar_surrogate.mean()
            # The DMH branch reward is detached.  Subtracting its detached
            # value makes the surrogate numerically zero while preserving its
            # accept/reject gradient in reverse mode.
            if self.use_alpha_annealing:
                dmh_ar_coefficient = np.minimum(1., 0.1 * self.current_epoch)
            else:
                dmh_ar_coefficient = 1.
            ar_loss = -dmh_ar_coefficient * (ar_signal - ar_signal.detach())
            loss = pathwise_loss + ar_loss

            debugger.record(
                "dmh_objective",
                values={
                    "counterfactual_transitions": dmh_result.counterfactual_transitions,
                    "meeting_opportunities": dmh_result.meeting_opportunities,
                },
                tensors={
                    "sum_log_weights": dmh_result.sum_log_weights,
                    "pathwise_loss": pathwise_loss,
                    "ar_signal": ar_signal,
                    "ar_loss": ar_loss,
                    "loss": loss,
                    "acceptance": dmh_result.all_acceptance,
                },
            )

            self.log("train_pathwise_loss", pathwise_loss.detach(), on_step=True, prog_bar=False)
            self.log("train_dmh_ar_signal", ar_signal.detach(), on_step=True, prog_bar=False)
            self.log(
                "train_dmh_ar_coefficient",
                float(dmh_ar_coefficient),
                on_step=True,
                prog_bar=False,
            )
            self.log(
                "train_dmh_acceptance_rate",
                dmh_result.all_acceptance.mean().detach(),
                on_step=True,
                prog_bar=False,
            )
            self.log(
                "train_dmh_counterfactual_transitions",
                float(dmh_result.counterfactual_transitions),
                on_step=True,
                prog_bar=False,
            )
            if dmh_result.meeting_opportunities:
                self.log(
                    "train_dmh_meeting_rate",
                    (dmh_result.meeting_count.sum() /
                     float(dmh_result.meeting_opportunities)).detach(),
                    on_step=True,
                    prog_bar=False,
                )

            optimizer = self.optimizers()
            self.manual_backward(loss, optimizer)
            all_params = [parameter for parameter in self.parameters() if parameter.requires_grad]
            if fail_fast:
                _require_finite_gradients(
                    self,
                    "DMH pre-clip",
                    f"batch={batch_idx}",
                )
            if self.grad_clip_val != 0.:
                grad_norm = torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip_val).item()
            else:
                grad_norm = gradient_norm(all_params)
            if fail_fast:
                _require_finite_gradients(
                    self,
                    "DMH post-clip",
                    f"batch={batch_idx}, reported_grad_norm={grad_norm}",
                )
            debugger.gradients(
                "dmh_backward",
                all_params,
                values={"reported_grad_norm": float(grad_norm)},
            )
            if self.grad_skip_val != 0.:
                if grad_norm < self.grad_skip_val:
                    optimizer.step()
            else:
                optimizer.step()
            if fail_fast:
                _require_finite_parameters(
                    self,
                    "DMH post-step",
                    f"batch={batch_idx}",
                )
            debugger.end()
            return {"loss": loss.detach()}

        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x = repeat_data(x, self.num_samples)
        debugger.begin(
            batch_idx,
            tensors={"x": x, "z_initial": z, "mu": mu, "logvar": logvar},
        )
        z_transformed, sum_log_weights, sum_log_alphas, all_acceptance = self.run_transitions(z=z,
                                                                                              x=x,
                                                                                              mu=mu,
                                                                                              logvar=logvar)
        loss = self.loss_function(sum_log_alphas=sum_log_alphas, sum_log_weights=sum_log_weights)

        debugger.record(
            "reinforce_objective",
            tensors={
                "sum_log_weights": sum_log_weights,
                "sum_log_alphas": sum_log_alphas,
                "loss": loss,
                "acceptance": all_acceptance,
            },
        )

        if (self.grad_skip_val != 0.) or (self.grad_clip_val != 0.):
            optimizer = self.optimizers()
            self.manual_backward(loss, optimizer)
            all_params = list(self.decoder_net.parameters()) + list(self.encoder_net.parameters()) + list(
                self.transitions.parameters())
            ## If we are using cloned decoder to approximate the true one, we add its params to inference optimizer
            if self.use_cloned_decoder:
                all_params += list(self.cloned_decoder.parameters())

            if fail_fast:
                _require_finite_gradients(
                    self,
                    "REINFORCE pre-clip",
                    f"batch={batch_idx}",
                )
            if self.grad_clip_val != 0.:
                grad_norm = torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip_val).item()
            else:
                grad_norm = gradient_norm(all_params)
            if fail_fast:
                _require_finite_gradients(
                    self,
                    "REINFORCE post-clip",
                    f"batch={batch_idx}, reported_grad_norm={grad_norm}",
                )
            debugger.gradients(
                "reinforce_backward",
                all_params,
                values={"reported_grad_norm": float(grad_norm)},
            )
            if self.grad_skip_val != 0.:
                if grad_norm < self.grad_skip_val:
                    optimizer.step()
            else:
                optimizer.step()
            if fail_fast:
                _require_finite_parameters(
                    self,
                    "REINFORCE post-step",
                    f"batch={batch_idx}",
                )
            debugger.end()
        else:
            debugger.end()
            return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        if self.gradient_estimator == "reinforce":
            return super().validation_step(batch, batch_idx)

        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x = repeat_data(x, self.num_samples)
        _, sum_log_weights, _, all_acceptance = self.run_transitions(
            z=z, x=x, mu=mu, logvar=logvar
        )
        output = {
            "val_loss": -sum_log_weights.mean(),
            "acceptance_rate": all_acceptance.mean(1),
        }
        if self.current_epoch % 10 == 9:
            nll = self.evaluate_nll(
                batch=batch,
                beta=torch.linspace(0., 1., 5, device=batch[0].device, dtype=torch.float32),
            )
            output["nll"] = nll
        return output


class LMCVAE(BaseMCMC):
    '''
    Class for L-MCVAE
    '''

    def __init__(self, use_score_matching, ula_skip_threshold, use_reverse_kernel=False, **kwargs):
        '''

        :param use_score_matching: If true, we are using score matching (we do not estimate the correct gradient)
        :param kwargs: Parameters for base function
        '''
        super().__init__(**kwargs)
        self.score_matching = use_score_matching
        if self.score_matching:
            additional_dims = 2 if kwargs['dataset'] == 'toy' else kwargs['shape'] ** 2
            transforms = lambda: ULA_nn_sm(input=kwargs['hidden_dim'] + additional_dims,
                                           output=kwargs['hidden_dim'],
                                           hidden=(kwargs['hidden_dim'] * 10, kwargs['hidden_dim'] * 10),
                                           h_dim=None)
        else:
            transforms = None
        self.transitions = nn.ModuleList(
            [ULA(step_size=self.epsilons[0], learnable=self.learnable_transitions, transforms=transforms,
                 ula_skip_threshold=ula_skip_threshold) for _ in range(self.K)])

        if use_reverse_kernel:
            self.reverse_kernels = nn.ModuleList(
                [backward_kernel_mnist(act_func=nn.GELU, hidden_dim=self.hidden_dim) for _ in range(self.K)])
        else:
            self.reverse_kernels = [None for _ in range(self.K)]

        self.save_hyperparameters()

    def one_transition(self, current_num, z, x, annealing_logdens, mu=None, nll=False):
        if nll:
            z_new = self.transitions_nll[current_num].make_transition(z=z, x=x,
                                                                      target=annealing_logdens)
            return z_new
        else:
            z_new, current_log_weights, directions, score_match_cur, forward_grad = self.transitions[
                current_num].make_transition(z=z,
                                             x=x,
                                             target=annealing_logdens,
                                             reverse_kernel=self.reverse_kernels[current_num],
                                             mu_amortize=mu)
            return z_new, current_log_weights, directions, score_match_cur, forward_grad

    def specific_transitions(self, z, x, init_logdensity, annealing_logdens, mu):
        all_acceptance = torch.tensor([], dtype=torch.float32, device=x.device)
        beta = self.get_betas()
        sum_log_weights = -init_logdensity(z=z)
        loss_sm = torch.zeros_like(z)
        z_transformed = z
        for i in range(1, self.K + 1):
            z_transformed, current_log_weights, directions, score_match_cur, forward_grad = self.one_transition(
                current_num=i - 1,
                z=z_transformed,
                x=x,
                annealing_logdens=annealing_logdens(beta=beta[i]),
                mu=mu)
            loss_sm += score_match_cur
            sum_log_weights += current_log_weights
            all_acceptance = torch.cat([all_acceptance, directions[None]])
            if self.variance_sensitive_step:
                self.update_stepsize(accept_rate=directions, current_tran_id=i - 1, current_gradient_batch=forward_grad)
        sum_log_weights += self.joint_logdensity(use_true_decoder=True)(z=z_transformed, x=x)
        if not self.variance_sensitive_step:
            self.update_stepsize(accept_rate=all_acceptance.mean(1))
        return z_transformed, sum_log_weights, loss_sm, all_acceptance

    def step(self, batch):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x = repeat_data(x, self.num_samples)
        z_transformed, sum_log_weights, loss_sm, all_acceptance = self.run_transitions(z=z,
                                                                                       x=x,
                                                                                       mu=mu,
                                                                                       logvar=logvar)

        loss = self.loss_function(sum_log_weights=sum_log_weights)
        loss_sm = loss_sm.sum(1).mean()
        return loss, all_acceptance, loss_sm, z_transformed

    def loss_function(self, sum_log_weights):
        batch_size = sum_log_weights.shape[0] // self.num_samples
        elbo_est = sum_log_weights.view((self.num_samples, batch_size))
        if self.num_samples > 1:
            with torch.no_grad():
                log_weight = elbo_est - torch.max(elbo_est, 0)[0]  # for stability
                weight = torch.exp(log_weight)
                weight = weight / torch.sum(weight, 0)
                weight = weight.detach()
            loss = -torch.mean(torch.sum(weight * elbo_est, dim=0))
        else:
            elbo_est = elbo_est.sum(0)
            loss = -torch.mean(elbo_est)
        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x = repeat_data(x, self.num_samples)
        z_transformed, sum_log_weights, loss_sm, all_acceptance = self.run_transitions(z=z,
                                                                                       x=x,
                                                                                       mu=mu,
                                                                                       logvar=logvar)
        loss = self.loss_function(sum_log_weights) + loss_sm.sum(1).mean()
        if (self.grad_skip_val != 0.) or (self.grad_clip_val != 0.):
            optimizer = self.optimizers()
            self.manual_backward(loss, optimizer)
            all_params = list(self.decoder_net.parameters()) + list(self.encoder_net.parameters()) + list(
                self.transitions.parameters())
            ## If we are using cloned decoder to approximate the true one, we add its params to inference optimizer
            if self.use_cloned_decoder:
                all_params += list(self.cloned_decoder.parameters())

            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip_val).item()
            if self.grad_skip_val != 0.:
                if grad_norm < self.grad_skip_val:
                    optimizer.step()
            else:
                optimizer.step()
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        output = self.step(batch)
        d = {"val_loss": output[0], "acceptance_rate": output[1].mean(1), "val_loss_score_match": output[2]}
        if self.current_epoch % 10 == 9:
            nll = self.evaluate_nll(batch=batch,
                                    beta=torch.linspace(0., 1., 5, device=batch[0].device, dtype=torch.float32))
            d.update({"nll": nll})
        return d
