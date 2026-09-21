import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Base(pl.LightningModule):
    def __init__(self, act_func, num_samples, hidden_dim, name="VAE"):
        super(Base, self).__init__()
        self.hidden_dim = hidden_dim
        # Define Encoder part
        self.encoder_net = nn.Sequential(
            nn.Linear(784, 400),
            act_func(),
            nn.Linear(400, 2 * hidden_dim)
        )
        # # Define Decoder part
        self.decoder_net = nn.Sequential(
            nn.Linear(hidden_dim, 400),
            act_func(),
            nn.Linear(400, 784)
        )
        # Number of latent samples per object
        self.num_samples = num_samples
        # Fixed random vector, which we recover each epoch
        self.random_z = torch.randn((64, hidden_dim), dtype=torch.float32)
        # Name, which is used for logging
        self.name = name

    def encode(self, x):
        # We treat the first half of output as mu, and the rest as logvar
        h = self.encoder_net(x)
        return h[:, :h.shape[1] // 2], h[:, h.shape[1] // 2:]

    def reparameterize(self, mu, logvar):
        # Reparametrization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder_net(z)

    def forward(self, z):
        return self.decode(z)

    def joint_density(self, ):
        # Defines joint density (in fact, this is useful only for VAE with MCMC)
        def density(z, x):
            x_reconst = self(z)
            log_Pr = torch.distributions.Normal(loc=torch.tensor(0., device=z.device, dtype=torch.float32),
                                                scale=torch.tensor(1., device=z.device, dtype=torch.float32)).log_prob(
                z).sum(-1)
            return -F.binary_cross_entropy_with_logits(x_reconst, x.view(-1, 784), reduction='none').sum(-1) + log_Pr

        return density

    def validation_epoch_end(self, outputs):
        # Some stuff, which is needed for logging and tensorboard
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_BCE = torch.stack([x['BCE'] for x in outputs]).mean()

        x_hat = torch.sigmoid(self(self.random_z.to(val_loss.device))).view((-1, 1, 28, 28))
        grid = torchvision.utils.make_grid(x_hat).mean(0, keepdim=True)

        self.logger.experiment.add_image(f'image/{self.name}', grid, self.current_epoch)
        self.logger.experiment.add_scalar(f'avg_val_loss/{self.name}', val_loss, self.current_epoch)
        self.logger.experiment.add_scalar(f'avg_val_BCE/{self.name}', val_BCE, self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        output = self.step(batch)
        return {"loss": output[0], "BCE": output[-1]}

    def validation_step(self, batch, batch_idx):
        output = self.step(batch)
        return {"val_loss": output[0], "BCE": output[-1]}


class VAE(Base):
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, 784), reduction='none').view(
            (self.num_samples, -1, 784)).mean(0).sum(-1).mean()
        KLD = -0.5 * torch.mean((1 + logvar - mu.pow(2) - logvar.exp()).view(
            (self.num_samples, -1, self.hidden_dim)).mean(0).sum(-1))
        loss = BCE + KLD
        return loss, BCE

    def step(self, batch):
        x, _ = batch
        mu, logvar = self.encode(x.view(-1, 784))
        mu = mu.repeat(self.num_samples, 1)
        logvar = logvar.repeat(self.num_samples, 1)
        z = self.reparameterize(mu, logvar)
        x_hat = self(z)
        loss, BCE = self.loss_function(x_hat, x.repeat(self.num_samples, 1, 1, 1), mu, logvar)
        return loss, x_hat, z, BCE


class IWAE(Base):
    def loss_function(self, recon_x, x, mu, logvar, z):
        log_Q = torch.distributions.Normal(loc=mu,
                                           scale=torch.exp(0.5 * logvar)).log_prob(z).view(
            (self.num_samples, -1, self.hidden_dim)).sum(-1)

        log_Pr = torch.sum((-0.5 * z ** 2).view((self.num_samples, -1, self.hidden_dim)), -1)
        BCE = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, 784), reduction='none').view(
            (self.num_samples, -1, 784)).sum(-1)
        log_weight = log_Pr - BCE - log_Q
        log_weight = log_weight - torch.max(log_weight, 0)[0]  # for stability
        weight = torch.exp(log_weight)
        weight = weight / torch.sum(weight, 0)
        weight = weight.detach()
        loss = torch.mean(torch.sum(weight * (-log_Pr + BCE + log_Q), 0))

        return loss, torch.sum(BCE * weight, dim=0).mean()

    def step(self, batch):
        x, _ = batch
        mu, logvar = self.encode(x.view(-1, 784))
        mu = mu.repeat(self.num_samples, 1)
        logvar = logvar.repeat(self.num_samples, 1)
        z = self.reparameterize(mu, logvar)
        x_hat = self(z)
        loss, BCE = self.loss_function(x_hat, x.repeat(self.num_samples, 1, 1, 1), mu, logvar, z)
        return loss, x_hat, z, BCE
