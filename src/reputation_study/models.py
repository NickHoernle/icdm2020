import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List
from torch.distributions import Poisson
import torch.distributions as distrib
import itertools

import numpy as np

from so_study.normalizing_flows import NormalizingFlow


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def ZeroInflatedPoisson_loss_function(recon_x, x, latent_loss):
    x_shape = x.size()

    recon_x_0_bin = torch.sigmoid(recon_x[:, 0, :])
    recon_x_0_count = F.softplus(recon_x[:, 1, :])

    poisson_0 = (x == 0).float() * Poisson(recon_x_0_count).log_prob(x)
    # else if x > 0
    poisson_greater0 = (x > 0).float() * Poisson(recon_x_0_count).log_prob(x)

    zero_inf = torch.cat((
        torch.log((1 - recon_x_0_bin) + 1e-9).view(x_shape[0], x_shape[1], -1),
        poisson_0.view(x_shape[0], x_shape[1], -1)
    ), dim=2)

    log_l = (x == 0).float() * torch.logsumexp(zero_inf, dim=2)
    log_l += (x > 0).float() * (torch.log(recon_x_0_bin + 1e-9) + poisson_greater0)

    return -log_l.sum() + latent_loss

def ZeroInflatedPoisson_loss_function_M2(recon_x, x, latent_loss, cluster_pred):
    x_shape = x.size()

    recon_x_0_bin = torch.sigmoid(recon_x[:, 0, :])
    recon_x_0_count = F.softplus(recon_x[:, 1, :])

    poisson_0 = (x == 0).float() * Poisson(recon_x_0_count).log_prob(x)
    # else if x > 0
    poisson_greater0 = (x > 0).float() * Poisson(recon_x_0_count).log_prob(x)

    zero_inf = torch.cat((
        torch.log((1 - recon_x_0_bin) + 1e-9).view(x_shape[0], x_shape[1], -1),
        poisson_0.view(x_shape[0], x_shape[1], -1)
    ), dim=2)

    log_l = (x == 0).float() * torch.logsumexp(zero_inf, dim=2)
    log_l += (x > 0).float() * (torch.log(recon_x_0_bin + 1e-9) + poisson_greater0)

    return -(cluster_pred.exp()*(log_l.sum(dim=1))).sum() + (cluster_pred.exp()*cluster_pred).sum() + latent_loss


class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(self,
                 nin: int,
                 hidden_sizes: List[int],
                 latent_dim: int,
                 nout: int,
                 num_masks: int = 1,
                 natural_ordering: bool = False):
        # thanks: https://github.com/karpathy/pytorch-made/blob/master/made.py
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """
        super().__init__()

        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        self.latent_dim = latent_dim
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"

        # define a simple MLP neural net
        self.net = []
        hs = [nin + self.latent_dim] + hidden_sizes + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                MaskedLinear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0  # for cycling through num_masks orderings

        self.m = {}
        self.update_masks()  # builds the initial self.m connectivity

    def update_masks(self):
        if self.m and self.num_masks == 1: return  # only a single seed, skip for efficiency
        hs = [self.nin] + self.hidden_sizes + [self.nout]
        L = len(self.hidden_sizes)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
        for l in range(L):
            self.m[l] = rng.randint(self.m[l - 1].min(), self.nin - 1, size=self.hidden_sizes[l])

        # construct the mask matrices
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        masks[0] = np.concatenate([masks[0], np.ones((self.latent_dim, hs[1]))], axis=0)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        return self.net(x)


class Baseline(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.latent_dim = kwargs.get('latent_dim', 20)
        self.date_of_threshold_cross = kwargs.get('date_of_threshold_cross', 20)
        self.input_lim = kwargs.get('input_lim', 20)
        self.output_len = kwargs.get('output_len', 51)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_lim, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, self.latent_dim * 2)
        )

        hidden_size = 5

        # self.decoder = MADE(51, [200, 200], self.latent_dim, 51 * 2, natural_ordering=True)
        self.decoder = nn.GRU(input_size=1, hidden_size=self.latent_dim, num_layers=1, batch_first=True, dropout=0)
        self.fc_decoder = nn.Sequential(nn.ReLU(True), nn.Linear(self.latent_dim, 2))

        self.apply(init_weights)

    def latent_loss(self, x, z_params):
        n_batch = x.size(0)

        # Retrieve mean and var
        mu, log_var = z_params

        sigma = torch.exp(0.5 * log_var)

        # Re-parametrize
        zeros = torch.zeros_like(mu[0])
        ones = torch.ones_like(mu[0])

        q = distrib.Normal(zeros, ones)
        z = (sigma * q.sample((n_batch,))) + mu

        # Compute KL divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return z, kl_div

    def encode(self, x):
        encoded = self.encoder(x[:, :self.input_lim])
        return encoded[:, :self.latent_dim], encoded[:, self.latent_dim:]

    def decode(self, x, z):
        hidden_layer = z.unsqueeze(0)
        predictions = []
        for i in range(0, x.size(1) - 1):
            preds, hidden_layer_ = self.decoder(x[:, i].unsqueeze(-1).unsqueeze(-1), hidden_layer)
            predictions.append(self.fc_decoder(preds.squeeze(1)))

        return torch.stack(predictions, dim=-1)

    def forward(self, x):
        zparams = self.encode(x)
        z, latent_loss = self.latent_loss(x, zparams)
        predictions = self.decode(x, z)
        return predictions, (latent_loss, z)


class AddBeta(Baseline):
    def __init__(self, **kwargs):
        super(AddBeta, self).__init__(**kwargs)

        self.k_binomial = nn.Parameter(torch.randn(self.output_len, requires_grad=True).float())
        self.k_activity = nn.Parameter(torch.randn(self.output_len, requires_grad=True).float())

        # self.decoder = MADE(51, [200, 200], 0, 51 * 2, natural_ordering=True)

        self.softplus = nn.Softplus()
        self.apply(init_weights)

    @property
    def k_bin_pos(self):
        return self.softplus(self.k_binomial)
        # return self.k_binomial

    @property
    def k_act_pos(self):
        return self.softplus(self.k_activity)
        # return self.k_activity

    def get_weights(self, x):

        # effect_weight = 1/(1+torch.abs((self.output_len/2 - torch.arange(len(self.k_binomial)))))
        effect_weight = torch.ones(self.output_len)
        effect_weight[self.date_of_threshold_cross:] *= -1

        bin_weights = (effect_weight*self.k_bin_pos).unsqueeze(0).repeat(len(x), 1)
        act_weights = (effect_weight*self.k_act_pos).unsqueeze(0).repeat(len(x), 1)

        return bin_weights, act_weights

    def decode(self, x, z):
        activity = super().decode(x, z)
        bin_weights, act_weights = self.get_weights(x)

        activity[:, 0, :] += bin_weights
        activity[:, 1, :] += act_weights

        return activity


class AddNormalizingFlow(Baseline):
    def __init__(self, **kwargs):
        super(AddNormalizingFlow, self).__init__(**kwargs)

        self.K = 6

        self.device = "cpu"
        self.encoder_dims = 100

        self.encoder = nn.Sequential(
            nn.Linear(self.input_lim, 100),
            nn.ReLU(True),
            nn.Linear(100, self.encoder_dims),
            nn.ReLU(True),
        )
        self.mu = nn.Linear(self.encoder_dims, self.latent_dim)
        self.log_var = nn.Linear(self.encoder_dims, self.latent_dim)
        self.flow_params = nn.Linear(self.encoder_dims, self.K * (self.latent_dim * 2 + 1))

        self.flow = NormalizingFlow(K=self.K, D=self.latent_dim)

        self.apply(init_weights)
        torch.nn.init.xavier_uniform_(self.flow_params.weight)

    def encode(self, x, **kwargs):
        h = self.encoder(x[:, :self.input_lim])
        return self.mu(h), self.log_var(h), self.flow_params(h)

    def latent_loss(self, x, z_params):
        n_batch = x.size(0)

        # Retrieve set of parameters
        mu, log_var, flow_params = z_params

        # Re-parametrize a Normal distribution
        q = distrib.Normal(torch.zeros(mu.shape[1]).to(self.device), torch.ones(log_var.shape[1]).to(self.device))

        sigma = torch.exp(0.5 * log_var)

        # Obtain our first set of latent points
        z_0 = (sigma * q.sample((n_batch,))) + mu

        # Complexify posterior with flows
        z_k, list_ladj = self.flow(z_0, flow_params.chunk(self.K, dim=1))

        # ln q(z_0)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # ladj = torch.cat(list_ladj)
        kl_div -= torch.sum(list_ladj)

        return z_k, kl_div


class AddBetaPersonal(AddBeta):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder = nn.GRU(input_size=1, hidden_size=self.latent_dim - 2, num_layers=1, batch_first=True, dropout=0)
        self.fc_decoder = nn.Sequential(nn.ReLU(True), nn.Linear(self.latent_dim - 2, 2))

    def decode(self, x, z):
        hidden_layer = z[:, :-2].unsqueeze(0)

        predictions = []
        for i in range(0, x.size(1) - 1):
            preds, hidden_layer_ = self.decoder(x[:, i].unsqueeze(-1).unsqueeze(-1), hidden_layer)
            predictions.append(self.fc_decoder(preds.squeeze(1)))

        activity = torch.stack(predictions, dim=-1)

        bin_weights, act_weights = self.get_weights(x)

        activity[:, 0] += torch.sigmoid(z[:, -1]).unsqueeze(-1) * bin_weights
        activity[:, 1] += torch.sigmoid(z[:, -2]).unsqueeze(-1) * act_weights

        return activity


class GeneralNFlow(AddBeta, AddNormalizingFlow):
    pass


class PersonalNFlow(AddBetaPersonal, AddNormalizingFlow):
    pass


class M2(AddBeta):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_clusters = 4
        self.encoder = nn.Sequential(
            nn.Linear(self.input_lim, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, self.latent_dim * 2 + self.num_clusters)
        )

        self.k_binomial = nn.Parameter(data=torch.randn(size=(4, self.output_len//2)))
        self.k_activity = nn.Parameter(data=torch.randn(size=(4, self.output_len//2)))

        self.proj_y = nn.Sequential(nn.Linear(self.num_clusters, self.latent_dim))

    def encode(self, x):
        enc = self.encoder(x[:, :self.input_lim])
        return enc[:, :self.latent_dim], \
               enc[:, self.latent_dim:-self.num_clusters], \
               enc[:, -self.num_clusters:]

    def get_weight_options(self):
        weights = [(0, 0), (0, 1), (1, 0), (1, -1)]
        bin_weights, act_weights = [], []
        for (b, a) in weights:
            bin_weight = torch.zeros(self.output_len)
            bin_weight[:self.output_len // 2] = b*(self.k_bin_pos[1] if b < 0 else self.k_bin_pos[0])
            bin_weight[self.output_len // 2:] = a*(self.k_bin_pos[3] if a < 0 else self.k_bin_pos[2])
            bin_weights.append(bin_weight)

            act_weight = torch.zeros(self.output_len)
            act_weight[:self.output_len // 2] = b * (self.k_act_pos[1] if b < 0 else self.k_act_pos[0])
            act_weight[self.output_len // 2:] = a * (self.k_act_pos[3] if a < 0 else self.k_act_pos[2])
            act_weights.append(act_weight)

        return torch.stack(bin_weights, dim=0), torch.stack(act_weights, dim=0), weights

    def get_weights(self, x):

        bin_weights, act_weights, weight_params = self.get_weight_options()

        num_users = x.size(0)

        bin_weights = bin_weights[(None,)].repeat(num_users, 1, 1)
        act_weights = act_weights[(None,)].repeat(num_users, 1, 1)

        return bin_weights, act_weights

    def decode(self, x, z):

        activities = Baseline.decode(self, x, z)
        bin_weights, act_weights = self.get_weights(x)

        num_clusters = bin_weights.size(1)

        weights = torch.stack((bin_weights, act_weights), dim=1)
        activities = activities.unsqueeze(2).repeat(1, 1, num_clusters, 1)

        activities += weights

        return activities

    def forward(self, x):
        mu, lv, cluster_pred = self.encode(x)
        z, latent_loss = self.latent_loss(x, (mu, lv))
        predictions = self.decode(x, z)
        return (predictions, cluster_pred), (latent_loss, z)
