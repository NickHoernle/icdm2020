import os
import time
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.backends import cudnn
from torch.nn import functional as F

def plot_kernel(dset, model, dset_shape, device, args, ax=None, title="", ylim=[-20, 20]):

    window_len = args.window_length

    if len(dset.__getitem__(0)) == 5:
        val_in, kernel_data, val_out, val_prox, badge_date = dset.__getitem__(0)
    else:
        val_in, kernel_data, val_out, val_prox, badge_date, _ = dset.__getitem__(0)

    val_in, kernel_data, val_out, val_prox, badge_date = val_in.reshape(-1, dset_shape[0], dset_shape[1]).to(device), \
                                                         kernel_data.reshape(-1, window_len * 2).to(device), \
                                                         val_out.reshape(-1, window_len * 2).to(device), \
                                                         val_prox.reshape(-1, dset_shape[0]).to(device), \
                                                         badge_date.reshape(-1, ).to(device)

    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))

    recon_batch, loss_params = model(val_in, kernel_data=kernel_data, dob=badge_date, prox_to_badge=val_prox)
    mu = model.get_z(val_in, kernel_data=kernel_data, dob=badge_date, prox_to_badge=val_prox)

    k = model.kernel(mu, val_in, kernel_data=kernel_data)
    for k_ in k:
        kern = k_.detach().numpy()
        kern[5 * 7 - 1] = kern[5 * 7 - 2]
        kern[5 * 7] = kern[5 * 7 - 2]
        ax.plot(np.arange(-5 * 7, 5 * 7), kern, lw=3, alpha=1, c='C1',
                label='$\\beta_1$ (Change in Activity Likelihood)')

    k = model.kernel_count(mu, val_in, kernel_data=kernel_data)
    for k_ in k:
        kern = k_.detach().numpy()
        kern[5 * 7 - 1] = kern[5 * 7 - 2]
        kern[5 * 7] = kern[5 * 7 - 2]
        ax.plot(np.arange(-5 * 7, 5 * 7), kern, lw=3, alpha=1, c='C0',
                label='$\\beta_2$ (Change in Expected Count)')

    ax.axvline(x=0, lw=2, ls='--', color='black')
    ax.axhline(y=0, lw=2, ls='--', color='black')
    ax.set_ylim(*ylim)
    title = model.__class__ if len(title) == 0 else title
    ax.set_title(title, fontsize=15)

    ax.set_xticks(np.arange(-5 * 7, 6 * 7, 7))
    ax.set_xticklabels(np.arange(-5 * 7, 6 * 7, 7), fontsize=20)

    ax.set_xlabel('Days before/after badge', fontsize=22)
    ax.set_ylabel('Parameter value', fontsize=22)

    ax.legend(loc='best', fontsize=22)

    ax.tick_params(axis='both', which='major', labelsize=20)
    #     ax.tick_params(axis='both', which='minor', labelsize=8)

    return ax

def ZeroInflatedPoisson_loss_function(recon_x, x, latent_loss):
    from torch.distributions import Poisson

    x_shape = x.size()
    # if x == 0
    recon_x_0_bin = recon_x[0]
    recon_x_0_count = recon_x[1]

    poisson_0 = (x==0).float()*Poisson(recon_x_0_count).log_prob(x)
    # else if x > 0
    poisson_greater0 = (x>0).float()*Poisson(recon_x_0_count).log_prob(x)

    zero_inf = torch.cat((
        torch.log((1-recon_x_0_bin)+1e-9).view(x_shape[0], x_shape[1], -1),
        poisson_0.view(x_shape[0], x_shape[1], -1)
    ), dim=2)

    log_l = (x==0).float()*torch.logsumexp(zero_inf, dim=2)
    log_l += (x>0).float()*(torch.log(recon_x_0_bin+1e-9)+poisson_greater0)

    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return -(log_l.sum()) + latent_loss

def calc_elbo(dset_loader, model, loss_fn, device):
    model.eval()
    validation_loss = 0
    reconst_loss = 0
    for val_in, kernel_data, val_out, val_prox, badge_date, _ in dset_loader:
        # Transfer to GPU
        val_in, kernel_data, val_out, val_prox, badge_date = val_in.to(device), kernel_data.to(device), val_out.to(
            device), val_prox.to(device), badge_date.to(device)
        recon_batch, latent_loss = model(val_in, kernel_data=kernel_data, dob=badge_date, prox_to_badge=val_prox)

    loss = loss_fn(recon_batch, val_out, latent_loss)
    validation_loss += loss.item()

    exp = (recon_batch[0] * recon_batch[1]).detach().numpy()
    #         print(exp)
    #         print((exp.detach()-val_out)**2)
    reconst_loss += np.mean((exp - val_out.detach().numpy()) ** 2)

    print(-validation_loss / len(dset_loader.dataset))
    print(reconst_loss / len(dset_loader.dataset))