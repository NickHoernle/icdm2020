import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

import torch
import fire
from torch.nn import functional as F

from reputation_study.plotting_utils import *

from reputation_study.data_loader import ElectorateDatasetAllActions, ReputationDatasetAllActions
from reputation_study.models import (
    ZeroInflatedPoisson_loss_function,
    ZIP_loss,
    Pois_loss,
    Baseline,
    AddBeta,
    AddClassPred, MultipleClassPred, SingleActivityFeed)


def criterion_mse(recon_x, x, latent_loss=0, log_cluster_pred=None, test=False):
    count_pred = F.softplus(recon_x[:, :, 1, :])
    log_theta = F.logsigmoid(recon_x[:, :, 0, :])

    predicted_val = log_theta.exp()*count_pred

    if type(log_cluster_pred) == type(None):
        mse_loss = F.mse_loss(predicted_val, x, reduction="none")
        return mse_loss.sum(dim=-1).mean()

    cluster_pred = log_cluster_pred.exp().unsqueeze(-1)
    mse_loss = F.mse_loss(predicted_val, x, reduction="none")
    return (cluster_pred*mse_loss).sum(dim=1).sum(dim=-1).mean()


class SharedSettings:
    root_dir = f"/Volumes/Seagate Backup Plus Drive/so_experiments"

    threshold_achievement = 70
    criterion = ZIP_loss
    epochs = 100

    log_files = "logs"
    figures = "figures"

    models = [Baseline, AddBeta, AddClassPred, MultipleClassPred]
    additional_params = [
        {},
        {},
        {"weights": [(0, 0, 0, 0), (0, 0, 1, -1), (1, -1, 0, 0), (1, -1, 1, -1)]},
        {"weights": [
            (0, 0, 0, 0, 0),
            (1, 0, 0, 1, -1), (1, 1, -1, 0, 0), (1, 1, -1, 1, -1),
            (2, 0, 0, 1, 0), (2, 1, 0, 0, 0), (2, 1, 0, 1, 0)
        ]}
    ]


class ElectorateStudy(SharedSettings):
    name = "electorate"
    dataset = ElectorateDatasetAllActions
    data_path = f"data/pt_{name}/"
    activity_indexes = [5]
    experiment_results = f"{SharedSettings.root_dir}/{name}"


class CivicDutyStudy(SharedSettings):
    name = "civicduty"
    dataset = ElectorateDatasetAllActions
    data_path = f"data/pt_{name}/"
    activity_indexes = [4,5]
    experiment_results = f"{SharedSettings.root_dir}/{name}"


class StrunkWhiteStudy(SharedSettings):
    name = "strunkwhite"
    dataset = ElectorateDatasetAllActions
    data_path = f"data/pt_{name}/"
    activity_indexes = [3]
    experiment_results = f"{SharedSettings.root_dir}/{name}"


class CopyEditorStudy(SharedSettings):
    name = "copyeditor"
    dataset = ElectorateDatasetAllActions
    data_path = f"data/pt_{name}/"
    activity_indexes = [3]
    experiment_results = f"{SharedSettings.root_dir}/{name}"


class ReputationStudySharedSettings(SharedSettings):
    data_dir = f"/Volumes/Seagate Backup Plus Drive/so_data"
    root_dir = f"/Volumes/Seagate Backup Plus Drive/so_reputation_experiments"

    threshold_achievement = 20

    additional_params = [
        {},
        {},
        {"weights": [(0, 0, 0, 0), (0, 0, 1, -1), (1, -1, 0, 0), (1, -1, 1, -1)]},
        {"weights": [
            (0, 0, 0, 0, 0),
            (1, 0, 0, 1, -1), (1, 1, -1, 0, 0), (1, 1, -1, 1, -1),
            (2, 0, 0, 0, 1), (2, 0, 1, 0, 0), (2, 0, 1, 0, 1),
            (3, 0, 0, 1, 0), (3, 1, 0, 0, 0), (3, 1, 0, 1, 0)
        ]}
    ]


class Reputation1000Study(ReputationStudySharedSettings):
    name = "reputation1000"
    threshold = 1000
    dataset = ReputationDatasetAllActions
    data_path = f"{ReputationStudySharedSettings.data_dir}/{threshold}/reputation_data"
    activity_indexes = [0,1,2]
    experiment_results = f"{ReputationStudySharedSettings.root_dir}/{name}"


class Reputation2000Study(ReputationStudySharedSettings):
    name = "reputation2000"
    threshold = 2000
    dataset = ReputationDatasetAllActions
    data_path = f"{ReputationStudySharedSettings.data_dir}/{threshold}/reputation_data"
    activity_indexes = [0,1,2]
    experiment_results = f"{ReputationStudySharedSettings.root_dir}/{name}"


class Reputation20000Study(ReputationStudySharedSettings):
    name = "reputation20000"
    threshold = 20000
    dataset = ReputationDatasetAllActions
    data_path = f"{ReputationStudySharedSettings.data_dir}/{threshold}/reputation_data"
    activity_indexes = [0,1,2]
    experiment_results = f"{ReputationStudySharedSettings.root_dir}/{name}"


class Reputation25000Study(ReputationStudySharedSettings):
    name = "reputation25000"
    threshold = 25000
    dataset = ReputationDatasetAllActions
    data_path = f"{ReputationStudySharedSettings.data_dir}/{threshold}/reputation_data"
    activity_indexes = [0,1,2]
    experiment_results = f"{ReputationStudySharedSettings.root_dir}/{name}"


experiment_options = {
    "electorate": ElectorateStudy,
    "civicduty": CivicDutyStudy,
    "strunkwhite": StrunkWhiteStudy,
    "copyeditor": CopyEditorStudy,
    "reputation1000": Reputation1000Study,
    "reputation2000": Reputation2000Study,
    "reputation20000": Reputation20000Study,
    "reputation25000": Reputation25000Study,
}


def run_experiment(on: str = "electorate", resume: bool = False):

    settings = experiment_options[on.lower()]

    model_params = {
        'latent_dim': 10,
        'date_of_threshold_cross': settings.threshold_achievement,
        'input_lim': 10,
        'output_len': settings.threshold_achievement * 2,
        'in_channels': 1,
        'block': True
    }

    loader_params = {
        'batch_size': 100,
        'shuffle': True,
        'num_workers': 6
    }

    training_set = settings.dataset(
        data_path=settings.data_path,
        dset_type="train",
        threshold_achievement=settings.threshold_achievement
    )

    validation_set = settings.dataset(
        data_path=settings.data_path,
        dset_type="validate",
        threshold_achievement=settings.threshold_achievement
    )

    testing_set = settings.dataset(
        data_path=settings.data_path,
        dset_type="test",
        threshold_achievement=settings.threshold_achievement
    )

    all_set = settings.dataset(
        data_path=settings.data_path,
        dset_type="all",
        threshold_achievement=settings.threshold_achievement,
        subsample=False,
    )

    train_loader = torch.utils.data.DataLoader(training_set, **loader_params)
    val_loader = torch.utils.data.DataLoader(validation_set, **loader_params)
    test_loader = torch.utils.data.DataLoader(testing_set, **loader_params)
    all_loader = torch.utils.data.DataLoader(all_set, **loader_params)

    criterion = settings.criterion

    exp_results = settings.experiment_results
    figures = os.path.join(exp_results, settings.figures)
    logs = os.path.join(exp_results, settings.log_files)

    for folder in [exp_results, figures, logs]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    all_results_file = open(os.path.join(settings.root_dir, "results.txt"), "a", buffering=1)

    for modelClass, additional_params in zip(settings.models, settings.additional_params):

        model = modelClass(**{**model_params, **additional_params})

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        logfile_name = f"{settings.name}_{model.name}.txt"
        checkpoint_name = f"{settings.name}_{model.name}"

        checkpoint_file = os.path.join(logs, checkpoint_name)

        best_prec1 = np.infty
        start_epoch = 0
        if resume:
            experiment_logfile = open(os.path.join(logs, logfile_name), "a", buffering=1)
            c_file = f"{checkpoint_file}.checkpoint.pth.tar"
            if os.path.isfile(c_file):
                print("=> loading checkpoint '{}'".format(c_file))
                checkpoint = torch.load(c_file)
                start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                print(f"=> loaded checkpoint '{c_file}' (epoch {c_file})")
            else:
                print(f"=> no checkpoint found at '{c_file}'")
        else:
            experiment_logfile = open(os.path.join(logs, logfile_name), "w", buffering=1)

        for epoch in range(start_epoch, settings.epochs):
            # train for one epoch
            train(train_loader, model, criterion, optimizer, scheduler, epoch, settings, experiment_logfile)

            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion, epoch, settings, experiment_logfile)

            # remember best prec@1 and save checkpoint
            is_best = prec1 < best_prec1
            best_prec1 = min(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, checkpoint_file)

        experiment_logfile.write(f'Best accuracy: {best_prec1}\n')

        checkpoint = torch.load(f"{checkpoint_file}.best.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])
        test_pred_elbo = validate(test_loader, model, criterion, 0, settings, experiment_logfile)
        test_pred_mse = validate(test_loader, model, criterion_mse, 0, settings, experiment_logfile)
        experiment_logfile.write(f'Test accuracy ====> {test_pred_elbo}\n')
        all_results_file.write(f"Setting: {settings.name}\t "
                               f"Model: {model.name}; \t "
                               f"Test ELBO: {round(test_pred_elbo, 3)}; \t "
                               f"Test MSE: {round(test_pred_mse, 3)}\n")

        experiment_logfile.flush()
        all_results_file.flush()

        plot_inference_images(model, figures, all_loader, settings)

        experiment_logfile.close()
    all_results_file.close()


def train(train_loader, model, criterion, optimizer, scheduler, epoch, settings, logfile):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, all_activities in enumerate(train_loader):

        activities = (torch.stack([a for i, a in enumerate(all_activities) if i in settings.activity_indexes], dim=1)
                      .sum(dim=1).unsqueeze(1).float().to("cpu"))

        if model.name in ["Model0", "Model1"]:
            (ts_pred), (latent_loss, z) = model(activities)
            cluster_preds = None
        else:
            (ts_pred, c_pred), (latent_loss, z) = model(activities)
            cluster_preds = c_pred.log_softmax(dim=1)

        loss = criterion(
            ts_pred,
            activities[:, :, 1:],
            latent_loss,
            log_cluster_pred=cluster_preds
        )

        losses.update(loss.data.item(), ts_pred.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    logfile.write(f'Train ({epoch}): Loss {round(losses.avg, 3)}\n')
    scheduler.step()


def validate(val_loader, model, criterion, epoch, settings, logfile):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.eval()

    end = time.time()

    for i, all_activities in enumerate(val_loader):
        activities = (torch.stack([a for i, a in enumerate(all_activities) if i in settings.activity_indexes], dim=1)
                      .sum(dim=1).unsqueeze(1).float().to("cpu"))

        if model.name in ["Model0", "Model1"]:
            (ts_pred), (latent_loss, z) = model(activities)
            cluster_preds = None
        else:
            (ts_pred, c_pred), (latent_loss, z) = model(activities)
            cluster_preds = c_pred.log_softmax(dim=1)

        loss = criterion(
            ts_pred,
            activities[:, :, 1:],
            latent_loss,
            log_cluster_pred=cluster_preds,
            test=True
        )

        losses.update(loss.data.item(), ts_pred.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    logfile.write(f'Valid ({epoch}): Loss {round(losses.avg, 3)}\n')
    return losses.avg


def plot_inference_images(model, figures_directory, all_loader, settings):
    plot_raw_mean_plot_of_activities(figures_directory, all_loader, settings)

    # model specific computation
    users_all, predictions_all, actions_all, every_type_of_action = [], [], [], []

    for i, all_activities in enumerate(all_loader):
        activities = (torch.stack([a for i, a in enumerate(all_activities) if i in settings.activity_indexes], dim=1)
                      .sum(dim=1).unsqueeze(1).float().to("cpu"))

        if model.name in ["Model0", "Model1"]:
            (ts_pred), (latent_loss, z) = model(activities)
            cluster_preds = torch.zeros(ts_pred.size(0), 1).detach().numpy()
        else:
            (ts_pred, c_pred), (latent_loss, z) = model(activities)
            cluster_preds = c_pred.log_softmax(dim=1).detach().numpy()

        users_all.append(cluster_preds)
        predictions_all.append(ts_pred.detach().numpy())
        actions_all.append(activities.squeeze(1).detach().numpy())

    users_all = np.concatenate(users_all, axis=0)
    predictions_all = np.concatenate(predictions_all, axis=0)
    actions_all = np.concatenate(actions_all, axis=0)
    weights = None

    if model.name in ["Model2", "Model3"]:
        weights = model.weights
        plot_model_cluster_assignments((users_all, weights), figures_directory, model, settings)
        plot_cluster_probabilities((users_all, weights), figures_directory, model, settings)
        plot_raw_mean_plot_of_activities_per_cluster((users_all, predictions_all, actions_all, weights),
                                                   figures_directory, model, settings)
        plot_model_inferred_beta((users_all, predictions_all, actions_all, weights), figures_directory, model, settings)

    plot_samples_of_reconstructed_trajectories((users_all, predictions_all, actions_all, weights), figures_directory, model, settings)


def save_checkpoint(state, is_best, fname):
    """Saves checkpoint to disk"""
    torch.save(state, f"{fname}.checkpoint.pth.tar")
    if is_best:
        shutil.copyfile(f"{fname}.checkpoint.pth.tar", f"{fname}.best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    fire.Fire(run_experiment)
