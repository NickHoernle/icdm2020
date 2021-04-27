import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter

plotting_params = {
    "lw": 3,
}


sigmoid = lambda x: 1/(1 + np.exp(-x))
softplus = lambda x: np.log((1.0 + np.exp(x)))

tick_label_size=28
text_size=30


def plot_raw_mean_plot_of_activities(figures_directory, loader, settings):
    fig = plt.figure(figsize=(15, 5))
    ax = fig.gca()
    all_actions = []
    for i, actions in enumerate(loader):
        all_actions.append(np.stack([a.numpy() for a in actions], axis=1))
    actions = np.concatenate(all_actions, axis=0).mean(axis=0)
    colors = ["C0", "C1", "C2", "C3"]
    counter = 0
    for i, l in enumerate(loader.dataset.activity_names):
        if i in settings.activity_indexes:
            ax.plot(actions[i], label=l, c=colors[counter], lw=plotting_params["lw"])
            counter += 1

    ax.axvline(x=settings.threshold_achievement, lw=2, ls='--', color='black')
    ax.axhline(y=0, lw=2, ls='--', color='black')

    start_week = settings.threshold_achievement // 7

    if settings.threshold_achievement > 25:
        ax.set_xticks(np.arange(0, (2*start_week+1) * 7, 7))
        ax.set_xticklabels(np.arange(-start_week * 7, (start_week+1) * 7, 7), fontsize=tick_label_size)
    else:
        start_week = settings.threshold_achievement // 5
        ax.set_xticks(np.arange(0, (2 * start_week + 1) * 5, 5))
        ax.set_xticklabels(np.arange(-start_week * 5, (start_week + 1) * 5, 5), fontsize=tick_label_size)

    x_lab = 'Days before/after threshold'
    if "reputation" in settings.name:
        x_lab = 'Weeks before/after threshold'
    ax.set_xlabel(x_lab, fontsize=text_size)
    ax.set_ylabel('Mean # Actions', fontsize=text_size)

    ax.legend(loc='best', fontsize=text_size)

    ax.tick_params(axis='both', which='major', labelsize=tick_label_size)

    fig_file = os.path.join(figures_directory, f"{settings.name}_activity_mean_plot.eps")

    plt.tight_layout()
    plt.savefig(fig_file, format='eps', dpi=1200)
    plt.savefig(fig_file.replace("eps", "png"))
    plt.close()


def plot_raw_mean_plot_of_activities_per_cluster(data, figures_directory, model, settings):
    (users, predictions, actions, weights) = data

    fig = plt.figure(figsize=(15, 5))
    ax = fig.gca()

    colors = ["C0", "C1", "C2", "C3"]
    lws = [2, 1, 1, 2]
    ls = ["-", "--", "--", "-"]

    for i in range(4):
        user_assignment = users.argmax(axis=1) == i
        if model.name == "Model3":
            user_assign = users.argmax(axis=1)
            user_assignment = np.array([weights[u][0] for u in user_assign]) == i

            acts = actions[user_assignment]
            lw, ls = 2, "-"
            if np.sum(user_assignment) / len(users) < .05:
                lw, ls = 1, "--"
            ax.plot(acts.mean(axis=0), lw=lw, ls=ls, color=colors[i], label=f"Group {i}")

        else:
            acts = actions[user_assignment]
            name = {
                (0, 0, 0, 0): "Non-Steerers",
                (1, -1, 0, 0): "Dropouts",
                (0, 0, 1, -1): "Strong-and-Steady",
                (1, -1, 1, -1): "Strong-Steerers"
            }
            lw, ls = 2, "-"
            if np.sum(user_assignment) / len(users) < .05:
                lw, ls = 1, "--"
            if np.sum(user_assignment) / len(users) > .01:
                ax.plot(acts.mean(axis=0), lw=lw, ls=ls, color=colors[i], label=name[weights[i]])

    ax.axvline(x=settings.threshold_achievement, lw=2, ls='--', color='black')
    ax.axhline(y=0, lw=2, ls='--', color='black')

    start_week = settings.threshold_achievement // 7
    if settings.threshold_achievement > 25:
        ax.set_xticks(np.arange(0, (2 * start_week + 1) * 7, 7))
        ax.set_xticklabels(np.arange(-start_week * 7, (start_week + 1) * 7, 7), fontsize=tick_label_size)
    else:
        start_week = settings.threshold_achievement // 5
        ax.set_xticks(np.arange(0, (2 * start_week + 1) * 5, 5))
        ax.set_xticklabels(np.arange(-start_week * 5, (start_week + 1) * 5, 5), fontsize=tick_label_size)

    x_lab = 'Days before/after threshold'
    if "reputation" in settings.name:
        x_lab = 'Weeks before/after threshold'
    ax.set_xlabel(x_lab, fontsize=text_size)
    ax.set_ylabel('Mean # Actions', fontsize=text_size)

    ax.legend(loc='best', fontsize=text_size)

    ax.tick_params(axis='both', which='major', labelsize=tick_label_size)

    fig_file = os.path.join(figures_directory, f"{model.name}_activity_mean_plot_{settings.name}.eps")

    plt.tight_layout()
    plt.savefig(fig_file, format='eps', dpi=1200)
    plt.savefig(fig_file.replace("eps", "png"))
    plt.close()


def plot_model_cluster_assignments(data, figures_directory, model, settings):
    users, weights = data

    bins = list(range(users.shape[1] + 1))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()

    print(settings.name, model.name)
    cnt = Counter(users.argmax(axis=1))
    print(cnt)
    sum_ = sum([v for k,v in cnt.items()])
    print({k: float(v)/sum_ for k,v in cnt.items()})

    if model.name == "Model2":
        plt.hist(users.argmax(axis=1), bins=bins, color="C0")
        name = {
            (0, 0, 0, 0): "Non-Steerers",
            (1, -1, 0, 0): "Dropouts",
            (0, 0, 1, -1): "Strong-and-Steady",
            (1, -1, 1, -1): "Strong-Steerers"
        }
        plt.xticks(np.arange(len(weights)) + .5, [name[w] for w in weights], rotation=-60)
    else:
        assignments = np.zeros_like(users.argmax(axis=1))
        for i in range(3):
            user_assign = users.argmax(axis=1)
            user_assignment = np.array([weights[u][0] for u in user_assign]) == i
            assignments[user_assign] = i

        plt.hist(assignments, bins=bins, color="C0")
        # ax.plot(acts.mean(axis=0), lw=lw, ls=ls, color=colors[i], label=f"Group {i}")
        labels = ["Group 0", "Group 1", "Group 2"]
        plt.xticks(np.arange(len(labels)) + .5, labels, rotation=-60)

    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_ylabel("Count", fontsize=text_size)

    fig_file = os.path.join(figures_directory, f"{model.name}_cluster_assignments_{settings.name}.eps")

    plt.tight_layout()
    plt.savefig(fig_file, format='eps', dpi=1200)
    plt.savefig(fig_file.replace("eps", "png"))
    plt.close()


def plot_cluster_probabilities(data, figures_directory, model, settings):
    users, weights = data

    fig, axes = plt.subplots((3 + len(weights)) // 4, 4, figsize=(15, 3 * ((3 + len(weights)) // 4)))
    axes = axes.flatten()
    for i in range(len(weights)):
        axes[i].hist(np.exp(users)[:, i], bins=np.arange(0, 1.1, .1), color="C0")
        axes[i].set_xlim([0, 1])
        axes[i].set_title(weights[i])

        axes[i].tick_params(axis='both', which='major', labelsize=20)

    fig_file = os.path.join(figures_directory, f"{model.name}_cluster_assignment_probs_{settings.name}.eps")

    plt.tight_layout()
    plt.savefig(fig_file, format='eps', dpi=1200)
    plt.savefig(fig_file.replace("eps", "png"))
    plt.close()


def plot_model_weights(data, figures_directory, model, settings):
    pass


def plot_model_inferred_beta(data, figures_directory, model, settings):

    bin_weights, act_weights = model.get_weight_options()

    if model.name == "Model3":
        rows = 3 if bin_weights.shape[0] >= 9 else 2
        fig, axes = plt.subplots(rows, 1, figsize=(15, rows * 5))
        for i in range(rows):
            # axes[i].plot(model.apply_window_fn(bin_weights[i*3+3]).detach().numpy())
            # axes[i].plot(model.apply_window_fn(act_weights[i*3+3]).detach().numpy())
            axes[i].plot((bin_weights[i * 3 + 3]).detach().numpy())
            axes[i].plot((act_weights[i * 3 + 3]).detach().numpy())
            axes[i].set_ylim([-4, 10])

            ax = axes[i]

            ax.axvline(x=settings.threshold_achievement, lw=2, ls='--', color='black')
            ax.axhline(y=0, lw=2, ls='--', color='black')

            start_week = settings.threshold_achievement // 7
            ax.set_xticks(np.arange(0, (2 * start_week + 1) * 7, 7))
            ax.set_xticklabels(np.arange(-start_week * 7, (start_week + 1) * 7, 7), fontsize=tick_label_size)
            ax.set_ylabel('Strength of Steering', fontsize=text_size)

        ax = axes[-1]
    else:
        rows = 1
        fig, axes = plt.subplots(rows, 1, figsize=(15, rows * 5))

        # axes.plot(model.apply_window_fn(bin_weights[-1]).detach().numpy())
        # axes.plot(model.apply_window_fn(act_weights[-1]).detach().numpy())
        axes.plot((bin_weights[-1]).detach().numpy(), label="$\\beta_\\alpha$")
        axes.plot((act_weights[-1]).detach().numpy(), label="$\\beta_\\lambda$")
        axes.set_ylim([-4, 10])
        ax = axes

        ax.axvline(x=settings.threshold_achievement, lw=2, ls='--', color='black')
        ax.axhline(y=0, lw=2, ls='--', color='black')

        start_week = settings.threshold_achievement // 7
        ax.set_xticks(np.arange(0, (2 * start_week + 1) * 7, 7))
        ax.set_xticklabels(np.arange(-start_week * 7, (start_week + 1) * 7, 7), fontsize=tick_label_size)

        ax.set_ylabel('Strength of Steering', fontsize=text_size)

    x_lab = 'Days before/after threshold'
    if "reputation" in settings.name:
        x_lab = 'Weeks before/after threshold'
    ax.set_xlabel(x_lab, fontsize=text_size)

    ax.legend(loc='best', fontsize=text_size)

    ax.tick_params(axis='both', which='major', labelsize=tick_label_size)

    fig_file = os.path.join(figures_directory, f"{model.name}_plot_of_beta_{settings.name}.eps")

    plt.tight_layout()
    plt.savefig(fig_file, format='eps', dpi=1200)
    plt.savefig(fig_file.replace("eps", "png"))
    plt.close()


def plot_samples_of_reconstructed_trajectories(data, figures_directory, model, settings):
    (users, predictions, actions, weights) = data

    if model.name in ["Model0", "Model1"]:
        columns = 1
        legend = [9]
    else:
        columns = 4
        legend = [9, 2]

    fig, axes = plt.subplots(10, columns, figsize=(5*columns, 15))

    for column in range(columns):

        user_assignment = users.argmax(axis=1) == column
        if model.name == "Model3":
            user_assign = users.argmax(axis=1)
            user_assignment = np.array([weights[u][0] for u in user_assign]) == column

        preds = predictions[user_assignment]
        acts = actions[user_assignment]
        uzers = users[user_assignment]
        num_users = len(preds)
        num_plots = np.min([num_users, 10])

        for ix, i in enumerate(np.random.choice(np.arange(num_users), size=num_plots, replace=False)):

            if columns == 1:
                ax = axes[ix]
            else:
                ax = axes[ix, column]
                if ix == 0:
                    if model.name == "Model3":
                        ax.set_title(column)
                    else:
                        name = {
                            (0, 0, 0, 0): "Non-Steerers",
                            (1, -1, 0, 0): "Dropouts",
                            (0, 0, 1, -1): "Strong-and-Steady",
                            (1, -1, 1, -1): "Strong-Steerers"
                        }
                        ax.set_title(name[weights[column]], fontsize=text_size)

            binary_pred = sigmoid(preds[i, uzers[i].argmax(), 0, :])
            count_pred = softplus(preds[i, uzers[i].argmax(), 1, :])

            ax.plot(acts[i][1:], label='True #Actions', lw=1, color="C1")
            ax.plot((binary_pred * count_pred), label='Predicted #Actions', lw=.75, color="C0", ls="--")

            ax.set_ylim(0, 20)
            # ax.legend(loc='best')

            ax.axvline(x=settings.threshold_achievement, lw=.5, ls='--', color='black')
            ax.axhline(y=0, lw=2, ls='--', color='black')

            start_week = settings.threshold_achievement // 7
            if ix == num_plots-1:
                ax.set_xticks(np.arange(10, 121, 20))
                # ax.set_xticklabels(np.arange(-start_week * 7, (start_week + 1) * 7, 20), fontsize=22)
                ax.set_xticklabels(np.arange(-60, 60, 20), fontsize=22)
                x_lab = '                   Days before/after threshold'
                if "reputation" in settings.name:
                    x_lab = '                   Weeks before/after threshold'
                if column == 1:
                    ax.set_xlabel(x_lab, fontsize=tick_label_size)
            else:
                ax.get_xaxis().set_visible(False)

            # ax.set_ylabel('Mean # Actions', fontsize=22)
            # ax.legend(loc='best', fontsize=22)

            ax.tick_params(axis='both', which='major', labelsize=22)

    if columns == 4:
        handles, labels = axes[-1, -1].get_legend_handles_labels()
        fig.legend(handles, labels, loc=(.2, 0.01), ncol=2, fontsize=text_size)

    fig_file = os.path.join(figures_directory, f"{model.name}_sampled_trajectories_{settings.name}.eps")
    plt.tight_layout()

    fig.subplots_adjust(bottom=0.125)
    plt.savefig(fig_file, format='eps', dpi=1200)
    plt.savefig(fig_file.replace("eps", "png"))
    plt.close()

