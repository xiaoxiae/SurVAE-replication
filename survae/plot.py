import numpy as np
from matplotlib import pyplot as plt

from survae import Dataset
from survae.layer import AbsoluteUnit
from survae.train import TrainedModels

SAMPLE_COUNT = 50_000
BINS = 100


def plot_models(models: TrainedModels, datasets: list[Dataset], sample_count=SAMPLE_COUNT, bins=BINS):
    """Plot the results for trained models on the given datasets."""
    model_count = max([j for _, j in models]) + 1

    # Create subplots (+1 for the raw data)
    fig, axs = plt.subplots(len(datasets), model_count + 1, figsize=(15, 10))

    # Plot the raw data
    for i, dataset in enumerate(datasets):
        X = dataset.sample(sample_count).cpu().numpy()

        axs[i, 0].hist2d(X[:, 0], X[:, 1], bins=bins)
        axs[i, 0].set_title(f'Data / {dataset.get_name()}')

    # Iterate over datasets and create heatmaps
    for i, dataset in enumerate(datasets):
        for j in range(model_count):
            model = models[i, j][0]  # [0] is the model, [1] is parameters/losses for log epochs

            X = model.sample(sample_count).cpu().numpy()

            axs[i, j + 1].hist2d(X[:, 0], X[:, 1], bins=bins)
            axs[i, j + 1].set_title(f'{model.get_name()} / {dataset.get_name()}')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_conditional_models(models: TrainedModels, datasets: list[Dataset], sample_count=SAMPLE_COUNT):
    """Plot the results for trained conditionally-trained models on the given datasets."""
    model_count = max([j for _, j in models]) + 1

    # Create subplots (+1 for the raw data)
    fig, axs = plt.subplots(len(datasets), model_count + 1, figsize=(15, 10))

    # Plot the raw data
    for i, dataset in enumerate(datasets):
        X, y = dataset.sample(sample_count, labels=True)
        X.cpu().numpy()
        y.cpu().numpy()

        unique_labels = np.unique(y)

        for label in unique_labels:
            label_indices = np.where(y == label)[0]
            axs[i, 0].scatter(
                X[label_indices, 0], X[label_indices, 1],
                label=f'Label {label}', alpha=100 / sample_count
            )

        axs[i, 0].set_title(f'Ground Truth / {dataset.get_name()}')

    for i, dataset in enumerate(datasets):
        for j in range(model_count):
            model = models[i, j][0]  # [0] is the model, [1] is parameters/losses for log epochs

            for label in range(model.condition_size):
                sampled_points = model.sample(sample_count // model.condition_size, condition=label).cpu().numpy()
                axs[i, j + 1].scatter(
                    sampled_points[:, 0], sampled_points[:, 1],
                    label=f'Label {label}', alpha=100 / sample_count
                )

            axs[i, j + 1].set_title(f'{model.get_name()} / {dataset.get_name()}')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_losses(models: TrainedModels, datasets: list[Dataset]):
    """Plot losses for trained models on the given datasets."""
    dataset_count = len(datasets)
    model_count = max([j for _, j in models]) + 1

    # Create subplots (+1 for the raw data)
    fig, axs = plt.subplots(dataset_count, model_count, figsize=(15, 10))

    # Iterate over datasets and create losses
    for i in range(dataset_count):
        for j in range(model_count):
            model, metrics = models[i, j]

            training_loss = [t.training_loss for t in metrics.values()]
            testing_loss = [t.testing_loss for t in metrics.values()]
            epoch_values = list(metrics.keys())

            axs[i, j].plot(epoch_values, training_loss, label='Training loss')
            axs[i, j].plot(epoch_values, testing_loss, label='Testing loss')
            axs[i, j].set_title(f'{model.get_name()} / {datasets[i].get_name()}')
            axs[i, j].set_xlabel('Epoch')
            axs[i, j].set_ylabel('Loss')
            axs[i, j].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_q_values(models: TrainedModels, datasets: list[Dataset]):
    """Plot Q-values of the first AbsoluteUnit layer for trained models on the given datasets."""
    model_count = max([j for _, j in models]) + 1

    fig, axs = plt.subplots(len(datasets), 1, figsize=(15, 10))

    for i, dataset in enumerate(datasets):
        # We're interested in the last model - that is the one that changes the q
        model, snapshots = models[i, model_count - 1]

        q1 = []
        q2 = []
        for epoch, snapshot in snapshots.items():
            model.load_state_dict(snapshot.model_state)

            for l in model.layers:
                if isinstance(l, AbsoluteUnit):
                    q1.append(l.q[0].item())
                    q2.append(l.q[1].item())

        epoch_values = list(snapshots.keys())

        axs[i].plot(epoch_values, q1, label="Q value for dimension 1")
        axs[i].plot(epoch_values, q2, label="Q value for dimension 2")
        axs[i].set_title(f'Q-values for {model.get_name()} / {dataset.get_name()}')
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel('Q-value')
        axs[i].set_ylim([0, 1])
        axs[i].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot(models: TrainedModels, datasets: list[Dataset]):
    """
    Just plot it.
    """
    # assume that if one is conditional, they all are
    conditional = list(models.values())[0][0].condition_size != 0

    if conditional:
        plot_conditional_models(models, datasets)
    else:
        plot_models(models, datasets)

    plot_losses(models, datasets)