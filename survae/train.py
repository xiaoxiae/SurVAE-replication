from typing import Callable

from tqdm import tqdm

from survae import Dataset, SurVAE, TrainingSnapshot

TrainedModels = dict[tuple[int, int], tuple[SurVAE, dict[int, TrainingSnapshot]]]


def train_models(model_generators: list[Callable[[], SurVAE]], datasets: list[Dataset],
                 batch_size=1_000, test_size=10_000, epochs=1_000, lr=0.001, log_period=10) -> TrainedModels:
    """Train models on datasets."""
    models = {}

    with tqdm(total=len(datasets) * len(model_generators)) as pbar:
        for i, dataset in (enumerate(datasets)):
            for j, model_generator in (enumerate(model_generators)):
                model = model_generator()

                results = model.train(
                    dataset, batch_size=batch_size, test_size=test_size,
                    epochs=epochs, lr=lr, log_period=log_period,
                )

                models[i, j] = (model, results)

                pbar.update(1)

    return models
