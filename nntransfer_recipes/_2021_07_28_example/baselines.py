from nntransfer.configs.base import Description
from nntransfer.configs.transfer_experiment import TransferExperiment
from nntransfer.configs.experiment import Experiment
from nntransfer.configs import *
from bias_transfer.configs import *

transfer_experiments = {}
experiments = []

experiments.append(
    Experiment(
        dataset=ImageDatasetConfig(dataset_cls="CIFAR10"),
        model=ModelConfig(
            type="lenet5",
        ),
        trainer=TrainerConfig(comment=f"Baseline Trainer"),
        seed=42,
    )
)

transfer_experiments[
    Description(
        name=f"Baseline Experiment",
        seed=42,
    )
] = TransferExperiment(experiments)
