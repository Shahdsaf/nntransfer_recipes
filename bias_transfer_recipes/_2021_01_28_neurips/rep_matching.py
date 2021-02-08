from bias_transfer.configs.base import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

transfer_experiments = {}


class BaselineDataset(dataset.ImageDatasetConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.dataset_cls: str = "CIFAR100"
        self.input_size: int = 32
        self.add_corrupted_test: bool = False
        self.batch_size = 64
        super(BaselineDataset, self).__init__(**kwargs)


class BaselineModel(model.CIFAR100):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type: str = "resnet50"
        super(BaselineModel, self).__init__(**kwargs)


class BaselineTrainer(
    trainer.mixins.TransferMixin,
    trainer.mixins.NoiseAugmentationMixin,
    trainer.mixins.RepresentationMatchingMixin,
    trainer.Classification,
):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.max_iter = 100
        self.patience = 1000
        self.lr_milestones = (60, 80)
        super(BaselineTrainer, self).__init__(**kwargs)


seed = 42

for rep_matching in (
    "representation matching bn-freeze",
    "no representation matching bn-freeze",
    "representation matching",
    "no representation matching",
):
    experiments = []
    layers = {
        # core output
        # "flatten": "core",
        "layer2.1.conv1": "layer2.1.conv1",
    }
    transfer_settings = {
        "representation matching": [
            {
                "model": {
                    "get_intermediate_rep": layers,
                },
                "trainer": {
                    "representation_matching": {
                        "representations": list(layers.values()),
                        "criterion": "mse",
                        "combine_losses": "avg",
                        "second_noise_std": {(0, 0.5): 1.0},
                        "lambda": 1.0,
                        "only_for_clean": True,
                    },
                },
            },
            {"trainer": {"freeze_bn": False}},
        ],
        "representation matching bn-freeze": [
            {
                "model": {
                    "get_intermediate_rep": layers,
                },
                "trainer": {
                    "representation_matching": {
                        "representations": list(layers.values()),
                        "criterion": "mse",
                        "combine_losses": "avg",
                        "second_noise_std": {(0, 0.5): 1.0},
                        "lambda": 1.0,
                        "only_for_clean": True,
                    },
                },
            },
            {"trainer": {"freeze_bn": True}},
        ],
        "no representation matching bn-freeze": [{}, {"trainer": {"freeze_bn": True}}],
        "no representation matching": [{}, {"trainer": {"freeze_bn": False}}],
    }

    # Step 1: Training on Noise
    experiments.append(
        Experiment(
            dataset=BaselineDataset(),
            model=BaselineModel(),
            trainer=BaselineTrainer(
                comment=f"Training ",
                noise_std={
                    0.08: 0.1,
                    0.12: 0.1,
                    0.18: 0.1,
                    0.26: 0.1,
                    0.38: 0.1,
                    -1: 0.5,
                },
            ),
            seed=seed,
        )
    )

    # Step 2: Training on Clean
    experiments.append(
        Experiment(
            dataset=BaselineDataset(),
            model=BaselineModel(),
            trainer=BaselineTrainer(
                freeze=(
                    "conv1.weight",
                    "bn1.weight",
                    "bn1.bias",
                    "bn1.running_mean",
                    "bn1.running_var",
                    "bn1.num_batches_tracked",
                    "layer1.0.conv1.weight",
                    "layer1.0.bn1.weight",
                    "layer1.0.bn1.bias",
                    "layer1.0.bn1.running_mean",
                    "layer1.0.bn1.running_var",
                    "layer1.0.bn1.num_batches_tracked",
                    "layer1.0.conv2.weight",
                    "layer1.0.bn2.weight",
                    "layer1.0.bn2.bias",
                    "layer1.0.bn2.running_mean",
                    "layer1.0.bn2.running_var",
                    "layer1.0.bn2.num_batches_tracked",
                    "layer1.1.conv1.weight",
                    "layer1.1.bn1.weight",
                    "layer1.1.bn1.bias",
                    "layer1.1.bn1.running_mean",
                    "layer1.1.bn1.running_var",
                    "layer1.1.bn1.num_batches_tracked",
                    "layer1.1.conv2.weight",
                    "layer1.1.bn2.weight",
                    "layer1.1.bn2.bias",
                    "layer1.1.bn2.running_mean",
                    "layer1.1.bn2.running_var",
                    "layer1.1.bn2.num_batches_tracked",
                    "layer2.0.conv1.weight",
                    "layer2.0.bn1.weight",
                    "layer2.0.bn1.bias",
                    "layer2.0.bn1.running_mean",
                    "layer2.0.bn1.running_var",
                    "layer2.0.bn1.num_batches_tracked",
                    "layer2.0.conv2.weight",
                    "layer2.0.bn2.weight",
                    "layer2.0.bn2.bias",
                    "layer2.0.bn2.running_mean",
                    "layer2.0.bn2.running_var",
                    "layer2.0.bn2.num_batches_tracked",
                    "layer2.0.downsample.0.weight",
                    "layer2.0.downsample.1.weight",
                    "layer2.0.downsample.1.bias",
                    "layer2.0.downsample.1.running_mean",
                    "layer2.0.downsample.1.running_var",
                    "layer2.0.downsample.1.num_batches_tracked",
                    "layer2.1.conv1.weight",
                ),
                reset=(
                    "layer2.1.bn1.weight",
                    "layer2.1.bn1.bias",
                    "layer2.1.bn1.running_mean",
                    "layer2.1.bn1.running_var",
                    "layer2.1.bn1.num_batches_tracked",
                    "layer2.1.conv2.weight",
                    "layer2.1.bn2.weight",
                    "layer2.1.bn2.bias",
                    "layer2.1.bn2.running_mean",
                    "layer2.1.bn2.running_var",
                    "layer2.1.bn2.num_batches_tracked",
                    "layer3.0.conv1.weight",
                    "layer3.0.bn1.weight",
                    "layer3.0.bn1.bias",
                    "layer3.0.bn1.running_mean",
                    "layer3.0.bn1.running_var",
                    "layer3.0.bn1.num_batches_tracked",
                    "layer3.0.conv2.weight",
                    "layer3.0.bn2.weight",
                    "layer3.0.bn2.bias",
                    "layer3.0.bn2.running_mean",
                    "layer3.0.bn2.running_var",
                    "layer3.0.bn2.num_batches_tracked",
                    "layer3.0.downsample.0.weight",
                    "layer3.0.downsample.1.weight",
                    "layer3.0.downsample.1.bias",
                    "layer3.0.downsample.1.running_mean",
                    "layer3.0.downsample.1.running_var",
                    "layer3.0.downsample.1.num_batches_tracked",
                    "layer3.1.conv1.weight",
                    "layer3.1.bn1.weight",
                    "layer3.1.bn1.bias",
                    "layer3.1.bn1.running_mean",
                    "layer3.1.bn1.running_var",
                    "layer3.1.bn1.num_batches_tracked",
                    "layer3.1.conv2.weight",
                    "layer3.1.bn2.weight",
                    "layer3.1.bn2.bias",
                    "layer3.1.bn2.running_mean",
                    "layer3.1.bn2.running_var",
                    "layer3.1.bn2.num_batches_tracked",
                    "layer4.0.conv1.weight",
                    "layer4.0.bn1.weight",
                    "layer4.0.bn1.bias",
                    "layer4.0.bn1.running_mean",
                    "layer4.0.bn1.running_var",
                    "layer4.0.bn1.num_batches_tracked",
                    "layer4.0.conv2.weight",
                    "layer4.0.bn2.weight",
                    "layer4.0.bn2.bias",
                    "layer4.0.bn2.running_mean",
                    "layer4.0.bn2.running_var",
                    "layer4.0.bn2.num_batches_tracked",
                    "layer4.0.downsample.0.weight",
                    "layer4.0.downsample.1.weight",
                    "layer4.0.downsample.1.bias",
                    "layer4.0.downsample.1.running_mean",
                    "layer4.0.downsample.1.running_var",
                    "layer4.0.downsample.1.num_batches_tracked",
                    "layer4.1.conv1.weight",
                    "layer4.1.bn1.weight",
                    "layer4.1.bn1.bias",
                    "layer4.1.bn1.running_mean",
                    "layer4.1.bn1.running_var",
                    "layer4.1.bn1.num_batches_tracked",
                    "layer4.1.conv2.weight",
                    "layer4.1.bn2.weight",
                    "layer4.1.bn2.bias",
                    "layer4.1.bn2.running_mean",
                    "layer4.1.bn2.running_var",
                    "layer4.1.bn2.num_batches_tracked",
                    "fc.weight",
                    "fc.bias",
                ),
            ),
            seed=seed,
        )
    )

    transfer_experiments[
        Description(
            name=f"Transfer noise augmented -> clean {rep_matching}",
            seed=seed,
        )
    ] = TransferExperiment(experiments, update=transfer_settings[rep_matching])
