

from nntransfer.configs.base import Description
from nntransfer.configs.experiment import Experiment
from nntransfer.configs.transfer_experiment import TransferExperiment
from neural_cotraining.configs import model, dataset, trainer

experiments = {}
transfer_experiments = {}


#The single-task trained baseline for image classification on TinyImageNet

experiments = {}
seeds = [1000] #[1000,2000,3000,4000,5000]
lr = 0.1
patience = lr_decay_steps = 5
lr_decay = 0.4

opts = ["SGD"]
for seed in seeds:
    for opt in opts:
        trainer_name = "opt_{}_lr_{}_lrDecay_0.4_seed_{}".format(opt, lr, seed)
        optimizer_options = {"momentum": 0.75, "lr": lr, "weight_decay": 5e-4}
        experiments[Description(name="TinyImgNet_B&W_VGG19_bn_ConvReadout_" + trainer_name, seed=seed)] = TransferExperiment([Experiment(
            dataset=dataset.ImageDatasetConfig(comment="TinyImgNet_B&W", dataset_cls="TinyImageNet", apply_grayscale=True,
                                               add_corrupted_test=False),
            model=model.ClassificationModelConfig(comment="vgg19_bn_conv_grayscaleInput", type="vgg19_bn",
                                                  readout_type="conv", input_channels=1, dataset_cls="TinyImageNet"),
            trainer=trainer.CoTrainerConfig(comment=trainer_name, optimizer=opt,
                                          optimizer_options=optimizer_options,
                                          lr_decay=lr_decay, loss_weighing=False, lr_decay_steps=lr_decay_steps, patience=patience,
                                          to_monitor=['img_classification'],
                                          max_iter=400, scheduler="adaptive", scheduler_options={"mtl": False},
                                          verbose=True, loss_functions={"img_classification": "CrossEntropyLoss"}),
            seed=seed,
        )])