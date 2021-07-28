

from nntransfer.configs.base import Description
from nntransfer.configs.experiment import Experiment
from nntransfer.configs.transfer_experiment import TransferExperiment
from neural_cotraining.configs import model, dataset, trainer

experiments = {}
transfer_experiments = {}


#This experiment reproduces training the VGG on the real V1 data to obtain a single-task trained model on monkey V1

layers = [17]
seeds = [1000]
dataset_cls = "neural_v1_dataset"
train_cycler =  "LongCycler"
loss_weighing = False
scale_loss = True
for seed in seeds:
    for layer in layers:
            dataset_name_neural = "v1_crop_{}_scale_{}".format(36, 0.4)
            model_name_neural = 'VGG19bn_classificationReadout_Conv_V1ReadoutLayer_{}_gamma_{}'.format(layer, 0.5)
            trainer_name_neural = "Adam_lr_0.0005_decay_0.5_trainCycler_{}_lossWeighing_{}".format(train_cycler, loss_weighing)
            experiments[Description(name=dataset_name_neural + "_" + model_name_neural + "_" + trainer_name_neural, seed=seed)] = TransferExperiment([Experiment(
                dataset=dataset.NeuralDatasetConfig(comment=dataset_name_neural, dataset=dataset_cls, subsample=1, crop=36, scale= 0.4,),

                model=model.MTLModelConfig(comment=model_name_neural, pretrained=False, vgg_type="vgg19_bn",
                                           v1_fine_tune=True, v1_gamma_readout=0.5, v1_model_layer=layer, neural_input_channels=1),

                trainer=trainer.CoTrainerConfig(comment=trainer_name_neural, optimizer="Adam",
                                              optimizer_options={"amsgrad": False, "lr": 0.0005, "weight_decay": 5e-4},
                                              train_cycler=train_cycler, loss_weighing=loss_weighing, scale_loss=scale_loss,
                                              train_cycler_args={}, patience=5, threshold=1e-6, lr_decay=0.5, to_monitor=['v1'],
                                              max_iter=150, scheduler="adaptive", scheduler_options={"mtl": False},
                                              verbose=True, loss_functions={"v1": "PoissonLoss"},
                                              mtl=True, lr_decay_steps=5),
                seed=seed)])