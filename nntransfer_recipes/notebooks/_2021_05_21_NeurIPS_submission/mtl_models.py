
from nntransfer.configs.base import Description
from nntransfer.configs.experiment import Experiment
from nntransfer.configs.transfer_experiment import TransferExperiment
from neural_cotraining.configs import model, dataset, trainer

experiments = {}
transfer_experiments = {}


#MTL on image classification and neural prediction. This experiment can reproduce MTL=Monkey, MTL-Oracle and MTL-Shuffled
#given the corresponding neural_dataset_folder_name.
neural_dataset_folder_name = "mtl_monkey_dataset"  #"mtl_shuffled_dataset", "mtl_oracle_dataset"

restore_best = True
to_monitor = ['img_classification']
train_cycler = "MTL_Cycler"
loss_weighing = True
scale_loss = False
ratio = {1:1}   #the batch ratio {1 : number of batches for neural prediction / number of batch for image classification}

lr = 0.1
opt = 'SGD'
lr_decay = 0.3
layer = 17
seeds = [1000] #[1000, 2001, 3000, 4000, 5000]
neural_loss = "MSELoss"

for seed in seeds:
    optimizer_options = {"momentum": 0.75, "lr": lr, "weight_decay": 5e-4}
    dataset_name = "dataset_comment"
    model_name = 'VGG19bn_MTL_classificationReadout_Conv_V1ReadoutLayer_{}_gamma_{}'.format(layer, 0.5)
    trainer_name = "opt_{}_lr_{}_decay_0.3_trainCycler_{}_lossWeighing_{}_batchRatio_{}".format(opt, lr, train_cycler, loss_weighing, ratio)
    experiments[Description(name=dataset_name + "_" + model_name + "_" + trainer_name, seed=seed)] = TransferExperiment([Experiment(
        dataset=dataset.MTLDatasetsConfig(comment=dataset_name, neural_dataset_dict={"comment": "", "dataset":neural_dataset_folder_name,
                                                                                     "scale": 1.0,
                                                                                     "crop": 0, "seed":1000, "subsample": 1,
                                                                                     "target_types" : ['v1'], "normalize" : True,
                                                                                     "apply_augmentation" : False, "add_fly_corrupted_test" : {},
                                                                                     "apply_grayscale" : False, "resize" : 0,
                                                                                     "stats" : {'mean': (0.4519,), 'std': (0.2221,)}  #stats of TIN images
                                                                                     },
                                          img_dataset_dict={"comment": "", "dataset_cls":"TinyImageNet", "apply_grayscale": True,
                                                            "add_corrupted_test":False}),


        model=model.MTLModelConfig(comment=model_name, classification=True, classification_readout_type="conv",
                                   input_size=64, num_classes=200, pretrained=False, vgg_type="vgg19_bn",
                                   v1_fine_tune=True, v1_gamma_readout=0.5, v1_model_layer=layer, neural_input_channels=1,
                                   classification_input_channels=1),

        trainer=trainer.CoTrainerConfig(comment=trainer_name, optimizer=opt,
                                      optimizer_options=optimizer_options,
                                      train_cycler=train_cycler, loss_weighing=loss_weighing, to_monitor=to_monitor,
                                      train_cycler_args={"main_key":"img_classification", "ratio":ratio},
                                      max_iter=407, scheduler="adaptive", scheduler_options={"mtl": False}, mtl=True,
                                      lr_decay_steps=5, scale_loss=scale_loss, patience=5, lr_decay=lr_decay, restore_best=restore_best,
                                      verbose=True, loss_functions={"img_classification": "CrossEntropyLoss",
                                                                    "v1": neural_loss}),
        seed=seed)])