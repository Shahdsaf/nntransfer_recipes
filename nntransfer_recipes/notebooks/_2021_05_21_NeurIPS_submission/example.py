
from nntransfer.configs.base import Description
from nntransfer.configs.experiment import Experiment
from nntransfer.configs.transfer_experiment import TransferExperiment
from neural_cotraining.configs import model, dataset, trainer

seed = 1000
experiments = {}
transfer_experiments = {}


#An example experiment for training a VGG on image classification
# experiments[Description(name="test",
#                         seed=seed)] = TransferExperiment([Experiment(dataset=dataset.ImageDatasetConfig(comment="",
#                                                                                dataset_cls="TinyImageNet", add_corrupted_test=False),
#                                                  model=model.ClassificationModelConfig(comment="",
#                                                                          dataset_cls="TinyImageNet"),
#                                                  trainer=trainer.CoTrainerConfig(
#                                                      comment="", max_iter=0), seed=seed)])

#An example experiment for training a VGG on neural prediction
# experiments[Description(name="test",
#                         seed=seed)] = TransferExperiment([Experiment(dataset=dataset.NeuralDatasetConfig(comment=""),
#                                                  model=model.MTLModelConfig(comment=""),
#                                                  trainer=trainer.CoTrainerConfig(mtl=True,
#                                                      comment="", max_iter=3, loss_functions={"v1": "PoissonLoss"},
#                                                  to_monitor=["v1"]), seed=seed)])



#An example experiment for training a VGG on neural prediction and image classification at once (MTL)
experiments[Description(name="test",
                        seed=seed)] = TransferExperiment([Experiment(dataset=dataset.MTLDatasetsConfig(comment="", v1_dataset_dict={"comment": ""},
                                                                                                       img_dataset_dict={"comment":"",
                                                                                                        "dataset_cls":"TinyImageNet", "add_corrupted_test":False}),
                                                 model=model.MTLModelConfig(comment="", classification=True, classification_readout_type="conv", input_size=64,
                                                                            ),
                                                 trainer=trainer.CoTrainerConfig(mtl=True,
                                                     comment="", max_iter=3, loss_functions={"v1": "MSELoss", "img_classification": "CrossEntropyLoss"},
                                                 to_monitor=["img_classification"], loss_weighing=True, train_cycler="MTL_Cycler",
                                                train_cycler_args={"main_key":"img_classification", "ratio":{1:1}}, scale_loss=False), seed=seed)])