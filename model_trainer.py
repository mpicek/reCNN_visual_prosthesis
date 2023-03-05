import wandb
from Lurz_dataset import LurzDataModule

from models import reCNN_FullFactorized
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import ProgressBar
import pytorch_lightning as pl
from models import reCNN_bottleneck_CyclicGauss3d
from datetime import timedelta

from pprint import pprint
from Antolik_dataset import AntolikDataModule


def get_best_model(
    wandb_run,
    model_class=reCNN_bottleneck_CyclicGauss3d,
    model_artifact_name="reCNN_bottleneck_CyclicGauss3d",
):
    """Returns the best model

    Args:
        wandb_run (wandb.run): Wandb run for logging
        model_class (class, optional): The class of the model. Defaults to reCNN_bottleneck_CyclicGauss3d.
        model_artifact_name (str, optional): Name of the artifact of the model. Defaults to "reCNN_bottleneck_CyclicGauss3d".

    Returns:
        pl.model: The best model
    """

    if model_artifact_name == None:
        model_artifact_name = model_class.__str__()

    model_artifact = wandb_run.use_artifact(model_artifact_name + ":best")
    model_checkpoint_path = model_artifact.download()

    best_model = model_class.load_from_checkpoint(model_checkpoint_path)

    return best_model


def Lurz_dataset_preparation_function(config, run=None):
    """Gets config, can edit it.
        Returns Pytorch Lightning DataModule of the dataset from Lurz's publication

    Args:
        config (dict): Configuration dictionary
        run (wandb.run, optional): Wandb run for logging. If None, nothing is logged. Defaults to None.

    Returns:
        pl.DataModule: Pytorch Lightning DataModule of the dataset.
    """    

    # setup datamodule - use artifact
    data_dir = "data/lurz2020/static20457-5-9-preproc0"

    if run is not None:
        dataset_artifact = run.use_artifact(config["dataset_artifact_name"])
        data_dir = dataset_artifact.download()

    dataset_config = {
        "data_dir": data_dir,
        "batch_size": config["batch_size"],
        "normalize": True,
        "exclude": "images",
    }

    dm = LurzDataModule(**dataset_config)
    dm.prepare_data()
    dm.setup()

    # update config for initialization of model (<- certain config parameters depend on data)
    config.update(
        {
            "input_channels": dm.get_input_shape()[0],
            "input_size_x": dm.get_input_shape()[1],
            "input_size_y": dm.get_input_shape()[2],
            "num_neurons": dm.get_output_shape()[0],
            "mean_activity": dm.get_mean(),
        }
    )

    return dm


def Antolik_dataset_preparation_function(config, run=None, load_data=True):
    """Gets config, can edit it.
        Uses only Antolik's in-silico dataset

    Args:
        config (dict): Configuration dictionary
        run (wandb.run, optional): Wandb run for logging. If None, nothing is logged. Defaults to None.

    Returns:
        pl.DataModule: Pytorch Lightning DataModule of the dataset.
    """
    

    if config["train_on_test"]:
        dataset_config = {
            "train_data_dir": config["test_data_dir"],
            "test_data_dir": config["test_data_dir"],
            "batch_size": config["batch_size"],
            "normalize": config["normalize"],
            "val_size": 500,
            "brain_crop": config["brain_crop"],
            "stimulus_crop": config["stimulus_crop"],
            "ground_truth_positions_file_path": config["ground_truth_positions_file_path"],
            "ground_truth_orientations_file_path": config["ground_truth_orientations_file_path"],
        }
    else:
        dataset_config = {
            "train_data_dir": config["train_data_dir"],
            "test_data_dir": config["test_data_dir"],
            "batch_size": config["batch_size"],
            "normalize": config["normalize"],
            "val_size": config["val_size"],
            "brain_crop": config["brain_crop"],
            "stimulus_crop": config["stimulus_crop"],
            "ground_truth_positions_file_path": config["ground_truth_positions_file_path"],
            "ground_truth_orientations_file_path": config["ground_truth_orientations_file_path"],
            "num_workers": 0,
        }

    if run is not None:
        raise NotImplementedError()

    dm = AntolikDataModule(**dataset_config)

    if load_data:
        dm.prepare_data()
        dm.setup()

        # update config for initialization of model (<- certain config parameters depend on data)
        config.update(
            {
                "input_channels": dm.get_input_shape()[0],
                "input_size_x": dm.get_input_shape()[1],
                "input_size_y": dm.get_input_shape()[2],
                "num_neurons": dm.get_output_shape()[0],
                "mean_activity": dm.get_mean(),
                "filtered_neurons":dm.get_filtered_neurons(),
            }
        )

    return dm

def get_model(
    config,
    dm,
    model_class=reCNN_FullFactorized,
):
    """Sets up a model.

    Args:
        config (dict): The most important fields are:
            - needs_ground_truth
            - model_needs_dataloader
        dm (pl.DataModule): The dataset
        model_class (class, optional): Class of the model to be created. Defaults to reCNN_FullFactorized.

    Returns:
        _type_: _description_
    """
    model = None
    # Set up model
    if config["needs_ground_truth"]:
        pos_x, pos_y, orientations = dm.get_ground_truth(
            ground_truth_positions_file_path=config["ground_truth_positions_file_path"], 
            ground_truth_orientations_file_path=config["ground_truth_orientations_file_path"], 
            in_degrees=False, 
            positions_minus_y=config["positions_minus_y"], 
            positions_minus_x=config["positions_minus_x"], 
            positions_swap_axes=config["positions_swap_axes"]
        )
        resolution = (dm.get_input_shape()[1], dm.get_input_shape()[2])
        xlim = [-dm.get_stimulus_visual_angle()/2, dm.get_stimulus_visual_angle()/2]
        ylim = [-dm.get_stimulus_visual_angle()/2, dm.get_stimulus_visual_angle()/2]
        model = model_class(pos_x, pos_y, orientations, resolution, xlim, ylim, **config)
    elif config["model_needs_dataloader"]:
        # model = model_class(dataloader=dm, **config)
        model = model_class(**config)
        model.init_neurons(dm)
    else:
        model = model_class(**config)
    
    return model


def run_wandb_training(
    config,
    dataset_preparation_function,
    entity,
    project,
    model_artifact_name=None,
    model_class=reCNN_FullFactorized,
    early_stopping_monitor="val/corr",
    early_stopping_mode="max",
    model_checkpoint_monitor="val/corr",
    model_checkpoint_mode="max",
):
    """Sets up a dataset and a model, sets up wandb session and runs a model
        training with the help of Wandb. Subsequently it evaluates the model
        and prints the results.

    Args:
        config (dict): Configuration dictionary
        dataset_preparation_function (function): A function that sets up the dataset
        entity (str): Name of the wandb user.
        project (str): Name of the wandb project
        model_artifact_name (str, optional): Name of the wandb artifact for the model. Defaults to None.
        model_class (class, optional): A class of the model. Defaults to reCNN_FullFactorized.
        early_stopping_monitor (str, optional): A measure which is watched and based on which the early stopping of the training occurs. Defaults to "val/corr".
        early_stopping_mode (str, optional): Whether we want to reach maximum or minimum of the watched measure which decides about the early stopping. Defaults to "max".
        model_checkpoint_monitor (str, optional): A measure which is watched and based on which the best model is decided. Defaults to "val/corr".
        model_checkpoint_mode (str, optional): Whether we want to reach maximum or minimum of the watched measure which decides about best model. Defaults to "max".

    Returns:
        _type_: _description_
    """


    # init wandb run
    run = wandb.init(
        config=config,
        project=project,
        entity=entity,
    )

    # Access all hyperparameter values through wandb.config
    config = dict(wandb.config)

    pl.seed_everything(config["seed"], workers=True)
    
    pprint(config)

    print("Setting up the dataset...")

    dm = dataset_preparation_function(config, None)

    model = get_model(config, dm, model_class)
    # summary(model, torch.zeros((config["batch_size"], dm.get_input_shape()[0], dm.get_input_shape()[1], dm.get_input_shape()[2])))

    # setup wandb logger

    print(wandb.run.name)

    wandb_logger = WandbLogger(log_model=False)
    wandb_logger.watch(model, log=None, log_freq=500, log_graph=False) # the log_freq doesn't work

    # define callbacks for the training
    early_stop = EarlyStopping(
        monitor=early_stopping_monitor,
        patience=config["patience"],
        mode=early_stopping_mode,
    )
    # we put it into brno2 storage as it is much much larger than the default budejovice1 storage
    # TODO (change in future)
    checkpoint_callback = ModelCheckpoint(
        dirpath="/storage/brno2/home/mpicek/MODEL_CHECKPOINTS/" + wandb.run.name, save_top_k=1, monitor=model_checkpoint_monitor, mode=model_checkpoint_mode#, save_weights_only=True
    )

    # define the trainer
    trainer = None
    if "max_time" in config.keys():
        trainer = pl.Trainer(
            callbacks=[early_stop, checkpoint_callback],
            max_time=timedelta(hours=config["max_time"]),
            gpus=[0],
            logger=wandb_logger,
            # deterministic=True,
            enable_checkpointing=True,
            log_every_n_steps=1000,
            # fast_dev_run=3
        )
    else:
        trainer = pl.Trainer(
            callbacks=[early_stop, checkpoint_callback],
            max_epochs=config["max_epochs"],
            gpus=[0],
            logger=wandb_logger,
            # deterministic=True,
            enable_checkpointing=True,
            log_every_n_steps=1000, # THIS WORKS... log_freq didn't work in wandb_logger
            # fast_dev_run=3
        )

    if config["train_on_val"]:
        trainer.fit(
            model,
            train_dataloaders=dm.val_dataloader(),
            val_dataloaders=dm.val_dataloader(),
        )

    else:
        trainer.fit(
            model,
            train_dataloaders=dm.train_dataloader(),
            val_dataloaders=dm.val_dataloader(),
        )

    best_observed_val_metric = (
        checkpoint_callback.best_model_score.cpu().detach().numpy()
    )
    print(
        "Best model's "
        + config["observed_val_metric"]
        + ": "
        + str(best_observed_val_metric)
    )

    if model_artifact_name == None:
        model_artifact_name = model.__str__()

    # add best corr to metadata
    metadata = {**config, "best_model_score": best_observed_val_metric}

    # add model artifact
    print(model_artifact_name)
    best_model_artifact = wandb.Artifact(
        model_artifact_name, type="model", metadata=metadata
    )
    print(best_model_artifact)
    best_model_artifact.add_file(checkpoint_callback.best_model_path)
    run.log_artifact(best_model_artifact)

    # say to wandb that the best val/corr of the model is the best one
    # and not the last one!! (it is the default behavour!!)
    run.summary[config["observed_val_metric"]] = best_observed_val_metric

    print(checkpoint_callback.best_model_path)

    model = model_class.load_from_checkpoint(checkpoint_callback.best_model_path)

    if config["test"]:
        dm.model_performances(model, trainer)

        # result_artifact = wandb.Artifact(name="RESULT_" + model_artifact_name, type="result",
        #     metadata=results[0])
        # run.log_artifact(result_artifact)

    return model


def run_training_without_logging(
    config,
    dataset_preparation_function,
    entity,
    project,
    model_artifact_name=None,
    model_class=reCNN_FullFactorized,
    early_stopping_monitor="val/corr",
    early_stopping_mode="max",
    model_checkpoint_monitor="val/corr",
    model_checkpoint_mode="max",
    # **config,
):
    """Sets up a dataset and a model and runs a model
        training WITHOUT Wandb. Subsequently it evaluates the model
        and prints the results.

    Args:
        config (dict): Configuration dictionary
        dataset_preparation_function (function): A function that sets up the dataset
        entity (str): Name of the wandb user.
        project (str): Name of the wandb project
        model_artifact_name (str, optional): Name of the wandb artifact for the model. Defaults to None.
        model_class (class, optional): A class of the model. Defaults to reCNN_FullFactorized.
        early_stopping_monitor (str, optional): A measure which is watched and based on which the early stopping of the training occurs. Defaults to "val/corr".
        early_stopping_mode (str, optional): Whether we want to reach maximum or minimum of the watched measure which decides about the early stopping. Defaults to "max".
        model_checkpoint_monitor (str, optional): A measure which is watched and based on which the best model is decided. Defaults to "val/corr".
        model_checkpoint_mode (str, optional): Whether we want to reach maximum or minimum of the watched measure which decides about best model. Defaults to "max".

    Returns:
        _type_: _description_
    """

    pl.seed_everything(config["seed"], workers=True)

    # # init wandb run
    # run = wandb.init(
    #     config=config,
    #     project=project,
    #     entity=entity,
    # )

    # Access all hyperparameter values through wandb.config
    # config = dict(wandb.config)
    pprint(config)
    pl.seed_everything(config["seed"], workers=True)

    dm = dataset_preparation_function(config, None)

    # Set up model
    model = get_model(config, dm, model_class)

    # summary(model, torch.zeros((config["batch_size"], dm.get_input_shape()[0], dm.get_input_shape()[1], dm.get_input_shape()[2])))

    # setup wandb logger
    # wandb_logger = WandbLogger(log_model=True)
    # wandb_logger.watch(model, log="all", log_freq=250)

    # define callbacks for the training
    early_stop = EarlyStopping(
        monitor=early_stopping_monitor,
        patience=config["patience"],
        mode=early_stopping_mode,
    )
    # checkpoint_callback = ModelCheckpoint(
    #     save_top_k=1, monitor=model_checkpoint_monitor, mode=model_checkpoint_mode
    # )

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath="/storage/brno2/home/mpicek/MODEL_CHECKPOINTS/" + wandb.run.name, save_top_k=1, monitor=model_checkpoint_monitor, mode=model_checkpoint_mode
    # )

    # define the trainer
    trainer = pl.Trainer(
        callbacks=[early_stop],
        max_epochs=config["max_epochs"],
        gpus=[0],
        logger=False,
        log_every_n_steps=100,
        # deterministic=True,
        enable_checkpointing=True,
    )

    if config["train_on_val"]:
        trainer.fit(
            model,
            train_dataloaders=dm.val_dataloader(),
            val_dataloaders=dm.val_dataloader(),
        )

    else:
        trainer.fit(
            model,
            train_dataloaders=dm.train_dataloader(),
            val_dataloaders=dm.val_dataloader(),
        )

    # best_observed_val_metric = (
    #     checkpoint_callback.best_model_score.cpu().detach().numpy()
    # )
    # print(
    #     "Best model's "
    #     + config["observed_val_metric"]
    #     + ": "
    #     + str(best_observed_val_metric)
    # )

    # add best corr to metadata
    # metadata = {**config, "best_model_score": best_observed_val_metric}

    # print(checkpoint_callback.best_model_path)

    # model = model_class.load_from_checkpoint(checkpoint_callback.best_model_path)

    # if config["test"]:
    #     dm.model_performances(model, trainer)

        # result_artifact = wandb.Artifact(name="RESULT_" + model_artifact_name, type="result",
        #     metadata=results[0])
        # run.log_artifact(result_artifact)

    return model
