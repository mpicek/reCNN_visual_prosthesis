import wandb
from Lurz_dataset import LurzDataModule

from models import reCNN_FullFactorized
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import ProgressBar
import pytorch_lightning as pl
from models import reCNN_bottleneck_CyclicGauss3d

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


def Antolik_dataset_preparation_function_test(config, run=None, return_ground_truth=False):
    """Gets config, can edit it.
        Uses only Antolik's in-silico test dataset (for both train and test), therefore
            the loading is much quicker.

    Args:
        config (dict): Configuration dictionary
        run (wandb.run, optional): Wandb run for logging. If None, nothing is logged. Defaults to None.

    Returns:
        pl.DataModule: Pytorch Lightning DataModule of the dataset.
    """

    path_train = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik/one_trials.pickle"
    path_test = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik/ten_trials.pickle"

    dataset_config = {
        "train_data_dir": path_test,
        "test_data_dir": path_test,
        "batch_size": config["batch_size"],
        "normalize": True,
        "val_size": 500,
        "brain_crop": 0.8,
        "stimulus_crop": "auto",
        "ground_truth_path": "data/antolik/position_dictionary.pickle",
    }

    if run is not None:
        raise NotImplementedError()

    dm = AntolikDataModule(**dataset_config)
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


def Antolik_dataset_preparation_function(config, run=None):
    """Gets config, can edit it.
        Uses only Antolik's in-silico dataset

    Args:
        config (dict): Configuration dictionary
        run (wandb.run, optional): Wandb run for logging. If None, nothing is logged. Defaults to None.

    Returns:
        pl.DataModule: Pytorch Lightning DataModule of the dataset.
    """


    path_train = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik/one_trials.pickle"
    path_test = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik/ten_trials.pickle"

    dataset_config = {
        "train_data_dir": path_train,
        "test_data_dir": path_test,
        "batch_size": config["batch_size"],
        "normalize": True,
        "val_size": 5000,
        "brain_crop": 0.8,
        "stimulus_crop": "auto",
        "ground_truth_path": "data/antolik/position_dictionary.pickle",
    }

    if run is not None:
        raise NotImplementedError()

    dm = AntolikDataModule(**dataset_config)
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
    needs_ground_truth=False,
    # **config,
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

    pl.seed_everything(config["seed"], workers=True)

    # init wandb run
    run = wandb.init(
        config=config,
        project=project,
        entity=entity,
    )

    # Access all hyperparameter values through wandb.config
    config = dict(wandb.config)
    pprint(config)

    dm = dataset_preparation_function(config, None)

    model = None
    # Set up model
    if needs_ground_truth:
        pos_x, pos_y, orientations = dm.get_ground_truth(config["ground_truth_positions_file_path"], config["ground_truth_orientations_file_path"])
        resolution = (dm.get_input_shape()[1], dm.get_input_shape()[2])
        xlim = [-dm.get_stimulus_visual_angle()/2, dm.get_stimulus_visual_angle()/2]
        ylim = [-dm.get_stimulus_visual_angle()/2, dm.get_stimulus_visual_angle()/2]
        model = model_class(pos_x, pos_y, orientations, resolution, xlim, ylim, **config)
    else:
        model = model_class(**config)

    # summary(model, torch.zeros((config["batch_size"], dm.get_input_shape()[0], dm.get_input_shape()[1], dm.get_input_shape()[2])))

    # setup wandb logger
    wandb_logger = WandbLogger(log_model=True)
    wandb_logger.watch(model, log="all", log_freq=250)

    # define callbacks for the training
    early_stop = EarlyStopping(
        monitor=early_stopping_monitor,
        patience=config["patience"],
        mode=early_stopping_mode,
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor=model_checkpoint_monitor, mode=model_checkpoint_mode
    )

    class LitProgressBar(ProgressBar):
        def get_metrics(self, trainer, model):
            # don't show the version number
            items = super().get_metrics(trainer, model)
            items.pop("v_num", None)
            return items

    bar = LitProgressBar()

    # define the trainer
    trainer = pl.Trainer(
        callbacks=[early_stop, checkpoint_callback, bar],
        max_epochs=config["max_epochs"],
        gpus=[0],
        logger=wandb_logger,
        log_every_n_steps=1,
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
    best_model_artifact = wandb.Artifact(
        model_artifact_name, type="model", metadata=metadata
    )
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

    dm = dataset_preparation_function(config, None)

    # Set up model
    model = model_class(**config)

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
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor=model_checkpoint_monitor, mode=model_checkpoint_mode
    )

    class LitProgressBar(ProgressBar):
        def get_metrics(self, trainer, model):
            # don't show the version number
            items = super().get_metrics(trainer, model)
            items.pop("v_num", None)
            return items

    bar = LitProgressBar()

    # define the trainer
    trainer = pl.Trainer(
        callbacks=[early_stop, checkpoint_callback, bar],
        max_epochs=config["max_epochs"],
        gpus=[0],
        # logger=wandb_logger,
        log_every_n_steps=1,
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

    best_observed_val_metric = (
        checkpoint_callback.best_model_score.cpu().detach().numpy()
    )
    print(
        "Best model's "
        + config["observed_val_metric"]
        + ": "
        + str(best_observed_val_metric)
    )

    # add best corr to metadata
    metadata = {**config, "best_model_score": best_observed_val_metric}

    print(checkpoint_callback.best_model_path)

    model = model_class.load_from_checkpoint(checkpoint_callback.best_model_path)

    if config["test"]:
        dm.model_performances(model, trainer)

        # result_artifact = wandb.Artifact(name="RESULT_" + model_artifact_name, type="result",
        #     metadata=results[0])
        # run.log_artifact(result_artifact)

    return model