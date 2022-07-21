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

    if model_artifact_name == None:
        model_artifact_name = model_class.__str__()

    model_artifact = wandb_run.use_artifact(model_artifact_name + ":best")
    model_checkpoint_path = model_artifact.download()

    best_model = model_class.load_from_checkpoint(model_checkpoint_path)

    return best_model


def Lurz_dataset_preparation_function(config, run=None):
    """
    Gets config, can edit it.
    Returns Pytorch Lightning DataModule
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


def Antolik_dataset_preparation_function_test(config, run=None):
    """
    Gets config, can edit it.
    Uses only test dataset (for both train and test), therefore
        the loading is much quicker.
    Returns Pytorch Lightning DataModule
    """

    path_train = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik/one_trials.pickle"
    path_test = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik/ten_trials.pickle"

    dataset_config = {
        "train_data_dir": path_test,
        "test_data_dir": path_test,
        "batch_size": config["batch_size"],
        "normalize": True,
        "val_size": 500,
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
        }
    )

    return dm


def Antolik_dataset_preparation_function(config, run=None):
    """
    Gets config, can edit it.
    Returns Pytorch Lightning DataModule
    """

    path_train = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik/one_trials.pickle"
    path_test = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik/ten_trials.pickle"

    dataset_config = {
        "train_data_dir": path_train,
        "test_data_dir": path_test,
        "batch_size": config["batch_size"],
        "normalize": True,
        "val_size": 5000,
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
    # **config,
):

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

    # Set up model
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

    # define trainer
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
    metadata = config | {"best_model_score": best_observed_val_metric}

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
