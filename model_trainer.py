import wandb
from Lurz_dataset import LurzDataModule
from tqdm import tqdm

from predict_neural_responses.data_utils import Antolik2016Datamodule

from models import Picek
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import ProgressBar
import pytorch_lightning as pl

from pprint import pprint


def get_best_model(wandb_run, model_class=Picek, model_artifact_name="RotEq_FullGaussian2d"):


    if model_artifact_name == None:
        model_artifact_name = model_class.__str__()

    model_artifact = wandb_run.use_artifact(model_artifact_name + ":best")
    model_checkpoint_path = model_artifact.download()

    best_model = model_class.load_from_checkpoint(model_checkpoint_path)

    return best_model


def Lurz_dataset_preparation_function(config):
    """
        Gets config, can edit it.
        Returns Pytorch Lightning DataModule
    """
    # setup datamodule - use artifact
    # dataset_artifact = run.use_artifact(dataset_artifact_name)
    # data_dir = dataset_artifact.download()
    data_dir = 'data/lurz2020/static20457-5-9-preproc0'

    #TODO: add artifact
    dataset_config = {"data_dir": data_dir, 
                      "batch_size": config["batch_size"], 
                      "normalize": True, 
                      "exclude": "images"}


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
            "mean_activity": dm.get_mean(), #dm.output_shape.mean(dim=0), TODO: spocitat.. trva
        }
    )

    return dm


def Antolik_dataset_preparation_function(config):
    """
        Gets config, can edit it.
        Returns Pytorch Lightning DataModule
    """

    dm = Antolik2016Datamodule(
        region=config["region"], batch_size=config["batch_size"], with_test_dataset=False
    )
    dm.setup()

    config.update(
        {
            "input_channels": dm.train_dataset[:][0].shape[1],
            "input_size_x": dm.train_dataset[:][0].shape[2],
            "input_size_y": dm.train_dataset[:][0].shape[3],
            "num_neurons": dm.train_dataset[:][1].shape[1],
            "mean_activity": dm.train_dataset[:][1].mean(dim=0),
        }
    )

    return dm


def run_wandb_training(
        config,
        dataset_preparation_function,
        entity, 
        project,
        model_artifact_name = None,
        dataset_artifact_name="Lurz_dataset:latest", 
        model_class=Picek, 
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

    dm = dataset_preparation_function(config)

    # Set up model
    model = model_class(**config)

    # summary(model, torch.zeros((config["batch_size"], dm.get_input_shape()[0], dm.get_input_shape()[1], dm.get_input_shape()[2])))

    # setup wandb logger
    wandb_logger = WandbLogger(log_model=True)
    wandb_logger.watch(model, log="all", log_freq=250)

    # define callbacks for the training
    early_stop = EarlyStopping(monitor=early_stopping_monitor, patience=config["patience"], mode=early_stopping_mode)
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor=model_checkpoint_monitor, mode=model_checkpoint_mode)

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

    best_observed_val_metric = checkpoint_callback.best_model_score.cpu().detach().numpy()
    print("Best model's " + config["observed_val_metric"] + ": " + str(best_observed_val_metric))
    
    if model_artifact_name == None:
        model_artifact_name = model.__str__()
    
    # add best corr to metadata
    metadata = config | {"best_model_score": best_observed_val_metric}

    # add model artifact
    best_model_artifact = wandb.Artifact(model_artifact_name, type="model",
        metadata=metadata)
    best_model_artifact.add_file(checkpoint_callback.best_model_path)
    run.log_artifact(best_model_artifact)

    # say to wandb that the best val/corr of the model is the best one 
    # and not the last one!! (it is the default behavour!!)
    run.summary[config["observed_val_metric"]] = best_observed_val_metric

    print(checkpoint_callback.best_model_path)

    model_class.load_from_checkpoint(checkpoint_callback.best_model_path)

    if config["test"]:
        best_model = model_class.load_from_checkpoint(checkpoint_callback.best_model_path)
        # test on the best model at all
        results = trainer.test(model=best_model, dataloaders=dm.test_dataloader())

        print(results)
        print("--------------")
        for x in results:
            print(x)
        
        result_artifact = wandb.Artifact(name="RESULT_" + model_artifact_name, type="result",
            metadata=results[0])
        run.log_artifact(result_artifact)
    
    return model_class.load_from_checkpoint(checkpoint_callback.best_model_path)