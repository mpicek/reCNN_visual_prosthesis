#%%
import wandb
from Lurz_dataset import LurzDataModule
from tqdm import tqdm

# from torchsummaryX import summary
import torch
# from pytorch_lightning.utilities.model_summary import summarize
from models import Picek
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl

# use as core: RotationEquivariant2dCore from neuralopredictors.layers.cores.conv2d 
#   - this will be the core that we will eventually use in our final experiment
# use as readout: you can use the FullGaussian readout layer used by Lurz or FullFactorized2d 
#   - (same used by Cadena). If the rotational equivariant core works well we will 
#     change the readout and put the Ultrasparse
a = None

def main():
    # Set up your default hyperparameters
    config = {
        # GENERAL
        "seed": 42, #TODO .. je to treba?
        "data_name": "PicekModel",
        "batch_size": 32,
        "lr": 0.00001,
        "max_epochs": 100,

        # CORE GENERAL CONFIG
        "core_hidden_channels": 12,
        "core_layers": 3,
        "core_input_kern": 3,
        "core_hidden_kern": 3,

        # ROTATION EQUIVARIANCE CORE CONFIG
        "num_rotations": 8,        # default
        "stride": 1,               # default
        "upsampling": 2,           # default
        "rot_eq_batch_norm": True, # default
        "stack": -1,               # default
        "depth_separable": False,  # default ... TODO

        # READOUT CONFIG
        "readout_bias": True,
        "nonlinearity": "softplus",
        
        # REGULARIZATION
        "core_gamma_input": 0.01,
        "core_gamma_hidden": 0.01,
        "readout_gamma": 1,
        "input_regularizer": "LaplaceL2norm", # for RotEqCore - default 
        "use_avg_reg": True
    }


    # set config seed for everything
    pl.seed_everything(config["seed"], workers=True)

    # init wandb run
    run = wandb.init(
        config=config,
        project="rotation_equivariant",
        entity="mpicek",
    )

    # Access all hyperparameter values through wandb.config
    config = dict(config)

    # setup datamodule
    data_dir = 'data/lurz2020/static20457-5-9-preproc0'

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
            # "mean_activity": torch.zeros(dm.get_output_shape()), #dm.output_shape.mean(dim=0), TODO: spocitat.. trva
            "mean_activity": dm.get_mean_fast(), #dm.output_shape.mean(dim=0), TODO: spocitat.. trva
        }
    )

    # Set up model
    model = Picek(**config)
    # summary(model, torch.zeros((config["batch_size"], dm.get_input_shape()[0], dm.get_input_shape()[1], dm.get_input_shape()[2])))
    

    # setup wandb logger
    wandb_logger = WandbLogger(log_model=True)
    wandb_logger.watch(model, log="all", log_freq=250)

    # define callbacks for train
    early_stop = EarlyStopping(monitor="val/corr", patience=7, mode="max")

    # define trainer
    trainer = pl.Trainer(
        callbacks=[early_stop],
        max_epochs=config["max_epochs"],
        gpus=[0],
        logger=wandb_logger,
        log_every_n_steps=30,
        deterministic=True,
        enable_checkpointing=True,
    )

    trainer.fit(
        model,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )

    # predictions = trainer.test(model, dm.test_dataloader())

    test = dm.test_dataloader()
    predictions = None
    global a
    for d in tqdm(test):
        pred = model(d.images)
        if predictions == None:
            predictions = pred
        else:
            predictions = torch.cat((predictions, pred))
        # a = d
        # break
    
    a = predictions
    

if __name__ == "__main__":
    main()