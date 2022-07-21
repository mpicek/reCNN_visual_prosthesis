import wandb
import glob
import pytorch_lightning as pl
from models import reCNN_bottleneck_CyclicGauss3d
import matplotlib.pyplot as plt


# run = wandb.init(project="reCNN_visual_prosthesis", entity="csng-cuni")
# pl.seed_everything(42)

# artifact = run.use_artifact(
#     "csng-cuni/reCNN_visual_prosthesis/model-hgwjlskb:v0", type="model"
# )

# artifact_dir = artifact.download()

# models_paths_list = glob.glob(artifact_dir + "/*.ckpt")
# m = None
# for path in models_paths_list:
#     m = reCNN_bottleneck_CyclicGauss3d.load_from_checkpoint(path)
#     print(f"Model from {path} loaded!")

def return_number_of_parameters(model, part_of_network="all"):

    if part_of_network == "all":
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif part_of_network == "core":
        return sum(p.numel() for p in model.core.parameters() if p.requires_grad)
    elif part_of_network == "readout":
        return sum(p.numel() for p in model.readout.parameters() if p.requires_grad)

    # print(f"Number of parameters in the core: {parameters_core}")
    # print(f"Number of parameters in the readout: {parameters_readout}")
    # print(f"Number of parameters in ther readout per neuron is: {parameters_readout / 30000}")

