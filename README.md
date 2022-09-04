# Rotation-equivariant convolutional neural network for design of visual prosthetic stimulation protocol

The code for Martin Picek's bachelor thesis supervised by Luca Baroni and Ján Antolík

To clone the repository:
```bash
git clone --recurse-submodules git@github.com:mpicek/reCNN_visual_prosthesis.git
```

Bachelor thesis is in [this github repo](https://github.com/mpicek/bachelor_thesis).

## Running the network

Use a Docker image from [this repository](https://github.com/mpicek/csng_dl_docker_image).
It can be obtained from the Docker Hub [here](https://hub.docker.com/repository/docker/picekma/csng_docker_dl/general) - more on instalation in the previous repository.

Run the image locally:
```
docker run --gpus all -it --rm -v local_dir:$(pwd) picekma/csng_docker_dl:0.1
```
Or on MetaCentrum:
```
singularity shell --nv -B $SCRATCHDIR /path/to/the/image.img
```
where you have to specify your path to a builded Singularity container. The build is
described in the repository with the Docker file.

In the container, execute `source activate csng-dl` in order to activate conda environment.

Then run `python train_on_lurz.py`, the network starts a training.

## Run the network

Run `python final_network_lurz.py` or `python final_network_antolik.py` based on
the dataset on which the network was trained. Ensemble networks are also run in
these scripts.

## Creating and running a sweep

Connect to MetaCentrum, clone this repository, build a Singularity image
of the docker image provided by us (previous section) and specify the path
to this image in `metacentrum/qsub_script.sh` as well as path to this repository.

Add your wandb API key to file `metacentrum/wandb_api_key.yaml` in this format:
```yaml
WANDB_API_KEY: your_api_key
```
Your wandb API key can be found in your [wandb settings](https://wandb.ai/settings).

Configure a sweep in sweep.yaml.

Create a sweep with `wandb sweep sweep.yaml` and copy the command into `metacentrum/cmd`
so that the file looks like this (for example):
```
# run the same command again and again
wandb agent csng-cuni/reCNN_visual_prosthesis/6ggort1b
```

To run 3 machines on MetaCentrum that connect as sweep agents, use this
command.
```bash
python3 ./run_commands.py --command_file=cmd --script=qsub_script.sh --wandb_api_key --num_of_command_repetitions=3
```
