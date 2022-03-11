#!/usr/bin/env bash
# VARIABLES PASSED TO THIS SCRIPT
#   - SEED ... seed to work with
#   - CMD  ... command to be run
#   - ENSEMBLE ... indicating whether we are creating an ensemble or not

# DEFINE RESOURCES:
#PBS -N qsub_script
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:scratch_local=40gb
#PBS -q gpu
#PBS -l walltime=1:00:00

# Directory I use as a main storage
DATADIR="/storage/budejovice1/home/$(whoami)"

# test if scratch directory is set
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

# Prepare scratch directory for singularity
chmod 700 $SCRATCHDIR
mkdir $SCRATCHDIR/tmp
export SINGULARITY_TMPDIR=$SCRATCHDIR/tmp


cd $SCRATCHDIR

# Create a script that will be run in a NGC container with Singularity
echo "source activate csng-dl" > my_new_script.sh
echo "cd /storage/budejovice1/home/mpicek/reCNN_visual_prosthesis" >> my_new_script.sh
echo "$CMD" >> my_new_script.sh

# --nv for gpu, bind scratch directory
singularity exec --nv -B $SCRATCHDIR /storage/budejovice1/home/mpicek/csng-dl.img bash my_new_script.sh

# print what command has been run and print the output of the program
echo "$CMD"

# if we are creating an ensamble, I want to copy the created model to a specific
# folder - ensure it exists (or create it) and copy the submodel$SEED folder
if [ "$ENSEMBLE" -eq "1" ]; then
    mkdir -p $DATADIR/models && cp -r submodel$SEED $DATADIR/models/submodel$SEED
fi

clean_scratch