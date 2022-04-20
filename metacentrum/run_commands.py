#!/usr/bin/env python3
"""
Runs specified commands or creates an ensamble.
"""

import sys
import os
import argparse
import subprocess
import yaml

def parse(filename):
    """
    Parses the commands from a file:
        - ignores # (=comments) 
        - each command has to be separated either by a newline (at least one) 
        or # (a comment)
    
    Returns:
        List of commands to be run.
    """

    commands = []
    command = ""
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                if len(command) > 0:
                    commands.append(command)
                command = ""
                continue

            if line[0] == "#":
                if len(command) > 0:
                    commands.append(command)
                command = ""
                continue

            command += " " + line

        if len(command) > 0:
            commands.append(command)        

    return commands


def main(args):
    if args.script == None:
        raise ValueError("No script defined.")
    
    wandb_api_kdy = ""
    if args.wandb_api_key:
        config = yaml.safe_load(open("wandb_api_key.yaml"))
        if "WANDB_API_KEY" in config:
            wandb_api_kdy = ',WANDB_API_KEY="' + config["WANDB_API_KEY"] + '"'

    commands = parse(args.command_file)

    if args.ensemble:
        # if creating an ensemble, we use just the first command
        command = commands[0]

        for i in range(args.models):
            seed = i + 1 # I don't know whether seed can be 0
            command_with_seed = command
            seed_var = 'SEED=' + str(seed)
            cmd_var = ',CMD="' + command_with_seed + '"'
            ensemble_var = ',ENSEMBLE="' + str(1) + '"'
            variables = seed_var + cmd_var + ensemble_var + wandb_api_kdy
            print(variables)
            list_files = subprocess.run(['qsub', '-v', variables, args.script])

    else:
        for command in commands:
            seed = 42
            command_with_seed = command
            seed_var = 'SEED=' + str(seed)
            cmd_var = ',CMD="' + command_with_seed + '"'
            ensemble_var = ',ENSEMBLE="' + str(0) + '"'
            variables = seed_var + cmd_var + ensemble_var + wandb_api_kdy
            print(variables)
            for i in range(args.num_of_command_repetitions):
                list_files = subprocess.run(['qsub', '-v', variables, args.script])

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command_file", default="commands", type=str, help="File with commands to be run.")
    parser.add_argument("--script", default=None, type=str, help="File with a qsub script.")
    parser.add_argument("--ensemble", action='store_true', help="Create ensemble.")
    parser.add_argument("--wandb_api_key", action='store_true', help="Read wandb_api_key.yaml file and set an ENV var with it.")
    parser.add_argument("--models", default=10, type=int, help="Number of models in an ensemble.")
    parser.add_argument("--num_of_command_repetitions", default=1, type=int, help="Number of repetitions of each command in the command_file..")

    args = parser.parse_args()
    main(args)
