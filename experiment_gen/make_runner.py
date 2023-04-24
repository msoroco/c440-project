import os
import sys
import argparse
import json
import math

def parse_axis(axis):
    if "combine" in axis:
        return combine_axes(*axis["combine"])
    elif "join" in axis:
        return join_axes(*axis["join"])
    else:
        return axis


def combine_axes(*axes):
    combination = {}
    for axis in axes:
        current_length = 1
        if combination.values():
            current_length = len(list(combination.values())[0])
        axis = parse_axis(axis)
        # combine axis
        added_length = len(list(axis.values())[0])
        for key in combination:
            combination[key] = combination[key]*added_length
        for key in axis:
            combination[key] = [value for value in axis[key] for i in range(current_length)]
    return combination


def join_axes(*axes):
    joined = {}
    for axis in axes:
        current_length = 0
        if joined.values():
            current_length = len(list(joined.values())[0])
        axis = parse_axis(axis)
        # join axis
        added_length = len(list(axis.values())[0])
        for key in joined:
            if key not in axis:
                joined[key] += ["N/a"]*added_length
        for key in axis:
            if key not in joined:
                joined[key] = ["N/a"]*current_length
            joined[key] += axis[key]
    return joined


def parse_args(config, id):
    command = ""
    for key, value in config.items():
        if value != "N/a":
            command += " --" + key + " "
            command += str(value)
    command += " --id " + str(id)
    return command


def parse_environment_variables(json_obj):
    command = ""
    for key, value in json_obj.items():
        if key == "test" or key == "draw_neighbourhood" or key == "animate":
            if value == True:
                command += " --" + key 
        elif key != "grid" and key != "iter":
            command += " --" + key + " " + str(value)
    return command


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", help="input JSON (or previous runner script if recompute is set)", type=str)
    parser.add_argument("--out_path", help="path to runner script", default="../runner.sh", type=str)
    args = parser.parse_args()

    # parse JSON
    with open(args.in_path) as json_file:
        json_obj = json.load(json_file)
        grid = parse_axis(json_obj["grid"])
    
    # make runner
    n_configs = len(next(iter(grid.values())))
    line = 1
    iterations = 1 if "iter" not in json_obj else json_obj["iter"]

    # open files
    command_file = open(args.out_path, 'w')

    # write to files
    for i in range(n_configs):
        print(str(i+1) + "/" + str(n_configs), end="\r")  
        # get config for this experiment (i)
        config = {}
        for key, value in grid.items():
            config[key] = value[i]
        # parse command for runner
        command = "python main.py" + parse_args(config, i+1) + parse_environment_variables(json_obj)
        # write command
        command_file.write(iterations * (command + "\n"))

    # close files
    command_file.close()