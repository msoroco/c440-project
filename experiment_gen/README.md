# How to use make_runner.py

<br />

## JSON file

<br />

To make a experiment runner we need to make a JSON that we will feed to ```make_runner.py``` to generate the runner.

The JSON will have the following form:

```json
{
  "grid": {"combine": [
                {"noise_multiplier": [2]},
                {"join": [
                    {"optimizer": [2],
                     "dataset": [1]},
                    {"combine": [
                        {"eps": [1]},
                        {"lr": [0.3, 0.4]}
                    ]}
                ]}]
          },
  "test_stepsize": 10
}
```

The ```"grid"``` field indicates that we want to create a grid from the following settings. We use two operations: ```"combine"``` and ```"join"```. The ```"combine"``` operation will take all combinations of the specified configurations. The ```"join"``` operation will join the configurations. If some hyperparameter is not present in a configuration that is being joined, we just use the default value indicated by ```"N/a"``` . So, the result will be:

```json
{"noise_multiplier": [2, 2, 2], "optimizer": [2, "N/a", "N/a"], "dataset": [1, "N/a", "N/a"], "eps": ["N/a", 1, 1], "lr": ["N/a", 0.3, 0.4]}
```

This is a total of 4 experiments. Each experiment takes a different ```"degree_bound"```, ```"r_hop"```, ```"setup"```, and ```"lr"```. The first experiment will take the first element of each list and set that to the corresponding hyperparameter.

<br />

## JSON fields

<br />

Valid JSON fields include:
 - ```"grid"``` accepts a series of operations (JSON object) which are used to construct the grid used for experiments
 - ```"combine"``` accepts a list of configurations/operations (JSON objects) from a which a combined configuration of all possible combinations will be composed
 - ```"join"``` accepts a list of configurations/operations (JSON objects) from a which a joined configuration will be composed
 - ```"batch_size"``` accepts an int for the batch size
 - ```"gamma"``` accepts float for discount factor
 - ```"eps_start"``` start value of epsilon
 - ```"eps_end"``` end value of epsilon
 - ```"eps_decay"``` epsilon decay rate
 - ```"tau"``` soft update weights of target net
 - ```"lr"``` learning rate
 - ```"episodes"``` number of episodes
 - ```"max_steps"``` maximum number of steps per episode
 - ```"simulation"``` simulation json
 - ```"draw_neighbourhood"``` whether or not to draw the neighbourhood is animation
 - ```"test"``` run model in test mode
 - ```"animate"``` animate the run
 - ```"wandb_project"``` name of wandb project
 - ```"experiment_name"``` name of experiment as it will be known in wandb
 - ```"model"``` model file for its parameters (in models)
 - ```"iter"``` number of iterations to do each experiment

<br />

## Generating bash script

<br />

To generate the runner bash script, feed the JSON to ```make_runner.py```, run the command:

```$ python make_runner.py <path-to-json> --out_path "./run.sh"```

Please note that the ```out_path``` filepath is with respect to ```make_runner.py``` and the ```results_path``` filepath is with respect to ```main.py```. Now you can run the bash script ```run.sh``` to do your experiments!