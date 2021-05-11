# i-asap

## Quickstart

1. create new experiment directory in `experiements` 
```
python make_experiement.py "hello_world"
cd experiments/hello_world
```
1. run `python ../../src/train.py` to train the model according to `train_config.json`
1. run `python ../../src/evaluate.py` to evaluate the model accoring to `eval_config.json`
1. run `python ../../src/plot_results.py` to evaluate the model accoring to the config

---
**NOTE**

The config files for training and evaluating have exactly the same fields but
normally different values in those fields. For example, the chips sizes might
be different for evaluation than for training. 

## Training
Optional command line arguments for training (defaults shown):

* --config=./train_config.json
* --epochs=60

## Evaluating
Optional command line arguments for evaluating (defaults shown):

* --config=./eval_config.json
* --th=0.01 
  Not sure what this does

* --truthfile=None
  expects a `*.npy` file with frame level truth data. If not provided, the plotting
  of results will be limited to only a graph of calculated anomaly scores per frame

* --outfile=""
  if empty (default) the name will be automatically generated using the 
  `anomaly_context` value and the number of chips in the x and y dimensions.

  The output file is a `.pickle` file which is the input to the `plot_results.py` 
  script

* --anomaly_context="all"
  There are currently 3 options for this which deteremine how an anomaly is defined
  (or in what context).
    - all: every chip of every frame
    - chips: like chips from every frame (only compare, say, chip 1 across all frames)
    - chips_video: like chips within a single video


## Plotting
Command line arguments for plotting the results stored in `*.pickle` files (defaults shown):

* --infile=""
  If not defined, will used the newest `*/pickle` file in the current directory

* --all
  A flag which will loop through and plot results for all `*.pickle` files in the current directory

* --frame_range=(45,70)
  The index range of frames to plot with overlayed anomaly_scores. A large number will
  create potentially problematically large html files
