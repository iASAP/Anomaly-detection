# i-asap

## Quickstart

1. create new run directory in `runs` (e.g. `runs/run1`)
1. run `python ../../src/train.py` to train the model according to the config
1. run `python ../../src/evaluate.py` to evaluate the model accoring to the config
1. run `python ../../src/plot_results.py --infile={eval_pickle_file}` to evaluate the model accoring to the config

## Training
TODO

## Evaluating
TODO

## Plotting
TODO

1. run `bokeh serve src/results_viewer.py --allow-websocket-origin 192.168.15.6:5006 --args --file=/path/to/results.pickle
