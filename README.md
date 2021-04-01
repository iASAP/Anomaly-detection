# i-asap

## Quickstart

1. create new run directory in `runs`
1. run `python ../../src/train.py` to train the model according to the config
1. run `python ../../src/evaluate.py` to evaluate the model accoring to the config
1. run `bokeh serve src/results_viewer.py --allow-websocket-origin 192.168.15.6:5006 --args --file=/path/to/results.pickle
