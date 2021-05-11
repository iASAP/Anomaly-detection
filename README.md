# i-asap

## Quickstart

1. create new experiment directory in `experiements` 
```
python make_experiement.py "hello_world"
cd experiments/hello_world
```
1. run `python ../../src/train.py` to train the model according to `train_config.json`
1. run `python ../../src/evaluate.py` to evaluate the model accoring to `eval_config.json`
1. run `python ../../src/plot_results.py --infile={eval_pickle_file}` to evaluate the model accoring to the config

---
**NOTE**

The config files for training and evaluating have exactly the same fields but
normally different values in those fields. For example, the chips sizes might
be different for evaluation than for training.

---

## Plotting
TODO

