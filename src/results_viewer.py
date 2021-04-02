from evaluate import *
from bokeh.layouts import row, widgetbox
from bokeh.models import Label, Select
from bokeh.plotting import figure, output_file, show, curdoc
import pickle

parser = argparse.ArgumentParser(description="Start a bokeh server to display certain results")
parser.add_argument('--file', type=str, required=True, help='the result pickle file')
args = parser.parse_args()

with open(args.file, "rb") as fh:
    psnr, fs, labels = pickle.load(fh)
l = plot_results(psnr, fs, labels)
curdoc().add_root(l)


# app_data = {}

# select = Select(title="Select bulk-file:")
# select.on_change('value', update_results)

# def update_results(attr, old, new):
#     with open(new, "rb") as fh:
#         print(f"loading {new}")
#         psnr, fs, labels = pickle.load(fh)

#     l = plot_results(psnr, fs, labels, alpha=config['alpha'])

#     app_data['psnr'] = psnr
#     app_data['fs'] = fs
#     app_data['labels'] = labels

# # plot the results
# curdoc().add_root(layout(column(select, l)))
