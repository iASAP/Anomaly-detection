from bokeh.layouts import row, widgetbox, column, Spacer
from pathlib import Path
import bokeh.models as bkm
import bokeh.palettes as bp
from bokeh.plotting import figure, output_file, show, curdoc, save
from bokeh.io import export_png
import pickle
from evaluate import *
from itertools import accumulate



def plot_results(anomaly_scores, labels, config, output_dir="."):
    """ Plot results in Bokeh"""
    # ceate the output directory
    #date_time_str = str(datetime.datetime.today())
    #output_dir = os.path.join(output_dir, date_time_str)
    os.makedirs(output_dir, exist_ok=True)

    # create the dataset
    test_dir = os.path.join(config['dataset_path'], config['dataset_type'], "testing", "frames")
    img_size = (config["image"]["size_x"], config["image"]["size_y"])
    win_size = (config["window"]["size_x"], config["window"]["size_y"])
    win_step = (config["window"]["step_x"], config["window"]["step_y"])

    cdl = ChipDataLoader(test_dir, transforms.Compose([transforms.ToTensor(),]), img_size, win_size, win_step, time_step=config['t_length']-1, color=config['image']['color'])

    # do max across chip dimension. this sets the anomaly score for each 
    # frame as the max score from all of the chips of that frame
    anomaly_score_max = np.max(anomaly_scores, axis=0)
    print(f"labels length={len(labels)} other length = {len(anomaly_score_max)}")

    accuracy = AUC(anomaly_score_max, np.expand_dims(1-labels, 0))

    # Generate PRC.

    # Generate ROC curve.
    fpr, tpr, thresholds = metrics.roc_curve(1 - labels, anomaly_score_max)

    # Plot ROC
    # --------
    p1 = figure(title=f"ROC Curve", x_axis_label="False Positive Rate", y_axis_label="True Positive Rate")
    p1.title.text_font_size="20px"
    p1.line(fpr, tpr)
    #p1.circle(fpr[best_idx], tpr[best_idx], size=12, fill_color="red", line_color="red", legend_label="Best F1 Score")

    # Plot PRC
    # --------
    p, r, thresholds = metrics.precision_recall_curve(1 - labels, anomaly_score_max)
    f1_scores = 2*p*r/(p + r)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    p2 = figure(title="PRC Curve", x_axis_label="Recall", y_axis_label="Precision")
    p2.title.text_font_size="20px"
    p2.line(r, p, legend_label="RPC")
    p2.circle(r[best_idx], p[best_idx], size=12, fill_color="red", line_color="red", legend_label="Best F1 Score")

    # Plot Anomaly/Normality score
    # -----------------------------------
    # change_points indicate indices where anomalous segments meet normal segments and visa versa
    change_points = np.where(np.diff(1 - labels) != 0)[0]

    # get the video boundary indices to potentially plot on Anomaly/Normality plot
    video_names = list(cdl.videos.keys())
    video_start_x = list(accumulate([v['length']-(config['t_length']-1) for k,v in cdl.videos.items()]))

    # now add 0 to beginning and forget the last index. We want the starting point of each video, not the ending point
    video_start_x = [0] + video_start_x[0:-1]

    p3 = figure(title="Normality", x_axis_label="Time", y_axis_label="Normality Score", tools=[])
    p3.title.text_font_size="20px"

    # plot the anomaly score for each frame (determined by the max of frames chips)
    data = {
          "max_scores" : anomaly_score_max,
          "frames" : ['/'.join(i.split("/")[-2::]) for i in cdl.seq_stops],
          "x" : np.arange(0, len(anomaly_score_max)),
          "y" : anomaly_score_max,
          #"legend_label" : "max_anomaly_score",
          }

    #anomaly_score_plot = p3.line(np.arange(0, change_points[-1]), anomaly_score_max, legend_label="max anomaly score")
    #anomaly_line_glyph = bkm.Line(, legend_label="max anomaly score")
    anomaly_line_r = p3.add_glyph(bkm.ColumnDataSource(data), bkm.Line(line_color='#3288bd'))

    # plot the threshold
    threshold_plot = p3.segment(x0=0, y0=best_threshold, x1=len(labels), y1=best_threshold, line_color="red", legend_label="Best Threshold")

    # plot the video transitions
    y_0 = np.zeros(len(video_start_x))
    y_1 = np.ones(len(video_start_x))
    p3.segment(x0=video_start_x, y0=y_0, x1=video_start_x, y1=y_1, line_color='orange', legend_label="transitions")
    p3.text(x=video_start_x, y=y_1, text=video_names, legend_label="videos")

    #p3.toolbar.active_inspect = [hover_tool, crosshair_tool]
    # colors=['red', 'blue', 'green', 'orange']
    # for c in range(cpf):
    #     p3.line(np.arange(0, change_points[-1]), anomaly_score_total_list[c, 0:change_points[-1]], legend_label=f"Chip {c}")
    
    # plot
    left_x = 0
    color = 'green'
    for right_x in np.append(change_points, len(labels)-1):
        p3.add_layout(BoxAnnotation(left=left_x, right=right_x, fill_alpha=0.1, fill_color=color))
        color = 'green' if color=='red' else 'red'
        left_x=right_x+1
        

    hover_tool = bkm.HoverTool(renderers=[anomaly_line_r], tooltips=[("index","$index"),("frame","@frames"),("max score", "@max_scores")])
    p3.add_tools(hover_tool, bkm.BoxZoomTool(), bkm.ResetTool())
    p3.legend.click_policy="hide"



    # build a small info section
    # --------------------------
    title = bkm.Div(text="""<b>INFO</b>""", style={"font-size":"20px"})
    #title = Div(text="""<b>INFO</b>""")
    info = column(children=[
            title,
            bkm.Div(text="AUC:         {:1.2}".format(accuracy), style={"font-size":"16px"}), 
            bkm.Div(text=f"img_size:    {config['image']['size_x']}x{config['image']['size_y']}", style={"font-size":"16px"}),
            bkm.Div(text=f"chip_size:    {config['window']['size_x']}x{config['window']['size_y']}", style={"font-size":"16px"}),
            bkm.Div(text=f"chip_stride:    {config['window']['step_x']}x{config['window']['step_y']}", style={"font-size":"16px"}),
            bkm.Spacer(sizing_mode="stretch_height")
            ],max_width=150, sizing_mode="stretch_height")


    return p1, p2, p3, info


def plot_frames(anomaly_scores, labels, config, frame_start=1600, frame_stop=1630):
    """  """
    #ai = np.asarray(ai)

    # create the dataset
    test_dir = os.path.join(config['dataset_path'], config['dataset_type'], "testing", "frames")
    img_size = (config["image"]["size_x"], config["image"]["size_y"])
    win_size = (config["window"]["size_x"], config["window"]["size_y"])
    win_step = (config["window"]["step_x"], config["window"]["step_y"])

    dataset = ChipDataLoader(test_dir, transforms.Compose([transforms.ToTensor(),]), img_size, win_size, win_step, time_step=config['t_length']-1, color=config['image']['color'])
    cpf = dataset.chips_per_frame()
    win_w, win_h = win_size

    # Build the anomaly score for each frame
    # anomaly_score_total_list = [[] for i in range(0, cpf)]
    
    # for c in range(0, cpf):
    #     for video in sorted(list(psnr.keys())):
    #         video_name = video.split('/')[-1]
    #         anomaly_score_total_list[c] += score_sum(anomaly_score_list(psnr[video_name][c,:]), 
    #                 anomaly_score_list_inv(fs[video_name][c,:]), alpha)
    
    # anomaly_score_total_list = np.asarray(anomaly_score_total_list)
    
    # iterate over all of the chips and get the anomalous frames
    #anomaly_range = np.arange(frame_start*cpf, frame_stop*cpf)
    color_mapper = bkm.LinearColorMapper(palette='Turbo256', low=0, high=1)

    plots = []
    row = []

    # for i, k in enumerate(ai[anomaly_range]):
    for i, frame_index in enumerate(range(frame_start, frame_stop)):

        # keep track of the max chip 
        data = {
              "scores" : [],
              "x" : [],
              "y" : [],
              }

        for c in range(cpf):
            chip_score = 1-anomaly_scores[c, frame_index]
            data['scores'].append(chip_score)

            chip_index = frame_index*cpf + c
            x1,x2,y1,y2 = dataset.get_chip_indices(chip_index)
            #print(f"{x2}-{x1}  {y1}-{y2}")

            # these coordinates should be the center of the chip
            data['x'].append(x2-(win_w//2))
            data['y'].append(256- (y2-(win_h//2)))


        new_path,frame = dataset.get_frame(frame_index*cpf)
        temp = new_path.split('/')[-2::]
        p = figure(title=f"{os.path.join(temp[0], temp[1])} - {labels[frame_index]}", plot_width=300, plot_height=300,  tools=[])
        p.xgrid.visible = False
        p.ygrid.visible = False
        p.image(image=[np.flipud(frame)], dw=256, dh=256, x=0, y=0)

        #p.rect(chip_xs, chip_ys, width=win_w, height=win_h, fill_color=colors, fill_alpha=0.6, line_color=None)
        chip_glyph = bkm.Rect(width=win_w, height=win_h, fill_color={'field':'scores', 'transform':color_mapper}, fill_alpha=0.6, line_color=None)
        chip_r = p.add_glyph(bkm.ColumnDataSource(data), chip_glyph)
        
        max_index = data['scores'].index(max(data['scores']))
        p.rect(x=data['x'][max_index], y=data['y'][max_index], width=win_w, height=win_h, fill_color=None, fill_alpha=0.6, line_color='red')

        hover_tool = bkm.HoverTool(renderers=[chip_r], tooltips=[("chip","$index"),("score", "@scores")])
        p.add_tools(hover_tool)

        # add this plot to the current row of plots
        row.append(p)

        # cap each row at 5 plots 
        if len(row) == 6:
            plots.append(row)
            row = []

        # chip_index = k % cpf

        # if chip_index == 0:
        #     new_path,frame = dataset.get_frame(k)
        #     p = figure(plot_width=300, plot_height=300)
        #     p.xgrid.visible = False
        #     p.ygrid.visible = False 
        #     p.image(image=[np.flipud(frame)], x=0, y=0, dw=256, dh=256, level="image")

        # c = int(255*anomaly_score_total_list[chip_index, frame_index])
        # x1,x2,y1,y2 = dataset.get_chip_indices(k)
        # p.add_layout(BoxAnnotation(left=x1, right=x2, bottom=y1, top=y2, fill_alpha=0.6, fill_color=(c, 0, 0)))
        # if chip_index == cpf-1: 
        #     row.append(p)


    if (len(row) != 0):
        plots.append(row)


    #l = gridplot(plots)
    # color_bar_figure = figure(plot_width=300, plot_height=300)
    # color_bar = bkm.ColorBar(color_mapper=color_mapper)
    # color_bar_figure.add_layout(color_bar, "below")

    # plots.insert(0, color_bar_figure)
    l = layout(plots)
    
    return l

def make_html_index(path=".."):
    fh = open(Path(path).joinpath("index.html").resolve(), 'w')
    fh.write("<html><head></head><body><ul>")

    # write links
    for p in Path("..").rglob("*.html"):
        parts = p.parts[-2::]
        href = Path('/', parts[0], parts[1])
        fh.write(f'<li><a href="{href}" target="_blank">{href}</a></li>')

    fh.write("</ul></body></html>")
    fh.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot results from an evaluated pickle. Defaults to the latest modified pickle file in the current directory.")
    parser.add_argument('--infile', type=str, help='A specific pickle file to plot results from')
    parser.add_argument('--all', action='store_const', const=True, default=False, help="plot results from all pickle files in the current directory. Option 'infile' overrides this")
    parser.add_argument('--png', action='store_const', const=True , default=False, help='whether or not to additionally save plots as png (Currently Broken?)')
    parser.add_argument('--show', action='store_const', const=True , default=False, help='whether or not to show the plots')
    parser.add_argument('--frame_range', type=str, default="(45, 70)", help='the range of frame to plot for one of the plots')
    args = parser.parse_args()

    min_frame, max_frame = eval(args.frame_range)


    # default implementation uses the latest modified pickle 
    files_to_plot = [max(glob.glob("*.pickle"), key=os.path.getctime)]

    # use all pickle files instead
    if args.all: files_to_plot = glob.glob("*.pickle")

    # use only the specified pickle
    if args.infile: files_to_plot = [args.infile]

    # load results
    for f in files_to_plot:
        print(f"plotting results from - {f}")
        with open(f, "rb") as fh:
            anomaly_scores, labels, config = pickle.load(fh)

        # 
        w = config['image']['size_x'] // config['window']['size_x']
        h = config['image']['size_y'] // config['window']['size_y']

        # plot results
        p1, p2, p3, info = plot_results(anomaly_scores, labels, config)
        l1 = layout([[p1, p2], [info, p3],], sizing_mode='stretch_both')
        #curdoc().theme = 'dark_minimal'
        save(l1, filename=f"{w}x{h}_results.html", title="Results")
        if args.png: export_png(l1, filename=f"{w}x{h}_results.png")
        if args.show: show(l1)

        # plot anomalies
        p4 = plot_frames(anomaly_scores, labels, config, min_frame, max_frame)
        save(p4, filename=f"{w}x{h}_frames.html", title="Anomalies")
        if args.png: export_png(p4, filename=f"{w}x{h}_frames.png")
        if args.show: show(p4)

    #TODO: create html file
    make_html_index()


    

