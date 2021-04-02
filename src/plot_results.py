from bokeh.layouts import row, widgetbox
from bokeh.models import Label, Select
from bokeh.plotting import figure, output_file, show, curdoc, save
import pickle
from evaluate import *

def plot_results(psnr, fs, labels, output_dir=".", alpha=0.1):
    """ Plot results in Bokeh"""
    # ceate the output directory
    #date_time_str = str(datetime.datetime.today())
    #output_dir = os.path.join(output_dir, date_time_str)
    os.makedirs(output_dir, exist_ok=True)

    chips_per_frame = np.shape(psnr[list(psnr.keys())[0]])[0]
    anomaly_score_total_list = [[] for i in range(0, chips_per_frame)]
    
    for c in range(0, chips_per_frame):
        for video in sorted(list(psnr.keys())):
            video_name = video.split('/')[-1]
            anomaly_score_total_list[c] += score_sum(anomaly_score_list(psnr[video_name][c,:]), 
                    anomaly_score_list_inv(fs[video_name][c,:]), alpha)

    anomaly_score_total_list = np.asarray(anomaly_score_total_list)

    # do max across chip dimension. this sets the anomaly score for each 
    # frame as the max score from all of the chips of that frame
    anomaly_score_max = np.max(anomaly_score_total_list, axis=0)
    print(f"labels length={len(labels)} other length = {len(anomaly_score_max)}")

    accuracy = AUC(anomaly_score_max, np.expand_dims(1-labels, 0))

    # Generate PRC.
    p, r, thresholds = metrics.precision_recall_curve(1 - labels, anomaly_score_max)
    f1_scores = 2*p*r/(p + r)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    # Generate ROC curve.
    fpr, tpr, thresholds = metrics.roc_curve(1 - labels, anomaly_score_max)


    # Plot ROC
    p1 = figure(title="ROC Curve", x_axis_label="False Positive Rate", y_axis_label="True Positive Rate")
    p1.line(fpr, tpr)
    #p1.circle(fpr[best_idx], tpr[best_idx], size=12, fill_color="red", line_color="red", legend_label="Best F1 Score")

    # Plot PRC
    p2 = figure(title="PRC Curve", x_axis_label="Recall", y_axis_label="Precision")
    p2.line(r, p, legend_label="RPC")
    p2.circle(r[best_idx], p[best_idx], size=12, fill_color="red", line_color="red", legend_label="Best F1 Score")

    # Plot Normality
    change_points = np.where(np.diff(1 - labels) != 0)[0]
    p3 = figure(title="Normality", x_axis_label="Time", y_axis_label="Normality Score")
    colors=['red', 'blue', 'green', 'orange']
    for c in range(chips_per_frame):
        p3.line(np.arange(0, change_points[-1]), anomaly_score_total_list[c, 0:change_points[-1]], legend_label=f"Chip {c}")
    p3.legend.click_policy="hide"
    
    left_x = 0
    anomaly_color = 'red'
    for right_x in change_points:
        p3.add_layout(BoxAnnotation(left=left_x, right=right_x, fill_alpha=0.1, fill_color=anomaly_color))
        anomaly_color = 'green' if anomaly_color=='red' else 'red'
        left_x=right_x+1


    l = layout([[p1, p2], [p3],], sizing_mode='stretch_both')
    return l


def plot_anomalous_frames(config, psnr, fs, labels, ai, max_num=100, alpha=0.1):
    """  """
    # create the dataset
    test_dir = os.path.join(config['dataset_path'], config['dataset_type'], "testing", "frames")
    img_size = (config["image"]["size_x"], config["image"]["size_y"])
    win_size = (config["window"]["size_x"], config["window"]["size_y"])
    win_step = (config["window"]["step_x"], config["window"]["step_y"])

    dataset = ChipDataLoader(test_dir, transforms.Compose([transforms.ToTensor(),]), img_size, win_size, win_step, time_step=config['t_length']-1, color=config['image']['color'])

    # Build the anomaly score for each frame
    chips_per_frame = np.shape(psnr[list(psnr.keys())[0]])[0]
    anomaly_score_total_list = [[] for i in range(0, chips_per_frame)]
    for c in range(0, chips_per_frame):
        for video in sorted(list(psnr.keys())):
            video_name = video.split('/')[-1]
            anomaly_score_total_list[c] += score_sum(anomaly_score_list(psnr[video_name][c,:]), 
                    anomaly_score_list_inv(fs[video_name][c,:]), alpha)

    anomaly_score_total_list = np.asarray(anomaly_score_total_list)

    # iterate over all of the chips and get the anomalous frames
    cpf = dataset.chips_per_frame()
    plots = []
    row = []
    for i, k in enumerate(ai[cpf*200::]):
        if i > max_num: 
            plots.append(row)
            break

        chip_index = k % cpf

        if chip_index == 0:
            new_path,frame = dataset.get_frame(k)
            p = figure(plot_width=300, plot_height=300)
            p.xgrid.visible = False
            p.ygrid.visible = False 
            p.image(image=[np.flipud(frame)], x=0, y=0, dw=256, dh=256, level="image")

        frame_index = k//cpf
        c = int(255*anomaly_score_total_list[chip_index,frame_index])
        x1,x2,y1,y2 = dataset.get_chip_indices(k)
        p.add_layout(BoxAnnotation(left=x1, right=x2, bottom=y1, top=y2, fill_alpha=0.6, fill_color=(c, 0, 0)))
        if chip_index == cpf-1: 
            row.append(p)

        if len(row) == 4:
            plots.append(row)
            row = []

    l = gridplot(plots)
    
    return l

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a bokeh server to display certain results")
    parser.add_argument('--file', type=str, required=True, help='the result pickle file')
    parser.add_argument('--config', type=str, default='./eval_config.json', help='directory of log')
    parser.add_argument('--show', help='whether or not to show the plots')
    args = parser.parse_args()
    
    # load config
    with open(args.config) as config_file:
        print(f"loading {args.config}")
        config = json.load(config_file)

    # load results
    with open(args.file, "rb") as fh:
        psnr, fs, labels, anomalies = pickle.load(fh)

    # plot results
    p1 = plot_results(psnr, fs, labels, alpha=config['alpha'])
    save(p1, filename="results.html", title="Results")
    if args.show is not None:
        show(p1)

    # plot anomalies
    p2 = plot_anomalous_frames(config, psnr, fs, labels, anomalies)
    save(p2, filename="anomalous_frames.html", title="Anomalies")
    if args.show is not None:
        show(p2)

    

