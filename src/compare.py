import sys
sys.path.append("../../src")
from evaluate import *
from plot_results import *

def compare_anomaly_plots(pickles = glob.glob("*.pickle")):
    plots = []
    for f in pickles:
        with open(f, "rb") as fh:
            normality_scores, labels, config = pickle.load(fh)

        anomaly_score_max = np.max(1-normality_scores, axis=0)
        accuracy = AUC(anomaly_score_max, np.expand_dims(labels, 0))
        
        p1, p2, p3, info = plot_results(normality_scores, labels, config)
        p3.title.text = f"{f}  - AUC = {accuracy}"
        plots.append(p3)


    l = column(plots, sizing_mode="stretch_both")
    save(l, filename="anomaly method comparison.html", title="evaluate comparison")

    make_html_index()


if __name__ == "__main__":
    compare_anomaly_plots(pickles=["4x4_all.pickle", "4x4_chips.pickle", "4x4_chips_video.pickle"])
    
    # chips = [1, 2, 4]
    # # heights = [1, 2, 4]
    # plots = []

    # for train_c in [1,2]:
    #     for eval_c in [1,2,4]:
    #         f = f"./../train_{train_c}x{train_c}/{eval_c}x{eval_c}_eval.pickle"
    #         with open(f, "rb") as fh:
    #             normality_scores, labels, config = pickle.load(fh)

    #         anomaly_score_max = np.max(1-normality_scores, axis=0)
    #         accuracy = AUC(anomaly_score_max, np.expand_dims(labels, 0))
            
    #         p1, p2, p3, info = plot_results(normality_scores, labels, config)
    #         p3.title.text = f"train_{train_c}x{train_c}/eval_{eval_c}x{eval_c}  - AUC = {accuracy}"
    #         plots.append(p3)


    # l = column(plots, sizing_mode="stretch_both")
    # save(l, filename="video evaluation comparison.html", title="evaluate comparison")


