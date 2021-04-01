import numpy as np
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
# import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
import datetime 
from data import ChipDataLoader
#from other.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
import sklearn.metrics as metrics
from utils import *
from model import *
import random
import glob
import shutil
import pickle
from distutils.dir_util import copy_tree
from tqdm import tqdm

from bokeh.models import BoxAnnotation
from bokeh.plotting import figure, output_file, show, curdoc
from bokeh.layouts import gridplot, layout

import argparse

def evaluate_model(config, truth_dir="./../../data", th=0.01):
    torch.manual_seed(2020)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus']
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance


    # calculate channel numbers and related
    n_channel = 3 if config['image']['color'] else 1
    indx_of_inflection = (config['t_length']-1) * n_channel

    #model_dir = os.path.join(model_directory, config['dataset_type'], "log")
    model_dir = config['model_dir']
    test_dir = os.path.join(config['dataset_path'], config['dataset_type'], "testing", "frames")

    # Loading dataset
    img_size = (config["image"]["size_x"], config["image"]["size_y"])
    win_size = (config["window"]["size_x"], config["window"]["size_y"])
    win_step = (config["window"]["step_x"], config["window"]["step_y"])

    test_dataset = ChipDataLoader(test_dir, transforms.Compose([transforms.ToTensor(),]), img_size, win_size, win_step, time_step=config['t_length']-1, color=config['image']['color'])
    test_size = len(test_dataset)



    test_batch = data.DataLoader(test_dataset, batch_size=1, 
                                 shuffle=False, num_workers=config['num_workers_test'], drop_last=False)

    loss_func_mse = nn.MSELoss(reduction='none')

    # Loading the trained model
    model = torch.load(os.path.join(model_dir, "model.pth"))
    model.cuda()
    m_items = torch.load(os.path.join(model_dir, "keys.pt"))

    labels = np.load(os.path.join(truth_dir, 'frame_labels_'+config['dataset_type']+'.npy'))
    if config['dataset_type'] == 'shanghai':
        labels = np.expand_dims(labels, 0)

    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_dir, '*')))
    for video in videos_list:
        video_name = video.split('/')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.tif'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])

    labels_list = []
    video_length = 0
    psnr_list = {}
    feature_distance_list = {}

    print('----------------- INFO --------------------')
    print(f'test dataset:  {test_dir}')
    print(f'model:         {model_dir}')
    print(f'ChipDataLoader:  ')
    print(f'    img_size:    {test_dataset.img_size}')
    print(f'    chip_size:   {test_dataset.win_size}')
    print(f'    chip_stride: {test_dataset.step_size}')
    print(f'    chips/frame: {test_dataset.chips_per_frame()}')
    print('-------------------------------------------\n')

    # Setting for video anomaly detection
    for video in videos_list:
        video_name = video.split('/')[-1]
        labels_list = np.append(labels_list, labels[0][config['t_length']-1+video_length:videos[video_name]['length']+video_length])
        video_length += videos[video_name]['length']
        psnr_list[video_name] = []
        feature_distance_list[video_name] = []

    # video_length = 0
    # video_num = 0
    # video_length += videos[videos_list[video_num].split('/')[-1]]['length']
    m_items_test = m_items.clone()

    model.eval()

    #print(f"length of dataset = {len(test_dataset)}")
    current_video = ""
    frames_per_video = 0
    frame = 0
    chips_per_frame = test_dataset.chips_per_frame()
    chip = 0
    for k,imgs in enumerate(tqdm(test_batch)):
        imgs = Variable(imgs).cuda()

        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:indx_of_inflection], m_items_test, False)
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,indx_of_inflection:]+1)/2)).item()
        mse_feas = compactness_loss.item()

        # Save normal data for next loop.
        imgs_prev = imgs
        updated_feas_prev = updated_feas
        outputs_prev = outputs

        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs[:,indx_of_inflection:])

        # Decide if we need to update the memory.
        if point_sc < th:
            query = F.normalize(feas, dim=1)
            query = query.permute(0,2,3,1) # b X h X w X d
            m_items_test = model.memory.update(query, m_items_test, False)
        else:
            a = 0

        # Update PSNR and distance lists.
        new_video = test_dataset.get_video(k)
        if (new_video != current_video):
            frames_per_video = videos[new_video]['length']-(config['t_length']-1)
            chips_by_frames = (chips_per_frame, frames_per_video)
            psnr_list[new_video] = np.zeros(chips_by_frames)
            feature_distance_list[new_video] = np.zeros(chips_by_frames)
            current_video = new_video

        psnr_list[current_video][chip, frame] = psnr(mse_imgs)
        feature_distance_list[current_video][chip, frame] = mse_feas

        # increment to if chip < chips_per_frame-1 else 0
        if chip < chips_per_frame-1:
            chip +=1
        else:
            chip = 0
            frame = frame+1 if frame < frames_per_video-1 else 0


    return psnr_list, feature_distance_list, labels_list


def plot_results(psnr, fs, labels, output_dir="figures", alpha=0.1):
    """ Plot results in Bokeh"""
    # ceate the output directory
    #date_time_str = str(datetime.datetime.today())
    #output_dir = os.path.join(output_dir, date_time_str)
    os.makedirs(output_dir, exist_ok=True)

    chips_per_frame = np.shape(psnr[list(psnr.keys())[0]])[0]
    print(f"chips_per_frame={chips_per_frame}")
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

    # generate an output file when 'show' is called
    output_file(os.path.join(output_dir,"results.html"))

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
        p3.line(np.arange(0, change_points[-1]), anomaly_score_total_list[c, 0:change_points[-1]], line_color=colors[c], legend_label=f"Chip {c}")
    p3.legend.click_policy="hide"
    
    left_x = 0
    anomaly_color = 'red'
    for right_x in change_points:
        p3.add_layout(BoxAnnotation(left=left_x, right=right_x, fill_alpha=0.1, fill_color=anomaly_color))
        anomaly_color = 'green' if anomaly_color=='red' else 'red'
        left_x=right_x+1


    l = layout([[p1, p2], [p3],], sizing_mode='stretch_both')
    return l




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MNAD")
    parser.add_argument('--config', type=str, default='./eval_config.json', help='directory of log')
    parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
    parser.add_argument('--truth_dir', default="./../../data", type=str, help='directory of model')
    parser.add_argument('--out_file', default="eval_results.pickle", type=str, help='directory of of results')
    args = parser.parse_args()

    with open(args.config) as config_file:
        print(f"loading {args.config}")
        config = json.load(config_file)
    

    # pass test data through the model
    psnr, fs, labels = evaluate_model(config, args.truth_dir, args.th)

    with open(args.out_file, "wb") as fh:
        pickle.dump((psnr, fs, labels), fh)

    # plot the results
    l = plot_results(psnr, fs, labels, alpha=config['alpha'])
    show(l)




