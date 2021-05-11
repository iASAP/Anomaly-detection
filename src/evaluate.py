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
import pickle
from distutils.dir_util import copy_tree
from tqdm import tqdm

from bokeh.models import BoxAnnotation
from bokeh.plotting import figure, output_file, show, curdoc
from bokeh.layouts import gridplot, layout

import argparse

def evaluate_model(config, anomaly_context="all",  th=0.01):
    torch.manual_seed(2020)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus']
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance


    # calculate channel numbers and related
    n_channel = 3 if config['image']['color'] else 1
    indx_of_inflection = (config['t_length']-1) * n_channel

    #model_dir = os.path.join(model_directory, config['dataset_type'], "log")
    model_dir = config['model_dir']
    #test_dir = os.path.join(config['dataset_path'], config['dataset_type'], "testing", "frames")

    # Loading dataset
    img_size = (config["image"]["size_x"], config["image"]["size_y"])
    win_size = (config["window"]["size_x"], config["window"]["size_y"])
    win_step = (config["window"]["step_x"], config["window"]["step_y"])

    test_dataset = ChipDataLoader(config['dataset_path'], transforms.Compose([transforms.ToTensor(),]), img_size, win_size, win_step, time_step=config['t_length']-1, color=config['image']['color'], ext=config['extension'])
    test_size = len(test_dataset)

    # NOTE: currently, I think batch_size must be 1 for evaluation
    #test_batch = data.DataLoader(test_dataset, batch_size=config['batch_size'], 
    test_batch = data.DataLoader(test_dataset, batch_size=1, 
                                 shuffle=False, num_workers=config['num_workers_test'], drop_last=False)

    loss_func_mse = nn.MSELoss(reduction='none')

    # Loading the trained model
    model = torch.load(os.path.join(model_dir, "model.pth"))
    model.cuda()
    m_items = torch.load(os.path.join(model_dir, "keys.pt"))

    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(config['dataset_path'], '*')))
    for video in videos_list:
        video_name = video.split('/')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.tif'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])

    video_length = 0
    psnr_list = {}
    feature_distance_list = {}

    print('----------------- INFO --------------------')
    print(f'test dataset:  {config["dataset_path"]}')
    print(f'model:         {model_dir}')
    print(f'anomaly_context:   {anomaly_context}')
    print(f'ChipDataLoader:  ')
    print(f'    img_size:    {test_dataset.img_size}')
    print(f'    chip_size:   {test_dataset.win_size}')
    print(f'    chip_stride: {test_dataset.step_size}')
    print(f'    chips/frame: {test_dataset.chips_per_frame()}')
    print('-------------------------------------------\n')

    # Setting for video anomaly detection
    for video in videos_list:
        video_name = video.split('/')[-1]
        video_length += videos[video_name]['length']
        psnr_list[video_name] = []
        feature_distance_list[video_name] = []

    # video_length = 0
    # video_num = 0
    # video_length += videos[videos_list[video_num].split('/')[-1]]['length']
    m_items_test = m_items.clone()

    # anomalous imgs
    anomalous_indices = []

    model.eval()

    #print(f"length of dataset = {len(test_dataset)}")
    current_video = ""
    frames_per_video = 0
    k = 0

    chips_per_frame = test_dataset.chips_per_frame()
    chips_by_frames = (chips_per_frame, len(test_dataset)//chips_per_frame)
    print(f'chips, frames = {chips_by_frames}')
    psnr_list = np.zeros(chips_by_frames)
    feature_distance_list = np.zeros(chips_by_frames)

    for imgs in tqdm(test_batch):
        #print(f'input = {imgs.shape}')
        imgs = Variable(imgs).cuda()

        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:indx_of_inflection], m_items_test, False)

        #print(f'outputs = {outputs.shape}')
        #print(f'compactness_loss = {compactness_loss.item()}')
        
        # loop over batch dimension
        for i in range(len(imgs)):
            mse_imgs = torch.mean(loss_func_mse((outputs[i]+1)/2, (imgs[i,indx_of_inflection:]+1)/2)).item()
            mse_feas = compactness_loss.item()
            #print(f'output = {len(mse_imgs)}')

            # Save normal data for next loop.
            #imgs_prev = imgs
            #updated_feas_prev = updated_feas
            #outputs_prev = outputs

            # Calculating the threshold for updating at the test time
            point_sc = point_score(outputs, imgs[:,indx_of_inflection:])

            # Decide if we need to update the memory.
            if point_sc < th:
                query = F.normalize(feas, dim=1)
                query = query.permute(0,2,3,1) # b X h X w X d
                m_items_test = model.memory.update(query, m_items_test, False)
            else:
                a = 0
            
            frame = k//chips_per_frame
            chip = k%chips_per_frame
            #print(f'chip, frame = {chip}, {frame}\n')

            psnr_list[chip, frame] = psnr(mse_imgs)
            feature_distance_list[chip, frame] = mse_feas
            k += 1

    if anomaly_context=="chips_video":
        normality_score_total_list = [np.empty((1,0)) for i in range(0, chips_per_frame)]
        for c in range(0, chips_per_frame):
            start = 0
            stop = 0
            for v in videos_list:
                v_name = v.split('/')[-1]
                stop += test_dataset.videos[v_name]['length']-(config['t_length']-1)
                #print(f"{c},{v_name} : {start}-{stop}")
                normality_score_total_list[c] = np.append(normality_score_total_list[c], score_sum(anomaly_score_array(psnr_list[c,start:stop]), anomaly_score_array_inv(feature_distance_list[c,start:stop]), config['alpha']))
                start=stop

    elif anomaly_context=="all":
        psnr_array = np.asarray(psnr_list).flatten()
        feature_distance_array = np.asarray(feature_distance_list).flatten()
        temp = score_sum(anomaly_score_array(psnr_array), anomaly_score_array_inv(feature_distance_array), config['alpha'])
        normality_score_total_list = temp.reshape((psnr_list.shape[0], psnr_list.shape[1]))
    elif anomaly_context=="chips":
        normality_score_total_list = [score_sum(anomaly_score_array(psnr_list[c,:]), anomaly_score_array_inv(feature_distance_list[c,:]), config['alpha']) for c in range(chips_per_frame)]
    else:
        print(f"Unrecognized anomaly_context method: {anomaly_context}")

    # turn into numpy array and return everything
    return np.asarray(normality_score_total_list), videos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNAD")
    parser.add_argument('--config', type=str, default='./eval_config.json', help='directory of log')
    parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
    parser.add_argument('--truthfile', default=None, type=str, help='directory of truth')
    parser.add_argument('--outfile', default="", type=str, help='directory of results pickle file')
    parser.add_argument('--anomaly_context', default="all", type=str, help='options: \n1). all (default) \n2). chips \n3). chips_video')
    args = parser.parse_args()

    with open(args.config) as config_file:
        print(f"loading {args.config}")
        config = json.load(config_file)
    
    # pass test data through the model
    anomaly_scores, videos = evaluate_model(config, args.anomaly_context, args.th)

    
    if args.truthfile is not None and os.path.exists(args.truthfile): 
        labels = np.load(args.truthfile)
        if config['dataset_type'] == 'shanghai':
            labels = np.expand_dims(labels, 0)

        # Setting for video anomaly detection
        labels_list=[]
        video_length = 0
        videos_list = sorted(glob.glob(os.path.join(config['dataset_path'], '*')))
        for video in videos_list:
            video_name = video.split('/')[-1]
            labels_list = np.append(labels_list, labels[0][config['t_length']-1+video_length:videos[video_name]['length']+video_length])
            video_length += videos[video_name]['length']

    else:
        labels = None

    # name the output file with the number of chips in each dimension if 
    # no name is given

    outfile = args.outfile
    if (outfile == ""):
        w = config['image']['size_x'] // config['window']['size_x']
        h = config['image']['size_y'] // config['window']['size_y']
        outfile = f"{w}x{h}_{args.anomaly_context}.pickle"

    # add the anomaly context parameter to the dictionary because it will be 
    # saved in the evaluation results pickle file. Thus, when plotting the results,
    # we will know what anomaly_context method was used to generate the results.
    config['anomaly_context'] = args.anomaly_context

    with open(outfile, "wb") as fh:
        #pickle.dump((psnr, fs, labels, anomalies, config), fh)
        pickle.dump((anomaly_scores, labels_list, config), fh)
    

