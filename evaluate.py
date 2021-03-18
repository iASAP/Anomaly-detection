import numpy as np
import os
import sys
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
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
import datetime 
from data import DataLoader
#from other.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
import sklearn.metrics as metrics
from utils import *
from model import *
import random
import glob
import shutil
from distutils.dir_util import copy_tree

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MNAD")
    parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
    parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
    parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
    parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
    parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
    parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
    parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
    parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
    parser.add_argument('--dataset_path', type=str, default='./dataset/', help='directory of data')
    parser.add_argument('--model_dir', default="./exp", type=str, help='directory of model')
    parser.add_argument('--color', type=str, default=True, help='directory of log')
    #parser.add_argument('--m_items_dir', default="./exp", type=str, help='directory of model')

    args = parser.parse_args()

    # ceate the output directory
    date_time_str = str(datetime.datetime.today())
    output_dir = os.path.join("figures", date_time_str)
    os.makedirs(output_dir, exist_ok=True)

    # calculate channel numbers and related
    n_channel = 3 if args.color else 1
    indx_of_inflection = (args.t_length-1) * n_channel

    torch.manual_seed(2020)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    if args.gpus is None:
        gpus = "0"
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus
    else:
        gpus = ""
        for i in range(len(args.gpus)):
            gpus = gpus + args.gpus[i] + ","
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

    test_dir = os.path.join(args.dataset_path, args.dataset_type, "testing", "frames")
    model_dir = os.path.join(args.model_dir, args.dataset_type, "log")

    # Loading dataset
    test_dataset = DataLoader(test_dir, transforms.Compose([
                 transforms.ToTensor(),            
                 ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1, color=args.color)

    test_size = len(test_dataset)

    test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                                 shuffle=False, num_workers=args.num_workers_test, drop_last=False)

    loss_func_mse = nn.MSELoss(reduction='none')

    # Loading the trained model
    model = torch.load(os.path.join(model_dir, "model.pth"))
    model.cuda()
    m_items = torch.load(os.path.join(model_dir, "keys.pt"))


    labels = np.load('./data/frame_labels_'+args.dataset_type+'.npy')
    if args.dataset_type == 'shanghai':
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
    label_length = 0
    psnr_list = {}
    feature_distance_list = {}

    print('Evaluation of', args.dataset_type)

    # Setting for video anomaly detection
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        labels_list = np.append(labels_list, labels[0][4+label_length:videos[video_name]['length']+label_length])
        label_length += videos[video_name]['length']
        psnr_list[video_name] = []
        feature_distance_list[video_name] = []

    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('/')[-1]]['length']
    m_items_test = m_items.clone()

    model.eval()

    for k,(imgs) in enumerate(test_batch):

        if k == label_length-4*(video_num+1):
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']

        imgs = Variable(imgs).cuda()

        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:indx_of_inflection], m_items_test, False)
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,indx_of_inflection:]+1)/2)).item()
        mse_feas = compactness_loss.item()

        # 
        if k > 0:
        
            # imgs_prev = torch.cat(buffer, dim=0).to('cuda:0')
            # imgs_prev = torch.mean(imgs_prev, dim=0, keepdim=True).to('cuda:0')

            # Compute diff outputs.
            outputs_diff = model.forward(imgs_prev[:,0:indx_of_inflection], m_items_test, False, x2=imgs[:,0:indx_of_inflection])
            outputs_diff = outputs_diff[0,:,:,:].detach().to('cpu').numpy()
            outputs_diff = ((outputs_diff + 1)*127.5).astype(np.uint8)
            # outputs_diff[outputs_diff >= 160] = 255
            # outputs_diff[outputs_diff < 160] = 0

            # Plot output reconstructions.
            if labels_list[k] == 0:  # anomaly

                # Plot diff outputs.
                plt.cla()
                plt.title('Error Image ' + str(k), fontsize=18)
                plt.imshow(np.moveaxis(outputs_diff, 0, 2), vmin=0, vmax=255)
                plt.savefig(os.path.join(output_dir, str(k) + '_normal.png'))

            elif labels_list[k] == 1:  # normal

                # Plot diff outputs.
                plt.cla()
                plt.title('Error Image ' + str(k), fontsize=18)
                plt.imshow(np.moveaxis(outputs_diff, 0, 2), vmin=0, vmax=255)
                plt.savefig(os.path.join(output_dir, str(k) + '_anomaly.png'))

            print('Frame=' + str(k) + ', Anomaly=' + str(labels_list[k]))    

        # Save normal data for next loop.
        imgs_prev = imgs
        updated_feas_prev = updated_feas
        outputs_prev = outputs

        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs[:,indx_of_inflection:])

        # Decide if we need to update the memory.
        if  point_sc < args.th:
            query = F.normalize(feas, dim=1)
            query = query.permute(0,2,3,1) # b X h X w X d
            m_items_test = model.memory.update(query, m_items_test, False)
        else:
            a = 0

        # Update PSNR and distance lists.
        psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))
        feature_distance_list[videos_list[video_num].split('/')[-1]].append(mse_feas)


    # Measuring the abnormality score and the AUC
    anomaly_score_total_list = []
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]), 
                                         anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha)

    anomaly_score_total_list = np.asarray(anomaly_score_total_list)

    accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))

    # Generate PRC.
    p, r, thresholds = metrics.precision_recall_curve(1 - labels_list, anomaly_score_total_list)
    f1_scores = 2*p*r/(p + r)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    # Generate ROC curve.
    fpr, tpr, thresholds = metrics.roc_curve(1 - labels_list, anomaly_score_total_list)

    # Plot ROC curve.
    plt.cla()
    plt.title('ROC Curve', fontsize=18)
    plt.xlabel('False Postive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.plot(fpr, tpr)
    #plt.scatter(fpr[best_idx], tpr[best_idx])
    plt.pause(0.01)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))

    # Plot PRC.
    plt.cla()
    plt.title('Precision-Recall Curve', fontsize=18)
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.plot(r, p, label='PRC')
    plt.scatter(r[best_idx], p[best_idx], color='r', label='Best F1 Score')
    plt.pause(0.01)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'prc.png'))

    # Plot anomaly curve.
    change_points = np.where(np.diff(1 - labels_list) != 0)[0]
    b1 = change_points[0]
    b2 = change_points[1]
    b3 = change_points[2]
    b4 = change_points[3]
    plt.cla()
    plt.title('Normality Score', fontsize=18)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Normality Score', fontsize=14)
    plt.xlim(xmin=b1, xmax=b4)
    plt.plot(np.arange(b1, b4), anomaly_score_total_list[b1:b4], label='Normality Score')
    plt.axhline(best_threshold, color='red', linestyle='--', label='Best Threshold')
    plt.axvspan(xmin=b1, xmax=b2, color='g', alpha=0.1)
    plt.axvspan(xmin=b2, xmax=b3, color='r', alpha=0.1)
    plt.axvspan(xmin=b3, xmax=b4, color='g', alpha=0.1)
    #plt.axvspan(xmin=b4, xmax=b5, color='r', alpha=0.1)
    plt.pause(0.01)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'anomaly_curve_all.png'), bbox_inches='tight')

    # Plot anomaly curve.
    plt.figure(figsize=(15,5)) 
    change_points = np.where(np.diff(1 - labels_list) != 0)[0]
    plt.cla()
    plt.title('Normality Score', fontsize=18)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Normality Score', fontsize=14)
    plt.xlim(xmin=0, xmax=change_points[-1])
    plt.plot(np.arange(0, change_points[-1]), anomaly_score_total_list[0:change_points[-1]], label='Normality Score')
    plt.axhline(best_threshold, color='red', linestyle='--', label='Best Threshold')
    plt.axvspan(xmin=0, xmax=change_points[0], color='g', alpha=0.1)
    plt.axvspan(xmin=change_points[0]+1, xmax=change_points[1], color='r', alpha=0.1)
    plt.axvspan(xmin=change_points[1]+1, xmax=change_points[2], color='g', alpha=0.1)
    plt.axvspan(xmin=change_points[2]+1, xmax=change_points[3], color='g', alpha=0.1)
    plt.axvspan(xmin=change_points[3]+1, xmax=change_points[4], color='r', alpha=0.1)
    plt.axvspan(xmin=change_points[4]+1, xmax=change_points[5], color='g', alpha=0.1)
    plt.axvspan(xmin=change_points[5]+1, xmax=change_points[6], color='g', alpha=0.1)
    plt.axvspan(xmin=change_points[6]+1, xmax=change_points[7], color='r', alpha=0.1)
    plt.axvspan(xmin=change_points[7]+1, xmax=change_points[8], color='g', alpha=0.1)
    plt.axvspan(xmin=change_points[8]+1, xmax=change_points[9], color='r', alpha=0.1)
    plt.axvspan(xmin=change_points[9]+1, xmax=change_points[10], color='g', alpha=0.1)
    plt.pause(0.01)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'anomaly_curve_all.png'), bbox_inches='tight')

    # Plot truth curve.
    change_points = np.where(np.diff(1 - labels_list) != 0)[0]
    b1 = 0
    b2 = change_points[2]
    plt.cla()
    plt.title('True Anomalies', fontsize=18)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Binary label (0 = anomaly, 1 = normal)', fontsize=14)
    plt.plot(1 - labels_list[b1:b2], label='Truth')
    #plt.axhline(best_threshold, color='red', linestyle='--', label='Best Threshold')
    plt.pause(0.01)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'truth_curve.png'))

    # copy all contents to the 'latest' folder
    shutil.rmtree("figures/latest")
    copy_tree(output_dir, "figures/latest")

    # Plot quantitative results.
    print('The result of ', args.dataset_type)
    print('AUC: ', accuracy*100, '%')

