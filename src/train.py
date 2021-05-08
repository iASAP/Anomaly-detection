import numpy as np
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
import cv2
import math
from collections import OrderedDict
import copy
import time
from data import ChipDataLoader
from model import *
#from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
from tqdm import tqdm

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNAD")
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
    parser.add_argument('--config', type=str, default='./train_config.json', help='directory of log')
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    torch.manual_seed(2020)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus']
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

    print('Creating dataloaders...')

    train_dir = os.path.join(config['dataset_path'], config['dataset_type'], "training", "frames/")

    # Loading dataset
    # train_dataset = DataLoader(train_dir, transforms.Compose([
    #              transforms.ToTensor(),          
    #              ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1, color=args.color)
    img_size = (config["image"]["size_x"], config["image"]["size_y"]) 
    win_size = (config["window"]["size_x"], config["window"]["size_y"])
    win_step = (config["window"]["step_x"], config["window"]["step_y"])
    train_dataset = ChipDataLoader(train_dir, transforms.Compose([transforms.ToTensor(),]), img_size, win_size, win_step, time_step=config['t_length']-1, color=config['image']['color'], ext=config['extension'])

    train_size = len(train_dataset)

    train_batch = data.DataLoader(train_dataset, batch_size = config['batch_size'], 
                                  shuffle=True, num_workers=config['num_workers'], drop_last=True)

    n_channel = 3 if config['image']['color'] else 1
    indx_of_inflection = (config['t_length']-1) * n_channel

    # Model setting
    print('Initializing model...')
    model = convAE(n_channel, config['t_length'], config['msize'], config['fdim'], config['mdim'])
    params_encoder = list(model.encoder.parameters()) 
    params_decoder = list(model.decoder.parameters())
    params = params_encoder + params_decoder
    optimizer = torch.optim.Adam(params, lr = config['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = args.epochs)
    model.cuda()


    # Report the training process
    loss_func_mse = nn.MSELoss(reduction='none')

    # Training

    m_items = F.normalize(torch.rand((config['msize'], config['mdim']), dtype=torch.float), dim=1).cuda() # Initialize the memory items

    for epoch in tqdm(range(args.epochs)):
        labels_list = []
        model.train()
        
        start = time.time()
        for j,imgs in enumerate(train_batch):
            imgs = Variable(imgs).cuda()
#            if j==0: print(f"{np.shape(imgs)}")
            
            outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(imgs[:,0:indx_of_inflection], m_items, True)
            
            optimizer.zero_grad()
            loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:,indx_of_inflection:]))
            loss = loss_pixel + config['loss_compact'] * compactness_loss + config['loss_separate'] * separateness_loss
            loss.backward(retain_graph=True)
            optimizer.step()
            
        scheduler.step()
        
        # print('----------------------------------------')
        # print('Epoch:', epoch+1)
        # print('Loss: Reconstruction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}'.format(loss_pixel.item(), compactness_loss.item(), separateness_loss.item()))
        # print('Memory_items:')
        # print(m_items)
        # print('----------------------------------------')
        
    print('Training is finished')
    # Save the model and the memory items
    os.makedirs(config['model_dir'], exist_ok=True)
    torch.save(model, os.path.join(config['model_dir'], 'model.pth'))
    torch.save(m_items, os.path.join(config['model_dir'], 'keys.pt'))
        
    # sys.stdout = orig_stdout
    # f.close()
