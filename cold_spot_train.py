#!/usr/bin/env python

import os, json, pdb
import torch
import numpy as np
from datetime import datetime

from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN, RPN
from faster_rcnn.utils.timer import Timer
from augmentations import Augmentation
import faster_rcnn.roi_data_layer.roidb as rdl_roidb
from faster_rcnn.roi_data_layer.layer import RoIDataLayer
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file
from Dataset import Dataset
from torch.utils.data import DataLoader

try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)



# hyper-parameters
# ------------
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
pretrained_model = 'data/pretrained_model/VGG_imagenet.npy'
output_dir = 'models/saved_model3'

start_step = 0
end_step = 100
lr_decay_steps = {60000, 80000}
lr_decay = 1./10

rand_seed = 1024
_DEBUG = True
use_tensorboard = True
remove_all_log = False   # remove all historical experiments in TensorBoard
exp_name = None # the previous experiment name in TensorBoard

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

# load net
net = FasterRCNN(classes=['__background__', 'building'], debug=_DEBUG)
network.weights_normal_init(net, dev=0.01)
network.load_pretrained_npy(net, pretrained_model)

net.cuda()
net.train()

params = list(net.parameters())
optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()

train_data = json.load(open('../data/train_data.json'))
dataset = Dataset('../data', train_data, transform=Augmentation()).even()

for step in range(start_step, end_step+1):
    iter = 0
    for i in np.random.permutation(len(dataset)):
        im_data, gt_boxes, im_info = dataset[i]
        im_info = np.expand_dims(im_info, 0)
        im_data = np.expand_dims(im_data, 0)
        gt_ishard = np.zeros(len(gt_boxes)).astype('int32')
        dontcare_areas = np.zeros(0, dtype='float64')
        net(im_data, im_info, gt_boxes, gt_ishard, dontcare_areas)
        loss = net.loss + net.rpn.loss
        if _DEBUG:
            tp += float(net.tp)
            tf += float(net.tf)
            fg += net.fg_cnt
            bg += net.bg_cnt

        train_loss += loss.data[0]

        # backward
        optimizer.zero_grad()
        loss.backward()
        network.clip_gradient(net, 10.)
        optimizer.step()

        print('[%d/%d] net loss: %f, rpn loss: %f' % (iter, len(dataset), net.loss.data[0], net.rpn.loss.data[0]))

        iter += 1


    save_name = os.path.join(output_dir, 'faster_rcnn_{}.h5'.format(step))
    network.save_net(save_name, net)
    print('save model: {}'.format(save_name))
    if False and step in lr_decay_steps:
        lr *= lr_decay
        optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)



