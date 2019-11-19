#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import pylab
from pylab import figure, show, legend
from mpl_toolkits.axes_grid1 import host_subplot

# read the log file
# fp = open('/home/jky/model/Detectron/output/aiia/train/aiia_e2e_faster_rcnn_R-101-FPN.log', 'r')
ROOT_PATH = "/home/lqp2018/mnt/lqp2018/jky/log/"

areas = ['rural', 'suburban', 'urban']
font_size = 13

fp_rual_x = open(ROOT_PATH + 'rural_e2e_bquad_rcnn_X_101_32x8d_FPN_1x_gpu3_trec.txt', 'r')
fp_suburban_x = open(ROOT_PATH + 'suburban_e2e_bquad_rcnn_X_101_32x8d_FPN_1x_gpu3_trec_loss_10.txt', 'r')
fp_urban_x = open(ROOT_PATH + 'urban_e2e_bquad_rcnn_X_101_32x8d_FPN_1x_gpu3_trec.txt', 'r')
# fp = open('/home/jky-cuda8/model/maskrcnn-benchmark/output/qinghai/e2e_faster_rcnn_R_101_FPN_1x_gpu1_bquad_loss_20_1.5_only_4points_minarea5/log.txt', 'r')

fig = plt.figure(figsize=(6.4, 3.9552))

host = host_subplot(111)
# host = plt.Axes(fig, [0, 0, 0, 0])
plt.subplots_adjust(left=0.088, bottom=0.12, right=0.995, top=0.98, wspace=0, hspace=0)
# plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
# par1 = host.twinx()
# set labels
host.set_xlabel("iterations", fontsize = font_size)
host.set_ylabel("joint loss", fontsize = font_size)
# host.axis('equal')

font = {'size' : 30}

train_iterations = []
train_loss = []
test_iterations = []
train_loss_bbox = []
train_loss_mask = []
train_loss_bquad = []
# test_accuracy = [


baseline_iterations = []
baseline_train_loss = []
baseline_train_loss_bbox = []
baseline_train_loss_mask = []

for ln in fp_rual_x:
    # get train_iterations and train_loss
    if 'iter:' in ln and 'loss:' in ln:
        arr = re.findall(r'iter: \b\d+\b  ', ln)
        print(arr)
        baseline_iterations.append(int(arr[0].strip('iter:')))
        arr = re.findall(r'loss: \b\d+.\d+ ', ln)
        print(arr)
        baseline_train_loss.append(float(arr[0].strip('loss:')))
        arr = re.findall(r'loss_box_reg: \b\d+.\d+ ', ln)
        print(arr)
        baseline_train_loss_bbox.append(float(arr[0].strip('loss_box_reg:')))
        # arr = re.findall(r'loss_mask: \b\d+.\d+ ', ln)
        # print(arr)
        # baseline_train_loss_mask.append(float(arr[0].strip('loss_mask:')))

fp_rual_x.close()

p1, = host.plot(baseline_iterations, baseline_train_loss, label="Rural")#, color = (9/255, 132/255, 227/255))#color = (9/255, 114/255, 185/255))

baseline_iterations = []
baseline_train_loss = []
baseline_train_loss_bbox = []
baseline_train_loss_mask = []

for ln in fp_suburban_x:
    # get train_iterations and train_loss
    if 'iter:' in ln and 'loss:' in ln:
        arr = re.findall(r'iter: \b\d+\b  ', ln)
        print(arr)
        baseline_iterations.append(int(arr[0].strip('iter:')))
        arr = re.findall(r'loss: \b\d+.\d+ ', ln)
        print(arr)
        baseline_train_loss.append(float(arr[0].strip('loss:')))
        arr = re.findall(r'loss_box_reg: \b\d+.\d+ ', ln)
        print(arr)
        baseline_train_loss_bbox.append(float(arr[0].strip('loss_box_reg:')))
        # arr = re.findall(r'loss_mask: \b\d+.\d+ ', ln)
        # print(arr)
        # baseline_train_loss_mask.append(float(arr[0].strip('loss_mask:')))

fp_suburban_x.close()

p2, = host.plot(baseline_iterations, baseline_train_loss, label="Suburban")#, color = (238/255, 90/255, 36/255))#, color = (204/255, 32/255, 42/255))

baseline_iterations = []
baseline_train_loss = []
baseline_train_loss_bbox = []
baseline_train_loss_mask = []

for ln in fp_urban_x:
    # get train_iterations and train_loss
    if 'iter:' in ln and 'loss:' in ln:
        arr = re.findall(r'iter: \b\d+\b  ', ln)
        print(arr)
        baseline_iterations.append(int(arr[0].strip('iter:')))
        arr = re.findall(r'loss: \b\d+.\d+ ', ln)
        print(arr)
        baseline_train_loss.append(float(arr[0].strip('loss:')))
        arr = re.findall(r'loss_box_reg: \b\d+.\d+ ', ln)
        print(arr)
        baseline_train_loss_bbox.append(float(arr[0].strip('loss_box_reg:')))
        # arr = re.findall(r'loss_mask: \b\d+.\d+ ', ln)
        # print(arr)
        # baseline_train_loss_mask.append(float(arr[0].strip('loss_mask:')))

fp_urban_x.close()

p3, = host.plot(baseline_iterations, baseline_train_loss, label="Urban")#, color = (39/255, 174/255, 96/255))#, color = (24/255, 184/255, 153/255))

# par1.set_ylabel("validation accuracy")

# plot curves
# p1, = host.plot(baseline_iterations, baseline_train_loss_bbox, label="baseline_bbox_loss")
# p2, = host.plot(baseline_iterations, baseline_train_loss_mask, label="baseline_mask_loss")
#
# p3, = host.plot(train_iterations, train_loss_bbox, label="bbox_loss")
# if len(train_loss_mask) > 0:
#     p4, = host.plot(train_iterations, train_loss_mask, label="mask_loss")
# if len(train_loss_bquad) > 0:
#     p5, = host.plot(train_iterations, train_loss_bquad, label="mask_bquad")
# p2, = par1.plot(test_iterations, test_accuracy, label="validation accuracy")

# set location of the legend,
# 1->rightup corner, 2->leftup corner, 3->leftdown corner
# 4->rightdown corner, 5->rightmid ...
host.legend(loc=1, fontsize = font_size)

# set label color
# host.axis["left"].label.set_color(p1.get_color())
# par1.axis["right"].label.set_color(p2.get_color())
# set the range of x axis of host and y axis of par1
#host.set_xlim([0, train_iterations[-1]])
host.set_ylim([0., 10.])
plt.xticks(fontsize = font_size)
plt.yticks(fontsize = font_size)


plt.draw()
plt.savefig("Fig04.pdf")
# plt.show()