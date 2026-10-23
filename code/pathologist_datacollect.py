"""
Copyright (C) 2026 Rajaram Lab, UT Southwestern Medical Center.

This file is part of the Delta-Marches project.

Delta-Marches is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Delta-Marches is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Delta-Marches. If not, see <https://www.gnu.org/licenses/>.

Author: Thuong Nguyen
Year: 2026
"""
# %%
# This code is to analyze the pathologist evaluation of the synthetic images

# %%
#%%
import torch
import pandas as pd
import os,sys
from PIL import Image
from configparser import ConfigParser
device = torch.device('cuda:0')
import time
from torch import nn, cuda, device
from torch.autograd import Variable
import torchvision
from torch.utils.data import Dataset# data management
import numpy as np
import os,sys
from configparser import ConfigParser
import torch.optim as optim # Optimization algorithm
import matplotlib.pyplot as plt
import torchvision as tv
import itertools
import pickle
from tqdm import tqdm
import torchvision.transforms as transforms
import random
from scipy.spatial.distance import cdist
from torchvision.utils import save_image
print(cuda.is_available())
device = device('cuda:0')
print(device)
import os as os
import sys
import scipy.stats as stats
from configparser import ConfigParser
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as mplCB
import cv2 as cv2
import pandas as pd
import matplotlib as mpl
import math
from numpy.linalg import norm 
import skimage


# %% import data

import matplotlib.pyplot as plt
import numpy as np
import yaml
yaml_paths = './data/data_dir.yaml'
with open(yaml_paths, 'r') as yaml_file:
    file_paths = yaml.safe_load(yaml_file)
    

save_dir = file_paths['pathologist_result']['ground_truth']
gt = pd.read_excel(save_dir)

# %% 
    
#Load the ground truth
gt_actualSynLow = len(np.where((gt['Tumor grade'] == "Low grade") &  (gt['Data'] == 'Synthetic'))[0])
gt_actualSynHigh = len(np.where((gt['Tumor grade'] == "High grade") &  (gt['Data'] == 'Synthetic'))[0])
gt_actualRealLow = len(np.where((gt['Tumor grade'] == "Low grade") &  (gt['Data'] == 'Real'))[0])
gt_actualRealHigh = len(np.where((gt['Tumor grade'] == "High grade") &  (gt['Data'] == 'Real'))[0])


# %%
#Load the answer from pathologists
ans = []
for i in range(1,5):
    save_dir = file_paths['pathologist_result']['pathologist_ans'][f'pathologist{i}_ans']
    ans.append(pd.read_excel(save_dir))


# %% confusion matrix for real vs synthetic prediction
allgt = gt['Data'].tolist()*4
allans = ans[0]['Data'].tolist()+ans[1]['Data'].tolist()+ans[2]['Data'].tolist()+ans[3]['Data'].tolist()
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(allgt, allans)
#take percentage of each row
cm = cm/cm.sum(axis = 1)[:,np.newaxis]
classes = np.unique(allgt)

import seaborn as sns
plt.figure(figsize=(4, 4))
ax = sns.heatmap(cm, annot=True, cmap='coolwarm', annot_kws={"fontsize": 20}, vmin=0, vmax=1)

plt.xticks(np.arange(len(classes)) + 0.5, classes, fontsize=20)
plt.yticks(np.arange(len(classes)) + 0.5, classes, fontsize=20)
plt.xlabel('Predicted', fontsize=20)
plt.ylabel('Actual', fontsize=20)

# Increase font size of colormap ticks
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)
plt.title('Real and Synthetic Prediction', fontsize = 20)

plt.savefig('./image/confusion_matrix_real_vs_synthetic_all4pathologistv2.png',dpi = 300, bbox_inches = 'tight')


# %% confusion matrix for Real and synthetic prediction\n on low grade (1, 2) data
real_syn_allgt = gt['Data'].tolist()*4
real_syn_allans = ans[0]['Data'].tolist()+ans[1]['Data'].tolist()+ans[2]['Data'].tolist()+ans[3]['Data'].tolist()

tumor_grade_allgt = gt['Tumor grade'].tolist()*4

low_idx = np.where(np.array(tumor_grade_allgt) == 'Low grade')[0]
allgt = np.array(real_syn_allgt)[low_idx]
allans = np.array(real_syn_allans)[low_idx]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(allgt, allans)
#take percentage of each row
cm = cm/cm.sum(axis = 1)[:,np.newaxis]
classes = np.unique(allgt)

import seaborn as sns
plt.figure(figsize=(4, 4))
ax = sns.heatmap(cm, annot=True, cmap='coolwarm', annot_kws={"fontsize": 20}, vmin=0, vmax=1)

plt.xticks(np.arange(len(classes)) + 0.5, classes, fontsize=20)
plt.yticks(np.arange(len(classes)) + 0.5, classes, fontsize=20)
plt.xlabel('Predicted', fontsize=20)
plt.ylabel('Actual', fontsize=20)

# Increase font size of colormap ticks
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)
plt.title('Real and synthetic prediction\n on low grade (1, 2) data', fontsize = 20)

plt.savefig('./image/confusion_matrix_real_vs_synthetic_all4pathologistv2_lowgrade.png',dpi = 300, bbox_inches = 'tight')


# %% confusion matrix for Real and synthetic prediction\n on high grade (3, 4) data
real_syn_allgt = gt['Data'].tolist()*4
real_syn_allans = ans[0]['Data'].tolist()+ans[1]['Data'].tolist()+ans[2]['Data'].tolist()+ans[3]['Data'].tolist()

tumor_grade_allgt = gt['Tumor grade'].tolist()*4

low_idx = np.where(np.array(tumor_grade_allgt) == 'High grade')[0]
allgt = np.array(real_syn_allgt)[low_idx]
allans = np.array(real_syn_allans)[low_idx]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(allgt, allans)
#take percentage of each row
cm = cm/cm.sum(axis = 1)[:,np.newaxis]
classes = np.unique(allgt)

import seaborn as sns
plt.figure(figsize=(4, 4))
ax = sns.heatmap(cm, annot=True, cmap='coolwarm', annot_kws={"fontsize": 20}, vmin=0, vmax=1)

plt.xticks(np.arange(len(classes)) + 0.5, classes, fontsize=20)
plt.yticks(np.arange(len(classes)) + 0.5, classes, fontsize=20)
plt.xlabel('Predicted', fontsize=20)
plt.ylabel('Actual', fontsize=20)

# Increase font size of colormap ticks
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)
plt.title('Real and synthetic prediction\n on high grade (3, 4) data', fontsize = 20)

plt.savefig('./image/confusion_matrix_real_vs_synthetic_all4pathologistv2_highgrade.png',dpi = 300, bbox_inches = 'tight')


# %% Tumor grade prediction\n on real data
tumor_grade_allgt = gt['Tumor grade'].tolist()*4
tumor_grade_allans = ans[0]['Tumor grade'].tolist()+ans[1]['Tumor grade'].tolist()+ans[2]['Tumor grade'].tolist()+ans[3]['Tumor grade'].tolist()
real_syn_allgt = gt['Data'].tolist()*4

real_idx = np.where(np.array(real_syn_allgt) == 'Real')[0]
allgt = np.array(tumor_grade_allgt)[real_idx]
allans = np.array(tumor_grade_allans)[real_idx]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(allgt, allans)
#take percentage of each row
cm = cm/cm.sum(axis = 1)[:,np.newaxis]
classes = np.unique(allgt)

import seaborn as sns
plt.figure(figsize=(4, 4))
ax = sns.heatmap(cm, annot=True, cmap='coolwarm', annot_kws={"fontsize": 20}, vmin=0, vmax=1)

plt.xticks(np.arange(len(classes)) + 0.5, ['High', 'Low'], fontsize=20)
plt.yticks(np.arange(len(classes)) + 0.5, ['High', 'Low'] , fontsize=20)
plt.xlabel('Predicted', fontsize=20)
plt.ylabel('Actual', fontsize=20)

# Increase font size of colormap ticks
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.title('Tumor grade prediction\n on real data', fontsize = 20)
plt.savefig('./image/confusion_matrix_tumorgrade_all4pathologist_realdata.png',dpi = 300, bbox_inches = 'tight')


# %% Tumor grade prediction\n on synthetic data
tumor_grade_allgt = gt['Tumor grade'].tolist()*4
tumor_grade_allans = ans[0]['Tumor grade'].tolist()+ans[1]['Tumor grade'].tolist()+ans[2]['Tumor grade'].tolist()+ans[3]['Tumor grade'].tolist()
real_syn_allgt = gt['Data'].tolist()*4

syn_idx = np.where(np.array(real_syn_allgt) == 'Synthetic')[0]
allgt = np.array(tumor_grade_allgt)[syn_idx]
allans = np.array(tumor_grade_allans)[syn_idx]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(allgt, allans)
#take percentage of each row
cm = cm/cm.sum(axis = 1)[:,np.newaxis]
classes = np.unique(allgt)

import seaborn as sns
plt.figure(figsize=(4, 4))
ax = sns.heatmap(cm, annot=True, cmap='coolwarm', annot_kws={"fontsize": 20}, vmin=0, vmax=1)

plt.xticks(np.arange(len(classes)) + 0.5, ['High', 'Low'], fontsize=20)
plt.yticks(np.arange(len(classes)) + 0.5, ['High', 'Low'] , fontsize=20)
plt.xlabel('Predicted', fontsize=20)
plt.ylabel('Actual', fontsize=20)

# Increase font size of colormap ticks
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.title('Tumor grade prediction\n on synthetic data', fontsize = 20)
plt.savefig('./image/confusion_matrix_tumorgrade_all4pathologist_synthetic.png',dpi = 300, bbox_inches = 'tight')


# %%  Tumor Grade Prediction
allgt = gt['Tumor grade'].tolist()*4
allans = ans[0]['Tumor grade'].tolist()+ans[1]['Tumor grade'].tolist()+ans[2]['Tumor grade'].tolist()+ans[3]['Tumor grade'].tolist()
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(allgt, allans)
#take percentage of each row
cm = cm/cm.sum(axis = 1)[:,np.newaxis]
classes = np.unique(allgt)

import seaborn as sns
plt.figure(figsize=(4, 4))
ax = sns.heatmap(cm, annot=True, cmap='coolwarm', annot_kws={"fontsize": 20}, vmin=0, vmax=1)

plt.xticks(np.arange(len(classes)) + 0.5, ['High', 'Low'], fontsize=20)
plt.yticks(np.arange(len(classes)) + 0.5, ['High', 'Low'] , fontsize=20)
plt.xlabel('Predicted', fontsize=20)
plt.ylabel('Actual', fontsize=20)

# Increase font size of colormap ticks
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.title('Tumor Grade Prediction', fontsize = 20)
plt.savefig('./image/confusion_matrix_tumorgrade_all4pathologist.png',dpi = 300, bbox_inches = 'tight')


# %%

#Collect the average real vs synthetic accuracy for each group
ans_predSyn_actualSynLow = []
ans_predSyn_actualSynHigh = []
ans_predReal_actualRealLow = []
ans_predReal_actualRealHigh = []
for i in range(4):
    ans_predSyn_actualSynLow.append(len(np.where((gt['Tumor grade'] == "Low grade") &  (gt['Data'] == 'Synthetic') &  (ans[i]['Data'] == 'Synthetic'))[0])/gt_actualSynLow)
    ans_predSyn_actualSynHigh.append(len(np.where((gt['Tumor grade'] == "High grade") &  (gt['Data'] == 'Synthetic') &  (ans[i]['Data'] == 'Synthetic'))[0])/gt_actualSynHigh)
    ans_predReal_actualRealLow.append(len(np.where((gt['Tumor grade'] == "Low grade") &  (gt['Data'] == 'Real') &  (ans[i]['Data'] == 'Real'))[0])/gt_actualRealLow)
    ans_predReal_actualRealHigh.append(len(np.where((gt['Tumor grade'] == "High grade") &  (gt['Data'] == 'Real') &  (ans[i]['Data'] == 'Real'))[0])/gt_actualRealHigh)


# %%
#Collect the average tumor grading accuracy for each group
ans_predLow_actualRealLow = []
ans_predLow_actualSynLow = []
ans_predHigh_actualRealHigh = []
ans_predHigh_actualSynHigh = []
for i in range(4):
    ans_predLow_actualRealLow.append(len(np.where((gt['Tumor grade'] == "Low grade") &  (gt['Data'] == 'Real') &  (ans[i]['Tumor grade'] == 'Low grade'))[0])/gt_actualRealLow)
    ans_predLow_actualSynLow.append(len(np.where((gt['Tumor grade'] == "Low grade") &  (gt['Data'] == 'Synthetic') &  (ans[i]['Tumor grade'] == 'Low grade'))[0])/gt_actualSynLow)
    ans_predHigh_actualRealHigh.append(len(np.where((gt['Tumor grade'] == "High grade") &  (gt['Data'] == 'Real') &  (ans[i]['Tumor grade'] == 'High grade'))[0])/gt_actualRealHigh)
    ans_predHigh_actualSynHigh.append(len(np.where((gt['Tumor grade'] == "High grade") &  (gt['Data'] == 'Synthetic') &  (ans[i]['Tumor grade'] == 'High grade'))[0])/gt_actualSynHigh)

# %%  plot bar chart with scatter points for each pathologist performance on Real vs synthetic accuracy
species = ("Low Grade", "High Grade")
means = {
    'Real':  (0.25*np.sum(ans_predReal_actualRealLow),0.25*np.sum(ans_predReal_actualRealHigh)),
    'Synthetic': (0.25*np.sum(ans_predSyn_actualSynLow),0.25*np.sum(ans_predSyn_actualSynHigh))
        }

mean_per_ans = np.array([ans_predReal_actualRealLow, ans_predReal_actualRealHigh, ans_predSyn_actualSynLow, ans_predSyn_actualSynHigh])

x = np.arange(len(species))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0

plt.figure(figsize=(12, 6))
fig, ax = plt.subplots(layout='constrained')

colors_list = ['skyblue', 'orange']
for attribute, measurement in means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color = colors_list[multiplier])
    # ax.bar_label(rects, padding=3)
    multiplier += 1

offset = width * multiplier

ans_color = ['blue', 'green', 'red', 'purple']
for i in range(2):
    for j in range(4):
        offset = width * i+0.02*j-0.02
        plt.scatter(x + offset, mean_per_ans[2*i:2*i+2,j], s=100, c=ans_color[j], marker='o', label=f'Reviewer {j+1}')
    if(i ==0):
        ax.legend(fontsize = 18, bbox_to_anchor=(1.05, 1.0), loc='upper left') 
        
# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Tumor Grading Accuracy', fontsize = 20)
ax.set_ylabel('Real vs synthetic accuracy', fontsize = 20)
ax.set_xticks(x + 0.5*width, ['Low grade\n(1, 2)','High grade\n(3, 4)' ], fontsize = 20)
plt.yticks(fontsize = 18)
ax.set_ylim(0, 1)
#add threshold line 0.5
plt.axhline(y=0.5, color='r', linestyle='--')
plt.savefig('./image/Real_syn_accuracy_all4pathologist_withMeanperAns.png',dpi = 300, bbox_inches = 'tight')
# 


# %%  plot bar chart with scatter points for each pathologist performance on Low and High grade prediction
species = ("Real", "Synthetic")
means = {
    'Low Grade': (0.25*np.sum(ans_predLow_actualRealLow),0.25*np.sum(ans_predLow_actualSynLow)),
    'High Grade': (0.25*np.sum(ans_predHigh_actualRealHigh),0.25*np.sum(ans_predHigh_actualSynHigh))
        }

mean_per_ans = np.array([ans_predLow_actualRealLow, ans_predLow_actualSynLow,ans_predHigh_actualRealHigh, ans_predHigh_actualSynHigh])

x = np.arange(len(species))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0

plt.figure(figsize=(12, 6))
fig, ax = plt.subplots(layout='constrained')

colors_list = ['palegreen', 'lightpink']
for attribute, measurement in means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color = colors_list[multiplier])
    # ax.bar_label(rects, padding=3)
    multiplier += 1

offset = width * multiplier

ans_color = ['blue', 'green', 'red', 'purple']
for i in range(2):
    for j in range(4):
        offset = width * i+0.02*j-0.02
        plt.scatter(x + offset, mean_per_ans[2*i:2*i+2,j], s=100, c=ans_color[j], marker='o', label=f'Reviewer {j+1}')
    if(i ==0):
        ax.legend(fontsize = 18, bbox_to_anchor=(1.05, 1.0), loc='upper left') 
        
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Tumor grading accuracy', fontsize = 20)
# ax.set_ylabel('Real vs Synthetic Accuracy', fontsize = 20)
ax.set_xticks(x + 0.5*width, species, fontsize = 20)
plt.yticks(fontsize = 18)
ax.set_ylim(0, 1)
plt.savefig('./image/Low_high_accuracy_all4pathologist_withMeanperAns.png',dpi = 300, bbox_inches = 'tight')


# %% Agreement Levels with Ground Truth on Tumor Grading Prediction
import numpy as np
import matplotlib.pyplot as plt
ans_low_high = []
for i in range(4):
    ans_low_high.append(ans[i]['Tumor grade'].tolist())
categories = [100, 75, 50, 25, 0]

# Plot the results in a single bar with different colors
categories_labels = ['100%', '75%', '50%', '25%', '0%']
colors = ['tomato', 'lightsalmon', 'azure', 'lightblue', 'steelblue']
ax = plt.figure(figsize=(12, 3))


ans_low_high = np.array(ans_low_high).transpose()
low_high_list = ['High grade', 'Low grade']
data_list = ['High grade-Real', 'High grade-Synthetic', 'Low grade-Real', 'Low grade-Synthetic']
for low_high_idx, low_high_value in enumerate(low_high_list):
    
    real_syn_list = ['Real', 'Synthetic']
    for real_syn_idx, real_syn_value in enumerate(real_syn_list): 
        
        data = ans_low_high[np.where((gt['Tumor grade'] == low_high_value) & (gt['Data'] == real_syn_value))[0], :]
        #set 0 for not data_name and 1 for data_name in data array
        data[data != low_high_value] = 0
        data[data == low_high_value] = 1
        data= data.astype(int)
        # Calculate the percentage of agreement for each image
        agreement_levels = np.sum(data, axis=1)

        category_counts = [0,0,0,0,0]
        for i in range(5):
            category_counts[i] = len(np.where(np.sum(data, axis=1) == 4-i)[0])
        

        # Convert counts to percentages
        total_images = len(agreement_levels)
        category_percentages = [count / total_images * 100 for count in category_counts]


        bottom = 0

        for i, (percent, color) in enumerate(zip(category_percentages, colors)):
            plt.barh(low_high_idx*2+real_syn_idx, percent, left=bottom, color=color, edgecolor='black')
            bottom += percent

        print(low_high_idx*2+real_syn_idx)
        for i, v in enumerate(category_percentages):
            if v > 0.2:
                plt.text(np.sum(category_percentages[:i])+(v/2), low_high_idx*2+real_syn_idx, f"{v:.1f}", color='black', ha='center', va='center', fontsize = 20)
        # if data_idx == 0:
        #     legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 20)
            
plt.yticks(range(len(data_list)),data_list, fontsize = 20)
plt.xlabel('Percentage', fontsize = 20)
plt.ylabel('Actual', fontsize = 20)
plt.xticks(fontsize = 20)
plt.title('Agreement Levels with Ground Truth on Tumor Grading Prediction', fontsize = 20)
plt.xlim(0, 100)

proxy_artists = []
for i, category in enumerate(categories_labels):
    proxy_artists.append(plt.Rectangle((0, 0), 1, 1, color=colors[i], edgecolor='black'))

# Create a custom legend that includes all categories
legend = plt.legend(proxy_artists, categories_labels,loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 20)

# Modify the edge color of the legend handles
for legend_handle in legend.legendHandles:
    legend_handle.set_edgecolor('black')

plt.savefig('./image/Agreement_Levels_on_Low_and_High_Prediction_4group.png',dpi = 300, bbox_inches = 'tight')
plt.show()

# %% Agreement Levels with Ground Truth on Real and Synthetic Prediction
import numpy as np
import matplotlib.pyplot as plt
ans_real_syn = []
for i in range(4):
    ans_real_syn.append(ans[i]['Data'].tolist())
ans_real_syn = np.array(ans_real_syn).transpose()
categories = [100, 75, 50, 25, 0]

# Plot the results in a single bar with different colors
categories_labels = ['100%', '75%', '50%', '25%', '0%']
colors = ['tomato', 'lightsalmon', 'azure', 'lightblue', 'steelblue']
ax = plt.figure(figsize=(12, 3))


ans_low_high = np.array(ans_low_high).transpose()
low_high_list = ['High grade', 'Low grade']
real_syn_list = ['Real', 'Synthetic']
data_list = ['Real-High grade', 'Real-Low grade', 'Synthetic-High grade', 'Synthetic-Low grade']
    
for real_syn_idx, real_syn_value in enumerate(real_syn_list): 
    for low_high_idx, low_high_value in enumerate(low_high_list):        
        data = ans_real_syn[np.where((gt['Tumor grade'] == low_high_value) & (gt['Data'] == real_syn_value))[0], :]
        #set 0 for not data_name and 1 for data_name in data array
        data[data != real_syn_value] = 0
        data[data == real_syn_value] = 1
        data= data.astype(int)
        # Calculate the percentage of agreement for each image
        agreement_levels = np.sum(data, axis=1)

        category_counts = [0,0,0,0,0]
        for i in range(5):
            category_counts[i] = len(np.where(np.sum(data, axis=1) == 4-i)[0])

        # Convert counts to percentages
        total_images = len(agreement_levels)
        category_percentages = [count / total_images * 100 for count in category_counts]


        bottom = 0

        for i, (percent, color) in enumerate(zip(category_percentages, colors)):
            plt.barh(low_high_idx*2+real_syn_idx, percent, left=bottom, color=color, edgecolor='black')
            bottom += percent


        for i, v in enumerate(category_percentages):
            if v > 0.2:
                plt.text(np.sum(category_percentages[:i])+(v/2), low_high_idx*2+real_syn_idx, f"{v:.1f}", color='black', ha='center', va='center', fontsize = 20)
        # if data_idx == 0:
        #     legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 20)
            
plt.yticks(range(len(data_list)),data_list, fontsize = 20)
plt.xlabel('Percentage', fontsize = 20)
plt.ylabel('Actual', fontsize = 20)
plt.xticks(fontsize = 20)
plt.title('Agreement Levels with Ground Truth on Real and Synthetic Prediction', fontsize = 20)
plt.xlim(0, 100)

proxy_artists = []
for i, category in enumerate(categories_labels):
    proxy_artists.append(plt.Rectangle((0, 0), 1, 1, color=colors[i], edgecolor='black'))

# Create a custom legend that includes all categories
legend = plt.legend(proxy_artists, categories_labels,loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 20)

# Modify the edge color of the legend handles
for legend_handle in legend.legendHandles:
    legend_handle.set_edgecolor('black')

plt.savefig('./image/Agreement_Levels_on_Real_Synthetic_Prediction_4group.png',dpi = 300, bbox_inches = 'tight')
plt.show()

# %% compare tumor grade prediction between synthetic and real data (correct vs incorrect)

#grount truth for synthetic and real
data_gt = gt['Data']
data_gt_x4 = np.concatenate([data_gt]*4)

grade_gt = gt['Tumor grade']
grade_gt_x4 = np.concatenate([grade_gt]*4)

ans = np.array(ans)
data_ans = ans[:,:,1]
data_ans = np.concatenate(data_ans)

grade_ans = ans[:,:,0]
grade_ans = np.concatenate(grade_ans)

correct_grade_ans = grade_ans == grade_gt_x4

#create fisher exact table for 2 group of data with 2 catogories in correct_grade_ans

import scipy.stats as stats
table = np.zeros((2,2))
table[0,0] = np.sum(correct_grade_ans & (data_gt_x4 == 'Real'))
table[0,1] = np.sum(correct_grade_ans & (data_gt_x4 == 'Synthetic'))
table[1,0] = np.sum(~correct_grade_ans & (data_gt_x4 == 'Real') )
table[1,1] = np.sum(~correct_grade_ans & (data_gt_x4 == 'Synthetic') )

print(table)

oddsratio, pvalue = stats.fisher_exact(table)
print(pvalue)#plot table with sns.heatmap remove colorbar
import seaborn as sns
plt.figure(figsize=(4,4))
sns.heatmap(table, annot=True, cbar=False, annot_kws={"fontsize": 20})
#replace 0,1 by Real, Synthetic
plt.xticks([0.5,1.5], ['Real', 'Synthetic'], fontsize=20)
plt.yticks([0.5,1.5], ['Correct', 'Incorrect'], fontsize=20)


#%% compare tumor grade prediction between synthetic and real data (all correct vs not all correct)
grade_ans = ans[:,:,0]
# print 1 if all 4 pathologists have correct answer
all_correct = []
for question_idx in range(grade_ans.shape[1]):
    all_correct.append(np.all(grade_ans[:,question_idx] == gt['Tumor grade'][question_idx]))
    
table = np.zeros((2,2))
table[0,0] = np.sum(all_correct & (data_gt == 'Real'))
table[0,1] = np.sum(all_correct & (data_gt == 'Synthetic') )
table[1,0] = np.sum((np.array(all_correct) == False) & (data_gt == 'Real'))
table[1,1] = np.sum((np.array(all_correct) == False) & (data_gt == 'Synthetic'))

oddsratio, pvalue = stats.fisher_exact(table)
print(pvalue)#plot table with sns.heatmap remove colorbar
import seaborn as sns
plt.figure(figsize=(4,4))
sns.heatmap(table, annot=True, cbar=False, annot_kws={"fontsize": 20})
#replace 0,1 by Real, Synthetic
plt.xticks([0.5,1.5], ['Real', 'Synthetic'], fontsize=20)
plt.yticks([0.5,1.5], ['All correct', 'Not all'], fontsize=20)



import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa
# ans_name = 'Tumor grade'
ans_name = 'Data'
ans_cluster = {'Data': ['Real', 'Synthetic'], 'Tumor grade': ['Low grade', 'High grade']}
df = pd.DataFrame({
    "r1": ans[0][ans_name],
    "r2": ans[1][ans_name],
    "r3": ans[2][ans_name],
    "r4": ans[3][ans_name],
})

# Convert to Fleiss format: [count_real, count_fake] per image
table = []
for _, row in df.iterrows():
    counts = [
        (row == ans_cluster[ans_name][0]).sum(),
        (row == ans_cluster[ans_name][1]).sum()
    ]
    table.append(counts)

table = np.array(table)

kappa = fleiss_kappa(table)
print("Fleiss' kappa:", kappa)
raters = ['r1', 'r2', 'r3', 'r4']

#plot the cohen kappa scores
from sklearn.metrics import cohen_kappa_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(4, 5))
kappa_scores = []
pair_list = []
for r1, r2 in itertools.combinations(raters, 2):
    k = cohen_kappa_score(df[r1], df[r2])
    kappa_scores.append(k)
    pair_list.append(f"{r1} vs {r2}")
#plot barplot all in gray color
sns.barplot(x=pair_list, y=kappa_scores, color='gray')
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=16)

plt.ylim(0, 1)
plt.ylabel("Cohen's Kappa Score", fontsize=20)
# plt.ylabel("Cohen's Kappa Score")
if ans_name == 'Data':
    plt.title(f"Inter-Rater agreement \n on real vs synthetic", fontsize=20)
else:
    plt.title(f"Inter-Rater agreement \n on tumor grade", fontsize=20)
    
plt.tight_layout()
plt.show()



from sklearn.metrics import accuracy_score, cohen_kappa_score

# Replace with your true labels (length = 40)
truth = pd.Series(gt[ans_name])
kappa_scores = []

for col in df.columns:
    acc = accuracy_score(truth, df[col])
    kappa = cohen_kappa_score(truth, df[col])
    print(f"{col}: Accuracy={acc:.4f}, Kappa={kappa:.4f}")
    kappa_scores.append(kappa)
    
plt.figure(figsize=(4, 5))
sns.barplot(x=df.columns, y=kappa_scores, color='gray')
plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=16)
plt.ylabel("Cohen's Kappa Score", fontsize=20)
plt.ylim(0, 1)
plt.title(f"Agreement of Each Pathologist with Ground Truth\n on {ans_name}")
plt.tight_layout()
plt.show()




import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa
ans_name = 'Tumor grade'
# ans_name = 'Data'
ans_cluster = {'Data': ['Real', 'Synthetic'], 'Tumor grade': ['Low grade', 'High grade']}
df = pd.DataFrame({
    "r1": ans[0][ans_name],
    "r2": ans[1][ans_name],
    "r3": ans[2][ans_name],
    "r4": ans[3][ans_name],
})

# Convert to Fleiss format: [count_real, count_fake] per image
table = []
for _, row in df.iterrows():
    counts = [
        (row == ans_cluster[ans_name][0]).sum(),
        (row == ans_cluster[ans_name][1]).sum()
    ]
    table.append(counts)

table = np.array(table)

kappa = fleiss_kappa(table)
print("Fleiss' kappa:", kappa)
raters = ['r1', 'r2', 'r3', 'r4']

#plot the cohen kappa scores
from sklearn.metrics import cohen_kappa_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(4, 5))
kappa_scores = []
pair_list = []
for r1, r2 in itertools.combinations(raters, 2):
    k = cohen_kappa_score(df[r1], df[r2])
    kappa_scores.append(k)
    pair_list.append(f"{r1} vs {r2}")
#plot barplot all in gray color
sns.barplot(x=pair_list, y=kappa_scores, color='gray')

mean_kappa = np.mean(kappa_scores)
plt.axhline(mean_kappa, color='red', linestyle='--', label=f'Mean Kappa: {mean_kappa:.2f}')
plt.legend(fontsize=16)
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=16)

plt.ylim(-0.1, 1)
plt.ylabel("Cohen's Kappa Score", fontsize=20)
plt.ylim(-0.02, 1)
plt.ylabel("Cohen's Kappa Score")
if ans_name == 'Data':
    plt.title(f"Inter-Rater agreement \n on real vs synthetic", fontsize=20)
else:
    plt.title(f"Inter-Rater agreement \n on tumor grade", fontsize=20)
    
plt.tight_layout()
plt.show()



from sklearn.metrics import accuracy_score, cohen_kappa_score

# Replace with your true labels (length = 40)
truth = pd.Series(gt[ans_name])
kappa_scores = []

for col in df.columns:
    acc = accuracy_score(truth, df[col])
    kappa = cohen_kappa_score(truth, df[col])
    print(f"{col}: Accuracy={acc:.4f}, Kappa={kappa:.4f}")
    kappa_scores.append(kappa)
    
plt.figure(figsize=(4, 5))
sns.barplot(x=df.columns, y=kappa_scores, color='gray')
#plot mean kappa score as horizontal line
mean_kappa = np.mean(kappa_scores)
plt.axhline(mean_kappa, color='red', linestyle='--', label=f'Mean Kappa: {mean_kappa:.2f}')
plt.legend(fontsize=16)
plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=16)
plt.ylabel("Cohen's Kappa Score", fontsize=20)
plt.ylim(-0.02, 1)
# plt.title(f"Agreement of Each Pathologist with Ground Truth\n on {ans_name}")
plt.tight_layout()
plt.show()