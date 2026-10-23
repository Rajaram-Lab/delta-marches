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
#%% This is the code to generate hypothesis of tumor grading rule for the paper. This consists of
# 1. Compute the Delta march heatmap for each grade transition and plot the boxplot for each grade transition
# 2. Compute the T-tests of tumor nuclei area between real and synthetic data
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
from function_define import *

#%% load encoder model
class Encoder(nn.Module):
    def __init__(self, nc, nf, nz):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, nf, 5, 2, 2, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf, nf*2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*2, nf*4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),            
            
            nn.Conv2d(nf*4, nf*4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(nf*4, nf*4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(in_features=7*7*nf*4, out_features=nz),
        )

    def forward(self, input):
        y =  self.net(input)
        return y

#%%load data
import pickle

#real data
import yaml
yaml_paths = './data/data_dir.yaml'
with open(yaml_paths, 'r') as yaml_file:
    file_paths = yaml.safe_load(yaml_file)
    
  
# save_dir = file_paths['synthetic_data']['synthetic_data_grade1to4']

import argparse

parser = argparse.ArgumentParser()


parser.add_argument(
    "data_type",
    type=str,
    help="Type of the data (real/syn)"
)

parser.add_argument(
    "save_dir",
    type=str,
    help="Path to the saved pickle file"
)

args = parser.parse_args()
save_dir = args.save_dir
data_real_syn = args.data_type
start_grd = 0
# start_grd = 3

jaccard_index_alldummytype_list = []
Component_alldummytype_list = []
grade_alldummytype_list = []
dummytype_list = []
if data_real_syn == 'syn':
    save_dir = args.save_dir
    with open(save_dir,'rb') as f:
        traverse_data = pickle.load(f)
    heatmap_type_list = ['same_grade', 'full_grade']
        
    #%% Obtain heatmap for observed image and plot boxplot for each grade transition

    #Pass images through encoder/cls filter
    nz=1024
    enc_Path = file_paths['WEIGHTS']['AAE']['Encoder']['n1024']
    # enc_Path = f'./checkpoints/AAE_model_n{nz}_enc.pth'
    enc = Encoder(nc=3, nf=256, nz=nz)
    enc.load_state_dict(torch.load(enc_Path))
    enc = enc.to(device)
    enc.eval()
    sub_model = enc.net[:9]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    #save_path = file_paths['WEIGHTS']['grade_classifier']
    # model_cls = torchvision.models.vgg19(pretrained= True)
    # model_cls.classifier[6] = nn.Linear(in_features = 4096, out_features = 4, bias=True)
    # model_cls.load_state_dict(torch.load(save_path))
    # model_cls.eval()
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # model = model_cls.to(device)
    # sub_model = model.features[:10]

    #%% Patient_across transition
    #load Patient_acros transition data

    data_type = 'same_grade'
    #compute heatmap for patient_across transition
    observed_img = []
    observed_img.append(traverse_data['img'][traverse_data['full_grade_list'][:,0,0],traverse_data['full_grade_list'][:,0,1],:,:,:])
    observed_img.append(traverse_data['patient_across_img'])
    observed_class = traverse_data['class']
    observed_img = np.array(observed_img).transpose(1,0,2,3,4)
    array_list = []
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    for i in range(observed_img.shape[0]):
        batch = []
        for j in range(observed_img.shape[1]):
            batch.append(transform(observed_img[i,j]))
        batch = torch.stack(batch).to(device)
        array_list.append(sub_model(batch).cpu().detach().numpy())
    array_list = np.array(array_list)

    #Compute the delta array and rescale it to 224x224
    Delta_array_list = np.squeeze(array_list[:,1:,:,:]-array_list[:,:-1,:,:])
    mean_rescale_delta_array_list = []
    for i in range(Delta_array_list.shape[0]):
        rescale_delta_array_list_i = []
        for j in range(Delta_array_list.shape[1]):
            rescale_delta_array_list_i.append(skimage.transform.resize(Delta_array_list[i][j], [224, 224], order = 2))
        rescale_delta_array_list_i = np.array(rescale_delta_array_list_i)
        mean_rescale_delta_array_list.append(np.abs(rescale_delta_array_list_i).mean(axis =0))
    mean_rescale_delta_array_list = np.array(mean_rescale_delta_array_list)
    #Compute mean over channels
    # mean_rescale_delta_array_list =  np.abs(rescale_delta_array_list).mean(axis =1)    

    traverse_data['mean_abs_transform_AAEfilter09_patient_across'] = mean_rescale_delta_array_list

        
    #load full grade transition data
    data_type = 'full_grade'
            
    full_grade_list = np.arange(0,traverse_data['full_grade_list'].shape[0])
    #shuffle rows in full_grade_list 
    full_grade_idx = traverse_data['full_grade_list']
    observed_class = traverse_data['class'][full_grade_idx[:,:,0],full_grade_idx[:,:,1]]
    observed_img = traverse_data['img'][full_grade_idx[:,:,0],full_grade_idx[:,:,1], :,:,:]
    observed_nuclei_mask = traverse_data['full_grade_list_Nuc_seg']
    observed_vas_mask = traverse_data['full_grade_list_vas']
    observed_nuclei_mask = np.array(observed_nuclei_mask)
    observed_class = np.array(observed_class)        
    observed_img = np.array(observed_img)
    transform = transforms.ToTensor()
    array_list = []

    array_list = []
    for i in range(observed_img.shape[0]):
        batch = []
        for j in range(observed_img.shape[1]):
            batch.append(transform(observed_img[i,j]))
        batch = torch.stack(batch).to(device)
        array_list.append(sub_model(batch).cpu().detach().numpy())
    array_list = np.array(array_list)

    #Compute the delta array and rescale it to 224x224
    Delta_array_list = array_list[:,1:,:,:]-array_list[:,:-1,:,:]
    import skimage
    mean_rescale_delta_array_list = []
    for i in range(Delta_array_list.shape[0]):
        mean_rescale_delta_array_list_i = []
        for j in range(Delta_array_list.shape[1]):
            rescale_delta_array_list_i_j = []
            for k in range(Delta_array_list.shape[2]):
                rescale_delta_array_list_i_j.append(skimage.transform.resize(Delta_array_list[i][j][k], [224, 224], order = 2))
            rescale_delta_array_list_i_j = np.array(rescale_delta_array_list_i_j)
            mean_rescale_delta_array_list_i.append(np.abs(rescale_delta_array_list_i_j).mean(axis = 0))
        mean_rescale_delta_array_list_i = np.array(mean_rescale_delta_array_list_i)
        mean_rescale_delta_array_list.append(mean_rescale_delta_array_list_i)
    mean_rescale_delta_array_list = np.array(mean_rescale_delta_array_list)

    traverse_data['mean_abs_transform_AAEfilter09_grade_across'] = mean_rescale_delta_array_list

    save_dir = args.save_dir
    with open(save_dir, 'wb') as f:
        pickle.dump(traverse_data, f, protocol = 4)
    data_type = 'jump'
    #compute heatmap for patient_across transition
    observed_img = []
    observed_img.append(traverse_data['img'][traverse_data['full_grade_list'][:,0,0],traverse_data['full_grade_list'][:,0,1],:,:,:])
    observed_img.append(traverse_data['img'][traverse_data['full_grade_list'][:,3,0],traverse_data['full_grade_list'][:,3,1],:,:,:])
    observed_class = traverse_data['class']
    observed_img = np.array(observed_img).transpose(1,0,2,3,4)
    array_list = []
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    for i in range(observed_img.shape[0]):
        batch = []
        for j in range(observed_img.shape[1]):
            batch.append(transform(observed_img[i,j]))
        batch = torch.stack(batch).to(device)
        array_list.append(sub_model(batch).cpu().detach().numpy())
    array_list = np.array(array_list)

    #Compute the delta array and rescale it to 224x224
    Delta_array_list = np.squeeze(array_list[:,1:,:,:]-array_list[:,:-1,:,:])
    mean_rescale_delta_array_list = []
    for i in range(Delta_array_list.shape[0]):
        rescale_delta_array_list_i = []
        for j in range(Delta_array_list.shape[1]):
            rescale_delta_array_list_i.append(skimage.transform.resize(Delta_array_list[i][j], [224, 224], order = 2))
        rescale_delta_array_list_i = np.array(rescale_delta_array_list_i)
        mean_rescale_delta_array_list.append(np.abs(rescale_delta_array_list_i).mean(axis =0))
    mean_rescale_delta_array_list = np.array(mean_rescale_delta_array_list)

    traverse_data['mean_abs_transform_AAEfilter09_grade_across_jump'] = mean_rescale_delta_array_list
# save_dir = file_paths['synthetic_data']['synthetic_data_grade1to4']
# with open(save_dir,'wb') as f:
#     pickle.dump(traverse_data, f, protocol=4)

#%% this code is to plot the jaccard index of each component for each grade transition
# Compute Jaccard rate for each component

    start_grd = 0
    save_dir = args.save_dir
    # start_grd = 3
    # save_dir = file_paths['synthetic_data']['synthetic_data_grade4to1']
    with open(save_dir,'rb') as f:
        observed_data = pickle.load(f)
else:
     
    heatmap_type_list = ['GradCam_mask']
    save_dir = args.save_dir
    with open(save_dir,'rb') as f:
        real_data = pickle.load(f)
    observed_data = {}
    observed_data['full_grade_list_vas'] = []
    observed_data['full_grade_list_Nuc_seg'] = []
    observed_data['full_grade_list_gradcam'] = {'Gradcampp_mask': []}
    for grade in range(4):
        idx = np.where(real_data['classes']== grade)[0]
        observed_data['full_grade_list_vas'].append(real_data['vas'][idx])
        observed_data['full_grade_list_Nuc_seg'].append(real_data['Nuc_Seg'][idx])
        observed_data['full_grade_list_gradcam']['Gradcampp_mask'].append(real_data['Gradcam']['Gradcampp_mask'][idx])
    observed_data['full_grade_list_vas'] = np.stack(observed_data['full_grade_list_vas'], axis=0).transpose(1,0,2,3)
    observed_data['full_grade_list_Nuc_seg'] = np.stack(observed_data['full_grade_list_Nuc_seg'], axis=0).transpose(1,0,2,3)
    observed_data['full_grade_list_gradcam']['Gradcampp_mask'] = np.stack(observed_data['full_grade_list_gradcam']['Gradcampp_mask'], axis=0).transpose(1,0,2,3)
    observed_data['dummy_list'] = np.arange(1000)
        
if 'GradCam_mask' in heatmap_type_list:
    dummy_list = [0]
    start_grd = 0
else:
    dummy_list = [1, 0]
    

for dummy in dummy_list:
    idx_sample = np.arange(observed_data['full_grade_list_Nuc_seg'].shape[0])
    jaccard_index_list = []
    Component_list = []
    grade_list = []

    # Processing samples and calculating heatmaps
    for heatmap_type in heatmap_type_list:
        idx_grade_list = start_grd
        if heatmap_type == 'same_grade':
            heatmap_array = observed_data['mean_abs_transform_AAEfilter09_patient_across'][:,None,:,:]
        elif heatmap_type == 'full_grade':
            heatmap_array = observed_data['mean_abs_transform_AAEfilter09_grade_across'][idx_sample]
        elif heatmap_type == 'GradCam_mask':
            gradcam_model = 'Gradcampp_mask' #'AblationCAM_mask', 'Gradcam_mask', 'Gradcampp_mask
            heatmap_array = observed_data['full_grade_list_gradcam'][gradcam_model][idx_sample]
            idx_grade_list = np.concatenate([0, 1, 2, 3]*len(idx_sample), axis=None)
        observed_masks = process_masks(observed_data, start_grd, idx_sample, heatmap_type)
        heatmap_list, component_name, grade = process_samples(
            len(idx_sample),
            heatmap_array,
            observed_masks, observed_data['dummy_list'], dummy, idx_grade_list, heatmap_type
        )
        jaccard_index_list.append(heatmap_list)
        Component_list.append(component_name)
        grade_list.append(grade)
    jaccard_index_list = np.concatenate(jaccard_index_list, axis=None)
    Component_list = np.concatenate(Component_list, axis=None)
    grade_list = np.concatenate(grade_list, axis=None)
    jaccard_index_alldummytype_list.append(jaccard_index_list.copy())
    Component_alldummytype_list.append(Component_list.copy())
    grade_alldummytype_list.append(grade_list.copy())
    if dummy == 1:
        dummytype_list.append(['Dummy']*len(grade_list))
    else:
        dummytype_list.append(['True']*len(grade_list))
  
import seaborn as sns
import pandas as pd
# components = ['Tumor nuclei','Vasculature',  'Immune nuclei', 'Other nuclei','Background']
components = ['Tumor nuclei', 'Vasculature',  'NonTumor nuclei','Background']

if dummy == 0:
    df = pd.DataFrame({'A': Component_list, 'B':grade_list , 'C': jaccard_index_list})

    # Step 2: Calculate the mean of C for each A-B group
    group_means = df.groupby(['A', 'B']).C.mean().reset_index()
    group_std= df.groupby(['A', 'B']).C.std().reset_index()
    pivot_table = np.asarray(group_means['C']).reshape(len(components),4)

    labels = (np.asarray(["{:.2f}".format(string)
                        for string in group_means['C']])
            ).reshape(len(components),4)
    # Plot heatmap
    plt.figure(figsize=(6, 4))
    
    if start_grd == 0:
        sns.heatmap(pivot_table, annot=labels, fmt = "", cmap='coolwarm', annot_kws={"fontsize": 14})
        
        if 'GradCam_mask' in heatmap_type_list:
            plt.xticks(np.arange(4)+0.5, ['1','2', '3', '4'], fontsize=14)
        else:
            plt.xticks(np.arange(4)+0.5, ['1to1','1to2', '2to3', '3to4'], fontsize=14)
    else:
        sns.heatmap(np.flip(pivot_table, axis = 1), annot=np.flip(labels, axis = 1), fmt = "", cmap='coolwarm', annot_kws={"fontsize": 14})
        if 'GradCam_mask' in heatmap_type_list:
            plt.xticks(np.arange(4)+0.5, ['4','3', '2', '1'], fontsize=14)
        else:
            plt.xticks(np.arange(4)+0.5, ['4to4','4to3', '3to2', '2to1'], fontsize=14)
    plt.yticks(np.arange(len(components))+0.5, sorted(components), rotation=0, fontsize=14)
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig('./image/heatmap_jaccard_rate.png')
    plt.show()

if 1 in dummy_list:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import ttest_ind

    # plot boxplot for each grade transition with dummy and true data
    grade_alldummytype_list = np.concatenate(grade_alldummytype_list, axis=None)
    Component_alldummytype_list = np.concatenate(Component_alldummytype_list, axis=None)
    dummytype_list = np.concatenate(dummytype_list, axis=None)
    jaccard_index_alldummytype_list = np.concatenate(jaccard_index_alldummytype_list, axis=None)

    transition_name_list = ['1to1', '1to2', '2to3', '3to4']
    # transition_name_list = ['4to4', '4to3', '3to2', '2to1']
    for transition_name in transition_name_list:
        idx_sample = np.where(grade_alldummytype_list == transition_name)[0]
        df = pd.DataFrame({
            'Jaccard':  list(jaccard_index_alldummytype_list[idx_sample]),
            'Mask_type': list(dummytype_list[idx_sample]),
            'Component': list(Component_alldummytype_list[idx_sample]),
        })
        # Sample data
        plt.figure(figsize=(8, 3))
        hue_colors = {'Dummy': 'red', 'True': 'blue'}
        sns.boxplot(x="Component", y="Jaccard", hue="Mask_type", order = components, data=df,showfliers = False, width= 0.43, palette=hue_colors)

        for i, component in enumerate(components):
            dummy_data = df[(df['Component'] == component) & (df['Mask_type'] == 'Dummy')]['Jaccard']
            true_data = df[(df['Component'] == component) & (df['Mask_type'] == 'True')]['Jaccard']
            
            p_value = ttest_ind(dummy_data, true_data, alternative= 'less').pvalue
            
            # Add p-value annotations
            x1, x2 = i - 0.2, i + 0.2  # x positions for the annotation

            y, h, col = 40, 0.2, 'k'  # y position, height of the line, color of the line
            
            plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1, color=col)
            plt.text((x1 + x2) * 0.5, y + h, f"p = {p_value:.2e}", ha='center', va='bottom', color=col, fontsize=12)
        #plot median line of true data
        true_data = df[(df['Mask_type'] == 'True')]['Jaccard']
        plt.axhline(y=true_data.mean(), color='black', linestyle='--')
        # Customizing the plot
        # remove the legend
        plt.legend('')
        plt.tight_layout()
        plt.xlabel('')
        plt.ylabel('')
        plt.ylim(0, 50)
        plt.xticks(fontsize=12)
        plt.xticks(np.arange(len(components)), components, fontsize=14)
        plt.savefig('./image/heatmap_jaccard_ratio.png')
        plt.show()