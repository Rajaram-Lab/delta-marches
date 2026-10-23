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
# define function 

import torch
import pandas as pd
import os,sys
from PIL import Image
device = torch.device('cuda:0')
import time
from torch import nn, cuda, device
from torch.autograd import Variable
import torchvision
from torch.utils.data import Dataset# data management
import numpy as np
import os,sys
                                       
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
import yaml
from skimage.morphology import extrema
from skimage.color import rgb2hed
from scipy import ndimage
from scipy.stats import mannwhitneyu
import seaborn as sns
#%% Jaccard rate computation

mask_dict = {'Other': 0, 'Vas': 1, 'NonTumor': 2, 'Tumor': 3}

# detect ghost nuclei
def ghost_nuclei_mask_detection(mask1,mask2):
    ghost_nuclei_mask = np.maximum(mask1, mask2)
    nObj, labelTest, statsTa, centroids=cv2.connectedComponentsWithStats(\
                np.uint8(ghost_nuclei_mask>0),8,cv2.CV_32S)
    for i in range(1,nObj):
        ghost_nuclei_detect = np.sum((labelTest == i)*(mask1 > 0)*(mask2 > 0))
        if ghost_nuclei_detect > 0:
            ghost_nuclei_mask[labelTest == i] = 0
    ghost_nuclei_mask = ghost_nuclei_mask > 0
    return ghost_nuclei_mask

# merge nuclei and vasculature masks
def merge_nuclei_vas_mask(nuclei_mask1, nuclei_mask2, vas_mask1, vas_mask2):
    #output mask: Other : 0, vas : 1, non-tumor: 2, tumor: 3
    nuclei_mask1 = nuclei_mask1.astype(np.uint8)
    nuclei_mask2 = nuclei_mask2.astype(np.uint8)
    vas_mask1 = vas_mask1.astype(np.uint8)
    vas_mask2 = vas_mask2.astype(np.uint8)
    ghost_nuclei_mask = ghost_nuclei_mask_detection(nuclei_mask1, nuclei_mask2)
    
    nuclei_mask = np.maximum(nuclei_mask1, nuclei_mask2)
    nuclei_mask[ghost_nuclei_mask] = 0
                        
    #merge vasculature masks
    vas_mask = np.maximum(vas_mask1, vas_mask2)
    
    #start merging the masks to have 0,1,2,3 values for Other, vasculature, non tumor nuclei, tumor nuclei
    nuclei_vas_mask = vas_mask.copy()
    
    for i in range(1,len(mask_dict.keys())-1):
        nuclei_vas_mask[(nuclei_mask == i) & (vas_mask == 0)] = i+1
        overlap = (nuclei_mask == i) & (vas_mask == 1)
        n_overlap = np.sum(overlap)
        random_list = np.random.choice([i+1,1],n_overlap)
        nuclei_vas_mask[overlap] = random_list
    
    return nuclei_vas_mask, ghost_nuclei_mask
    
# Compute Jaccard index
def jaccard_index(binary_delta_march, segmentation_mask):
        
    intersection = np.sum(binary_delta_march*segmentation_mask)
    # union = np.sum(mask1_label_binary | mask2_label_binary)
    union = np.sum(binary_delta_march | segmentation_mask)
    
    if union == 0:
        return 0  # If both masks are empty in the region, define Jaccard as 1
    else:
        return intersection / union

# mask processing for different transition
def process_masks(observed_data, grd, idx_sample, heatmap_type='same_grade'):
    
    if heatmap_type == 'same_grade':
        observed_vas_mask = []
        observed_nuclei_mask = []

        for i in idx_sample:
            observed_vas_mask.append([observed_data['full_grade_list_vas'][i, grd], observed_data['patient_across_img_vas'][i]])
            observed_nuclei_mask.append([observed_data['full_grade_list_Nuc_seg'][i, grd], observed_data['patient_across_img_Nuc_seg'][i]])
    else:
        observed_vas_mask = observed_data['full_grade_list_vas']
        observed_nuclei_mask = observed_data['full_grade_list_Nuc_seg']
    return {
        "vas": np.array(observed_vas_mask),
        "nuclei": np.array(observed_nuclei_mask),
    }
# compute jaccard index for each class
def calculate_intensities(mask, heatmap, total_intensity):
    return {
        'Vasculature': jaccard_index(heatmap.astype(bool), mask['Vasculature'].astype(bool)) / total_intensity,
        'Tumor nuclei': jaccard_index(heatmap.astype(bool), mask['Tumor nuclei'].astype(bool)) / total_intensity,
        'NonTumor nuclei': jaccard_index(heatmap.astype(bool), mask['NonTumor nuclei'].astype(bool)) / total_intensity,
        'Other': jaccard_index(heatmap.astype(bool), mask['Other'].astype(bool)) / total_intensity,
    }

# main code to process sample to compute jaccard index
def process_samples(num_sample, mean_rescale_delta_array_list, observed_masks, dummy_list, dummy_value, grade, heatmap_type):
    heatmap_list = []
    component = []
    grades = []
    presence = []
    for i in range(num_sample):
        idx = dummy_list[i] if dummy_value == 1 else i
        for j in range(mean_rescale_delta_array_list.shape[1]):
            A = j
            if heatmap_type == 'GradCam_mask':
                B = j
            else:
                B = j+1
            nuclei_vas_mask, ghost_nuclei_mask = merge_nuclei_vas_mask(
                observed_masks['nuclei'][idx, A], observed_masks['nuclei'][idx, B],
                observed_masks['vas'][idx, A], observed_masks['vas'][idx, B]
            )
            temp_mask = nuclei_vas_mask.copy()
            
            mask ={'Tumor nuclei': temp_mask == 3, 
                   'NonTumor nuclei': temp_mask == 2, 
                   'Vasculature': temp_mask == 1, 
                   'Other': temp_mask == 0, 
                }
            heatmap_i = mean_rescale_delta_array_list[i, j].copy()
            heatmap_i[ghost_nuclei_mask] = 0
            heatmap_i = (heatmap_i - heatmap_i.min()) / (heatmap_i.max() - heatmap_i.min())
            heatmap_i[heatmap_i < 0.5] = 0
            heatmap_i[heatmap_i >= 0.5] = 1
            

            total_intensity = 1
            intensities = calculate_intensities(mask, heatmap_i, total_intensity)
                
            if heatmap_type == 'same_grade':
                current_grade = str(grade+1) + 'to' + str(grade+1)
            elif heatmap_type == 'full_grade':
                if grade == 3:
                    current_grade = str(j + 2) + 'to' + str(j + 1)
                else:
                    current_grade = str(j + 1) + 'to' + str(j + 2)
            else:
                current_grade = str(grade[idx]+1)
            for component_name, intensity in intensities.items():
                # if np.max(mask[component_name]) > 0.5:
                heatmap_list.append(intensity)
                component.append(component_name)
                grades.append(current_grade)
                presence.append(np.max(mask[component_name]) > 0.5)
    return np.array(heatmap_list), np.array(component), np.array(grades), np.array(presence)

def process_samples1to4(num_sample, mean_rescale_delta_jump_array_list, observed_masks, dummy_list, dummy_value, start_grd):
#%% Compute Jaccard rate:
    heatmap_list = []
    grades = []
    component = []
    presence = []
    for i in tqdm(range(num_sample)):
        idx = dummy_list[i] if dummy_value == 1 else i
        nuclei_vas_mask, ghost_nuclei_mask = merge_nuclei_vas_mask(
            observed_masks['nuclei'][idx, 0], observed_masks['nuclei'][idx, 1],
            observed_masks['vas'][idx, 0], observed_masks['vas'][idx, 1]
        )
        temp_mask = nuclei_vas_mask.copy()
        
        mask ={'Tumor nuclei': temp_mask == 3, 
                'NonTumor nuclei': temp_mask == 2, 
                'Vasculature': temp_mask == 1, 
                'Other': temp_mask == 0, 
            }
        heatmap_i = mean_rescale_delta_jump_array_list[i].copy()
        heatmap_i[ghost_nuclei_mask] = 0
        heatmap_i = (heatmap_i - heatmap_i.min()) / (heatmap_i.max() - heatmap_i.min())
        heatmap_i[heatmap_i < 0.5] = 0
        heatmap_i[heatmap_i >= 0.5] = 1


        total_intensity = 1
        intensities = calculate_intensities(mask, heatmap_i, total_intensity)
            
        for component_name, intensity in intensities.items():
            heatmap_list.append(intensity)
            component.append(component_name)
            presence.append(np.max(mask[component_name]) > 0.5)
            if start_grd == 0:
                grades.append(['1to4'])
            else:
                grades.append(['4to1'])
                
    return np.array(heatmap_list), np.array(component), np.array(grades), np.array(presence)


# detect nucleoli and compute number of nucleoli per nuclei
def nucleoli_extract(img, nuclei_mask, data_type):
    
    hed=rgb2hed(img)
    hImg=hed[:,:,0]
    hImg=(hImg-np.min(hImg))/(np.max(hImg)-np.min(hImg))
    hImg = ndimage.gaussian_filter(hImg, sigma=1)
    nuclei_name_list = ['-NonTumor', '-Tumor']
    nucleoli_type = []
    nucleoli_num = []
    nuclei_num = []
    # for i in range(1,3):
    for i in range(1,len(mask_dict.keys())-1):
        nuclei_name = nuclei_name_list[i-1]    
        nObj, labelTest, statsT, centroids=cv2.connectedComponentsWithStats(\
                np.uint8(nuclei_mask==i),8,cv2.CV_32S)
        nObj = nObj-1
        if nObj >0:
            temp=extrema.h_maxima(hImg,0.08)
            locR,locC=np.where(temp)
            num_nucleoli = 0
            for nuclei_idx in range(nObj):
                nuclei = labelTest == nuclei_idx+1
                label_list = nuclei[locR,locC]
                label_list = label_list[label_list!=0]
                num_nucleoli += max(0,len(label_list)-1)
            nucleoli_type.append(data_type+nuclei_name)
            nucleoli_num.append(num_nucleoli/nObj)
            nuclei_num.append(nObj)
    return nucleoli_num, nucleoli_type, nuclei_num


import matplotlib.pyplot as plt
#plot mask
def plot_mask(img, nuclei_mask, vas_mask):
    tumColor=np.array([229,140,50])/255
    nonTumoColor=np.array([31,119,180])/255
    vasColor=np.array([64,255,64])/255
    bgColor=np.array([255,255,255])/255
    cmap=np.stack((bgColor,vasColor, nonTumoColor, tumColor))
    
    mask=np.uint8(vas_mask.copy())
    mask[nuclei_mask==1]=2
    mask[nuclei_mask==2]=3
    maskRGB=cmap[mask]
    return maskRGB

# compute mean area of nuclei per patch
def mean_nuclei_area_extract(nuclei_mask, nuclei_class, data_type):
    
    grade_list = []
    nuclei_type_list = []
    Area_list = []
    num_nuclei_list = []
    # nuclei_name_list = ['-Other', '-Tumor', '-Immune']
    nuclei_name_list = ['-NonTumor', '-Tumor']
    
    for i in range(1,len(nuclei_name_list)+1):
        nuclei_name = nuclei_name_list[i-1]    
        nObj, labelTest, statsT, centroids=cv2.connectedComponentsWithStats(\
                np.uint8(nuclei_mask==i),8,cv2.CV_32S)
        if nObj > 1:
            nObj = nObj-1
            nuclei_type_list.append(data_type+nuclei_name)
            Area_list.append(np.mean(statsT[1:,4]))
            grade_list.append([nuclei_class]) 
            num_nuclei_list.append(nObj)

    if len(nuclei_type_list) >0 :
        nuclei_type_list = np.concatenate(nuclei_type_list,axis = None)
        Area_list = np.concatenate(Area_list,axis = None)
        grade_list = np.concatenate(grade_list,axis = None)
        num_nuclei_list = np.concatenate(num_nuclei_list,axis = None)
    return nuclei_type_list, Area_list, grade_list, num_nuclei_list

# perform t test for tumor nuclei area
def t_test_tumor_nuclei_area(Nuc_Seg_low_list_unique, Nuc_Seg_high_list_unique, num_sample, num_test, num_patch_per_group, n_nuclei_per_patch  = 10, test_name = 'ttest_ind'):
    low_grade_area_list = []
    high_grade_area_list = []
    num_high_nuclei = []
    num_low_nuclei = []
    idx_list = []
    for idx in range(Nuc_Seg_low_list_unique.shape[0]):
        #obtain low grade nuclei area
        nObj, labelTest, statsTa, centroids=cv2.connectedComponentsWithStats(\
                np.uint8(Nuc_Seg_low_list_unique[idx]==2),8,cv2.CV_32S)
        nObj = nObj-1
        
        #obtain high grade nuclei area
        nObj, labelTest, statsTb, centroids=cv2.connectedComponentsWithStats(\
                np.uint8(Nuc_Seg_high_list_unique[idx]==2),8,cv2.CV_32S)
        nObj = nObj-1

        if (statsTa.shape[0]>n_nuclei_per_patch and statsTb.shape[0]>n_nuclei_per_patch): #only take example with more than 2 tumor nuclei
            low_grade_area_list.append(statsTa[1:n_nuclei_per_patch+1,4])
            high_grade_area_list.append(statsTb[1:n_nuclei_per_patch+1,4])
            idx_list.append(idx)
            
            # high_grade_list.append(nuclei_dict['high_list_unique'][same_row_high_idx[idx]][0])
            num_high_nuclei.append(statsTb.shape[0]-1)
            num_low_nuclei.append(statsTa.shape[0]-1)

                
    idx_list = np.array(idx_list)
    sample_idx = random.sample(range(len(idx_list)),num_sample)
    num_high_nuclei = np.array(num_high_nuclei)
    num_low_nuclei = np.array(num_low_nuclei)
    low_grade_area_list = [low_grade_area_list[i] for i in sample_idx]
    high_grade_area_list = [high_grade_area_list[i] for i in sample_idx]
    num_high_nuclei = [num_high_nuclei[i] for i in sample_idx]
    num_low_nuclei = [num_low_nuclei[i] for i in sample_idx]
    
    
    # print t-test result with different number of pairs

    pvalue_list = [] #list of t-test for each pair of grade
    statistic_list = []
    for num_test_id in range(num_test):
        idx_sample = random.sample(range(num_sample),num_patch_per_group)
        merge_low_grade_list = []
        merge_high_grade_list = []
        for idx in range(len(idx_sample)):
            merge_low_grade_list.append(low_grade_area_list[idx_sample[idx]])
            merge_high_grade_list.append(high_grade_area_list[idx_sample[idx]])
        merge_low_grade_list = np.concatenate(merge_low_grade_list)
        merge_high_grade_list = np.concatenate(merge_high_grade_list)
        
        if test_name == 'ttest_ind':
            pvalue_list.append(stats.ttest_ind(merge_low_grade_list, merge_high_grade_list, alternative='less').pvalue)
            statistic_list.append(stats.ttest_ind(merge_low_grade_list, merge_high_grade_list, alternative='less').statistic)
        elif test_name == 'mannwhitneyu': 
            pvalue_list.append(stats.mannwhitneyu(merge_low_grade_list, merge_high_grade_list, alternative='less').pvalue)
            statistic_list.append(stats.mannwhitneyu(merge_low_grade_list, merge_high_grade_list, alternative='less').statistic)
        elif test_name == 'wilcoxon': 
            pvalue_list.append(stats.wilcoxon(merge_low_grade_list, merge_high_grade_list, alternative='less').pvalue)
            statistic_list.append(stats.wilcoxon(merge_low_grade_list, merge_high_grade_list, alternative='less').statistic)
        else:
            print('Test name not recognized')
    pvalue_list = np.array(pvalue_list)
    statistic_list = np.array(statistic_list)
    
    return pvalue_list, statistic_list, num_low_nuclei, num_high_nuclei, low_grade_area_list, high_grade_area_list

def p_value_low_vs_high(low_grade_area_list, high_grade_area_list, num_test,num_patch_per_group,  test_name):
    
    pvalue_list = [] #list of t-test for each pair of grade
    statistic_list = []
    num_sample = min(len(low_grade_area_list), len(high_grade_area_list))
    for num_test_id in range(num_test):
        idx_sample = random.sample(range(num_sample),num_patch_per_group)
        low_grade_list = low_grade_area_list[idx_sample]
        idx_sample = random.sample(range(num_sample),num_patch_per_group)
        high_grade_list = high_grade_area_list[idx_sample]
        
        if test_name == 'ttest_ind':
            pvalue_list.append(stats.ttest_ind(low_grade_list, high_grade_list).pvalue)
            statistic_list.append(stats.ttest_ind(low_grade_list, high_grade_list).statistic)
        elif test_name == 'mannwhitneyu': 
            pvalue_list.append(stats.mannwhitneyu(low_grade_list, high_grade_list).pvalue)
            statistic_list.append(stats.mannwhitneyu(low_grade_list, high_grade_list).statistic)
        elif test_name == 'wilcoxon': 
            pvalue_list.append(stats.wilcoxon(low_grade_list, high_grade_list).pvalue)
            statistic_list.append(stats.wilcoxon(low_grade_list, high_grade_list).statistic)
        else:
            print('Test name not recognized')
    pvalue_list = np.array(pvalue_list)
    statistic_list = np.array(statistic_list)
    return pvalue_list, statistic_list

def significance_value(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'

def plot_statistic_test(class_list, tumor_type_list, Value_list, num_test, num_patch_per_group, test_name, component_name):
    
    idx = np.where((class_list<2)&(tumor_type_list == 'Real'+str(component_name)))[0]
    low_grade_list = Value_list[idx]
    idx = np.where((class_list>1)&(tumor_type_list == 'Real'+str(component_name)))[0]
    high_grade_list = Value_list[idx]
    real_pvalue_list, real_statistic_list = p_value_low_vs_high(low_grade_list, high_grade_list,num_test,num_patch_per_group, test_name)

    idx = np.where((class_list<2)&(tumor_type_list == 'Syn'+str(component_name)))[0]
    low_grade_list = Value_list[idx]
    idx = np.where((class_list>1)&(tumor_type_list == 'Syn'+str(component_name)))[0]
    high_grade_list = Value_list[idx]

    my_col = ['#3274A1','#C00000']
    syn_pvalue_list, syn_statistic_list =  p_value_low_vs_high(low_grade_list, high_grade_list,num_test,num_patch_per_group, test_name)
    log_pvalue_list = -np.log10(np.concatenate((real_pvalue_list, syn_pvalue_list), axis = None))
    df = pd.DataFrame({'P-value': log_pvalue_list, 'Data Type': ['Real']*num_test+['Syn']*num_test, 'T-statistic': np.concatenate((real_statistic_list, syn_statistic_list), axis = None)})
    plt.figure(figsize = (2,4))
    ax = sns.boxplot(x = 'Data Type', y = 'P-value', data = df, width=0.3, palette= my_col)
    plt.xlabel('Data')
    plt.ylabel('-log(P-value)', fontsize = 16)
    p_value_ontop = stats.mannwhitneyu(-np.log10(real_pvalue_list), -np.log10(syn_pvalue_list), alternative='two-sided').pvalue
    significance = significance_value(p_value_ontop)
    x1, x2 = 0, 1
    y, h, col = df['P-value'].max()-0.1, 0.1, 'k'  # y: the height at which to draw the line; h: line height; col: line color
    plt.plot([x1, x2], [y-h, y-h], lw=1.5, c=col)
    plt.text((x1+x2)*.5,y, f"{significance}", ha='center', va='bottom', color=col, fontsize=16)
    plt.show()
    
import numpy as np


def label_nuclei_by_majority_pixel(
    nuclei_mask,
    pixel_class_mask,
    background_id=0,
    ignore_pixel_labels=None,
):
    """
    Assign one class label to each nucleus based on the majority pixel class
    inside that nucleus.

    Parameters
    ----------
    nuclei_mask : np.ndarray
        2D instance segmentation mask, shape (H, W).
        Background should be 0. Each nucleus has a unique integer ID:
        1, 2, 3, ...

    pixel_class_mask : np.ndarray
        2D pixel classification mask, shape (H, W).
        Each pixel contains a class label, for example:
        0 = background, 1 = tumor, 2 = non-tumor, etc.

    background_id : int
        Label used for background in nuclei_mask.

    ignore_pixel_labels : list, tuple, set, or None
        Pixel classes to ignore when voting.
        For example, use [0] if you do not want background pixels
        to affect the nucleus label.

    Returns
    -------
    nuclei_class_mask : np.ndarray
        2D mask, shape (H, W).
        Each nucleus region is filled with its assigned class label.

    nucleus_label_dict : dict
        Dictionary mapping:
        nucleus_instance_id -> assigned_class_label
    """

    nuclei_mask = np.asarray(nuclei_mask)
    pixel_class_mask = np.asarray(pixel_class_mask)

    if nuclei_mask.shape != pixel_class_mask.shape:
        raise ValueError(
            f"Shape mismatch: nuclei_mask has shape {nuclei_mask.shape}, "
            f"but pixel_class_mask has shape {pixel_class_mask.shape}"
        )

    if ignore_pixel_labels is None:
        ignore_pixel_labels = set()
    else:
        ignore_pixel_labels = set(ignore_pixel_labels)

    nuclei_class_mask = np.zeros_like(nuclei_mask, dtype=pixel_class_mask.dtype)
    nucleus_label_dict = {}

    nucleus_ids = np.unique(nuclei_mask)
    nucleus_ids = nucleus_ids[nucleus_ids != background_id]

    for nucleus_id in nucleus_ids:
        nucleus_region = nuclei_mask == nucleus_id

        pixel_labels = pixel_class_mask[nucleus_region]

        if ignore_pixel_labels:
            keep = ~np.isin(pixel_labels, list(ignore_pixel_labels))
            pixel_labels = pixel_labels[keep]

        if pixel_labels.size == 0:
            assigned_label = background_id
        else:
            labels, counts = np.unique(pixel_labels, return_counts=True)
            assigned_label = labels[np.argmax(counts)]

        nuclei_class_mask[nucleus_region] = assigned_label
        nucleus_label_dict[int(nucleus_id)] = int(assigned_label)

    return nuclei_class_mask, nucleus_label_dict