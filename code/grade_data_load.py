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
#%% This code is to load patches from original dataset 
# This includes:
# Big data with 120000 patches from 4 grades
# Real data with 1000 patches for each grade
# Save as pkl file for training AAE and as figures for training DIFFAE
#%%
import torch
import pandas as pd
import os,sys
from PIL import Image

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))                                                                                                                
import Utils.Patch_Generation as pg 
import numpy as np
device = torch.device('cuda:0')
import time
from torch import nn, cuda, device
from torch.autograd import Variable
import torchvision
from torch.utils.data import Dataset# data management
import torch.optim as optim # Optimization algorithm
import matplotlib.pyplot as plt
import torchvision as tv
import itertools
import pickle
from torchvision.utils import save_image
print(cuda.is_available())
device = device('cuda:0')
print(device)
import yaml
# Load YAML file
yaml_paths = './data/data_dir.yaml'
with open(yaml_paths, 'r') as yaml_file:
    file_paths = yaml.safe_load(yaml_file)
import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    type=str,
    required=True,
    choices=["Mayo", "UTSW"]
)

parser.add_argument(
    "--data_path",
    type=str,
    required=True,
    default= './Driver_Mutations_From_HnE/QuPath_Projects/NucSegRegionGrade_Classification/patches/v1/224/',
    help="Path to the original dataset"
)

args = parser.parse_args()

print("Dataset:", args.data)
print("Original data path:", args.data_path)

if args.data == "UTSW":
        
##%% Nuclei segmentation dataset

    org_path = args.data_path
    # org_path = file_paths['original_dir']['grade']

    import random 
    def partition (list_in, n):
        random.shuffle(list_in)
        return [list_in[i::n] for i in range(n)]

    Data_list = os.listdir(org_path)
    sum_pat = 0        
    grade_dict = {'NucSegGrade_1':0, 'NucSegGrade_2':1, 'NucSegGrade_3':2, 'NucSegGrade_4':3}    
    slide_list =[[],[],[],[]] 
    grade_num =[[],[],[],[]]   
            
    for name in Data_list:
        path = org_path + name

        data = pg.LoadHdf5Data(path)
        grd = grade_dict[np.unique(data['classLabels'])[0]]
        idx = np.where(data['classLabels'] == np.unique(data['classLabels'])[0])[0]
        slide_list[grd].append(name)
        grade_num[grd].append(len(idx))


    # split patients into 10 parts in each grade in order to get 5 fold cross validation   
    slide_list_new = slide_list.copy()
    for i in range(4): 
        idx = np.argsort(grade_num[i])
        slide_list_sorted = [slide_list[i][j] for j in idx]
        slide_list_new[i] = partition(slide_list_sorted,5)
    # load data from each fold
    pat_name = []
    patches = []
    classes = []

    for i in range(5):    
        for j in range(4):
            pat_name_s = []
            patches_s = []
            classes_s = []
            for name in slide_list_new[j][i]:
                path = org_path + name        
                data = pg.LoadHdf5Data(path)
                if len(np.unique(data['classLabels']))>0:
                    num_patch_s =data['numberOfPatches']
                    patches_s.append(data['patchData'][0]['patches'])
                    pat_name_s.append(np.full(num_patch_s,name.replace('.hdf5','')))
                    classes_s.append(data['classLabels'])
            pat_name_s = np.concatenate(pat_name_s, axis =0)                
            classes_s = np.concatenate(classes_s, axis =0)           
            grade_dict = {'NucSegGrade_1':0, 'NucSegGrade_2':1, 'NucSegGrade_3':2, 'NucSegGrade_4':3}
            classes_s = np.array([grade_dict[i] for i in classes_s])
            patches_s = np.concatenate(patches_s, axis =0)  
            index_grade = list(np.where(classes_s==j)[0])
            idx = random.sample(index_grade, k=6000)
            pat_name.append(pat_name_s[idx])
            classes.append(classes_s[idx])
            patches.append(patches_s[idx])
            

    pat_name = np.concatenate(pat_name, axis =0)                
    classes = np.concatenate(classes, axis =0)                
    patches = np.concatenate(patches, axis =0)  

    #save pkl file
    data = {'patches': patches, 'classes': classes, 'patients': pat_name}
    save_dir = './DIFFAE_grade/datasets/grade_data_120000.pkl'
    with open(save_dir,'wb') as f:
        pickle.dump(data,f,protocol = 4)

    #%% create a subset of real data with 1000 patches for each grade for further analysis
    save_dir = './DIFFAE_grade/datasets/grade_data_120000.pkl'
    with open(save_dir,'rb') as f:
        data = pickle.load(f)

    real_data = {'patches': [], 'classes': [], 'patients': [], 'org_idx': []}
    for i in range(4):
        idx = np.where(data['classes']==i)[0]
        idx = np.random.choice(idx, 1000, replace=False)
        real_data['patches'].append(data['patches'][idx])
        real_data['classes'].append(data['classes'][idx])
        real_data['patients'].append(data['patients'][idx])
        real_data['org_idx'].append(idx)
    real_data['patches'] = np.concatenate(real_data['patches'], axis =0)
    real_data['classes'] = np.concatenate(real_data['classes'], axis =0)
    real_data['patients'] = np.concatenate(real_data['patients'], axis =0)
    real_data['org_idx'] = np.concatenate(real_data['org_idx'], axis =0)

    save_dir = './DIFFAE_grade/datasets/grade_data_4000.pkl'
    with open(save_dir,'wb') as f:
        pickle.dump(real_data,f,protocol = 4)
   
#%% save image to testing list in DIFFAE
    num_patch_per_grade = 2000
    save_dir = './DIFFAE_grade/datasets/grade/'
    save_dir_18k = './DIFFAE_grade/datasets/grade18k/'
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)
    if os.path.exists(save_dir_18k) == False:
        os.mkdir(save_dir_18k)

    num_patch = 0
    for i in range(4):
        idx = np.where(data['classes']==i)[0]
        np.random.shuffle(idx)
        for j in range(num_patch_per_grade):
            save_name = save_dir +f'i{num_patch}.png'
            im = Image.fromarray(data['patches'][idx[j]])
            num_patch +=1
            im.save(save_name)   
            
            save_name = save_dir_18k +f'i{num_patch}.png'
            im.save(save_name)

    #save csv file for testing classifier in DIFFAE       
    save_dir = './DIFFAE_grade/datasets/class_grade.csv'
    import pandas as pd
    index_name = ['i{}.png'.format(x) for x in range(int(num_patch_per_grade*4))]
    data = -np.ones((4*num_patch_per_grade,4))
    for i in range(4):
        for j in range(num_patch_per_grade):
            data[num_patch_per_grade*i+j,i]=1
    mat = {'grade_1':data[:,0], 'grade_2':data[:,1], 'grade_3':data[:,2], 'grade_4':data[:,3]}
    df = pd.DataFrame(mat)
    df.index = index_name
    df.to_csv(save_dir, sep = ' ',index = index_name , header=True)

elif args.data == 'Mayo':
    #%% Mayo data 
    # save_dir = file_paths['original_dir']['Mayo']
    save_dir = args.data_path
    with open(save_dir, 'rb') as f:
        data_full = pickle.load(f)
    idx = np.where(np.array(data_full['NumofPatch']).astype(int) == 20)[0]
    #find idx within idx that data_full['slide_name'] is not in  data['patients']
    idx = np.array([i for i in idx if data_full['slide_name'][i].replace('.npy','') not in data['patients']])
    idx = np.random.choice(idx, 1000, replace = False)
    save_dir = './DIFFAE_grade/datasets/grade18k/'
    num_patch = 8000
    slide_name_list = []
    patch_idx_list = []
    slide_list = list(data_full['Extracted_20PatchperSlide_list'].keys())
    for slide_idx in idx:
        slide_name = slide_list[slide_idx]
        slide_name_list.append(slide_name)
        patch_idx = np.random.choice(20, 10, replace = False)
        patch_idx_list.append(patch_idx)
        patches = data_full['Extracted_20PatchperSlide_list'][slide_name]['patch_list'][patch_idx]
        for j in range(10):
            save_name = save_dir +f'i{num_patch}.png'
            im = Image.fromarray(patches[j][:,:,:3])
            num_patch +=1
            im.save(save_name) 
    # data_info = {'slide_name': slide_name_list, 'patch_idx': patch_idx_list}
    # with open(save_dir + 'Mayo_data_info.pkl', 'wb') as f:
    #     pickle.dump(data_info, f)

        