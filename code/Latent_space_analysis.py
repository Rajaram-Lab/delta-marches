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
#%% This code is to analyze the latent space of AAE and DIFFAE model
#1. Load the latent space of AAE and DIFFAE model
#2. Compute the accuracy of classifier in latent space
#3. Compute the percentage of same patient and same grade neighbor in latent space
#4. Compute the reconstruction error of AAE and DIFFAE model
#5. Plot the latent space analysis: classifier accuracy, same patient neighbor, same grade neighbor
#6. Plot the reconstruction error of AAE and DIFFAE model
#7. Plot the bar plot of AAE and DIFFAE latent metric
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

#%% load encoder for AAE model
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

    


import pickle
# import yaml
# yaml_paths = './data/data_dir.yaml'
# with open(yaml_paths, 'r') as yaml_file:
#     file_paths = yaml.safe_load(yaml_file)
    
    
# save_dir = file_paths['real_data']['real_data_120000patches']
# with open(save_dir,'rb') as f:
#   data = pickle.load(f)
import argparse
import pickle

#real data
import yaml
yaml_paths = './data/data_dir.yaml'
with open(yaml_paths, 'r') as yaml_file:
    file_paths = yaml.safe_load(yaml_file)
    
parser = argparse.ArgumentParser()
parser.add_argument(
    "--full_data",
    type=str,
    default= './data/datasets/grade_data_120000.pkl',
    help="Path to the saved pickle file"
)
args = parser.parse_args()
save_dir = args.full_data

with open(save_dir, "rb") as f:
    data = pickle.load(f)
idx_list = []
patch_list = []
class_list = []

test_data = 'all_patients'
 
if test_data == 'all_patients':
    patient_list = np.unique(data['patients'])
elif test_data == 'test_patients':
    patient_list = np.unique(data['patients'][:24000])
    
# UMAP plot for latent space  of test_patients with nodes of 20 patches per patient
# only take up to 20 patches per patients
# num_patch_per_pat = 20


for i in patient_list:
    idx = np.where(data['patients']==i)[0]
    # np.random.shuffle(idx)
    # idx = idx[:num_patch_per_pat]
    idx_list.append(idx)
    patch_list.append(data['patches'][idx])
    class_list.append(data['classes'][idx])


idx_list = np.concatenate(idx_list, axis = 0)
patch_list = np.concatenate(patch_list, axis = 0)
class_list = np.concatenate(class_list, axis = 0)


enc_name = 'AAE'
transform = transforms.ToTensor()
if enc_name == 'DIFFAE':
    # load latent list from DIFFAE
    DIFFAE_latent_list = np.load(file_paths['latent_space']['DIFFAE'])
    latent_list = DIFFAE_latent_list[idx_list]
elif enc_name == 'AAE':
    #load latent list from AAE
    latent_size = [128, 256, 512, 1024, 2048]
    for nz in latent_size:
        print(nz)
        enc_Path = file_paths['WEIGHTS']['AAE']['Encoder'][f'n{nz}']
        # enc_Path = f'./checkpoints/AAE_model_n{nz}_enc_NucSeg.pth'
        enc = Encoder(nc=3, nf=256, nz=nz)
        enc.load_state_dict(torch.load(enc_Path))
        enc = enc.to(device)
        enc.eval()
        batch_size = 100
        latent_list = []
        classes_list = []
        patients_list = []
        with torch.no_grad():
            for i in range(math.ceil(patch_list.shape[0]/batch_size)):
                x = []
                for j in range(batch_size):
                    if i*batch_size+j < patch_list.shape[0]:
                        img = patch_list[i*batch_size+j]
                        img = transform(img)
                        x.append(img)
                x = torch.stack(x)
                if cuda:
                    x = x.cuda()
                    
                vec  = enc(x)
                latent_list.append(vec.cpu().numpy())
                classes_list.append(class_list[i*batch_size:(i+1)*batch_size])
                patients_list.append(data['patients'][i*batch_size:(i+1)*batch_size])
        latent_list = np.concatenate(latent_list, axis = 0)
        classes_list = np.concatenate(classes_list, axis = 0)
        patients_list = np.concatenate(patients_list, axis = 0)
        lat_data = {'lat_vector': latent_list, 'classes': classes_list, 'patients': patients_list, 'idx': np.arange(patch_list.shape[0])}
        save_dir = f'./analysis/{test_data}_AAE_lat_vector_n{nz}.pkl'
        with open(save_dir,'wb') as f:
            pickle.dump(lat_data, f, protocol=4)
        

#%% Accuracy of classifier in latent space

  
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model_list = ['AAE', 'AAE', 'AAE', 'AAE', 'AAE', 'DIFFAE']
latent_size = [128, 256, 512, 1024, 2048, 512]
fold = 0
accuracy_list = []
for i in range(len(model_list)):
    model = model_list[i]
    latent_dim = latent_size[i]
    if model == 'DIFFAE':
        DIFFAE_latent_list = np.load(file_paths['latent_space']['DIFFAE'])
        # DIFFAE_latent_list = np.load('./DIFFAE_grade/imgs_manipulated/cond_vec_list_18ktrainingdata.npy')
        fold_idx = np.arange(0,len(DIFFAE_latent_list))
        fold_plit = np.array_split(fold_idx, 5)
        accuracy_list_fold = []
        for fold in range(5):
            train_idx = np.setdiff1d(fold_idx, fold_plit[fold])
            test_idx = fold_plit[fold]
            train_data = {'lat_vector':DIFFAE_latent_list[train_idx], 'classes':data['classes'][train_idx]}
            test_data = {'lat_vector':DIFFAE_latent_list[test_idx], 'classes':data['classes'][test_idx]}  


            clf = LinearDiscriminantAnalysis()
            clf.fit(train_data['lat_vector'], train_data['classes'])
            y_pred = clf.predict(test_data['lat_vector'])
            accuracy_list_fold.append(np.sum(y_pred == test_data['classes'])/len(test_data['classes']))
    elif model == 'AAE':
        save_dir = f'./analysis/all_patients_AAE_lat_vector_n{latent_dim}.pkl'
        with open(save_dir,'rb') as f:
            lat_data = pickle.load(f)
            
        #split len(data) into 5 fold
        fold_idx = np.arange(0,len(lat_data['lat_vector']))
        fold_plit = np.array_split(fold_idx, 5)
        accuracy_list_fold = []
        for fold in range(5):
            train_idx = np.setdiff1d(fold_idx, fold_plit[fold])
            test_idx = fold_plit[fold]
            train_data = {'lat_vector':lat_data['lat_vector'][train_idx], 'classes':lat_data['classes'][train_idx]}
            test_data = {'lat_vector':lat_data['lat_vector'][test_idx], 'classes':lat_data['classes'][test_idx]}  


            clf = LinearDiscriminantAnalysis()
            clf.fit(train_data['lat_vector'], train_data['classes'])
            y_pred = clf.predict(test_data['lat_vector'])
                
            accuracy_list_fold.append(np.sum(y_pred == test_data['classes'])/len(test_data['classes']))
    accuracy_list.append(np.mean(accuracy_list_fold))
accuracy_list = np.array(accuracy_list)

#%% find nearest neighbor for each point and propability of having same class or from same patient

from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
def same_origin_nei(test_data):
    nei_list = []
    
    nei_info = NearestNeighbors(n_neighbors=10, metric='cosine').fit(test_data['lat_vector'])
    
    nei_list = nei_info.kneighbors(test_data['lat_vector'], return_distance=False)
    grade_nei_list = test_data['classes'][nei_list]
    patient_nei_list = test_data['patients'][nei_list]


    same_class_nei = grade_nei_list == test_data['classes'][:,None]
    prop_same_class_nei =np.mean(np.sum(same_class_nei, axis = 1)/same_class_nei.shape[1], axis = 0)

    same_patients_nei = patient_nei_list == test_data['patients'][:,None]
    prop_same_patients_nei = np.mean(np.sum(same_patients_nei, axis = 1)/same_patients_nei.shape[1], axis = 0)
    return prop_same_class_nei, prop_same_patients_nei


#%% plot the latent space analysis: classifier accuracy, same patient neighbor, same grade neighbor
model_list = ['AAE', 'AAE', 'AAE', 'AAE', 'AAE', 'DIFFAE']
latent_size = [128, 256, 512, 1024, 2048, 512]
fold = 0
prop_same_class_nei_list = []
prop_same_patients_nei_list = []
for i in range(len(model_list)):
    model = model_list[i]
    latent_dim = latent_size[i]
    print(i)
    if model == 'DIFFAE':
        DIFFAE_latent_list = np.load('./DIFFAE_grade/imgs_manipulated/cond_vec_list.npy')
        test_data = {'lat_vector':DIFFAE_latent_list, 'classes':data['classes'], 'patients': data['patients']}  
    elif model == 'AAE':
        # save_dir = file_paths['latent_space']['AAE']['all_data']['n'+str(latent_dim)]
        save_dir = f'./analysis/all_patients_AAE_lat_vector_n{latent_dim}.pkl'
        with open(save_dir,'rb') as f:
            lat_data = pickle.load(f)
        test_data = {'lat_vector':lat_data['lat_vector'], 'classes':lat_data['classes'], 'patients': lat_data['patients']}
    prop_same_class_nei, prop_same_patients_nei = same_origin_nei(test_data)
    prop_same_class_nei_list.append(prop_same_class_nei)
    prop_same_patients_nei_list.append(prop_same_patients_nei)

latent_analysis_data = {'prop_same_class_nei': prop_same_class_nei_list, 'prop_same_patients_nei': prop_same_patients_nei_list, 'accuracy': accuracy_list}
#save as csv
df = pd.DataFrame(latent_analysis_data)
df.to_csv('./latent_analysis_data.csv', index = False)
list_model_name = ['128', '256', '512', '1024', '2048']
plt.figure(figsize=(5,3))
plt.plot(list_model_name,prop_same_patients_nei_list[:5],marker='o', markersize=10, label='Same patient neighbor')
plt.plot(list_model_name,prop_same_class_nei_list[:5],marker='v',markersize=10,  label='Same grade neighbor')
plt.plot(list_model_name,accuracy_list[:5],marker='x',markersize=10,  label='Classifier Accuracy')
plt.xlabel('Latent space size', fontsize =16)
plt.xticks(fontsize =16)
plt.ylabel('Percentage', fontsize =16)
plt.yticks(fontsize =16)
plt.legend(fontsize =16, bbox_to_anchor=(0.0, -0.2, 0.8, 0.0))
plt.savefig('./image/AAE_latent_analysis.png', bbox_inches='tight')
#%% Plot bar plot of AAE and DIFFAE latent metric
 
import numpy as np 
import matplotlib.pyplot as plt 
 
# set width of bar 
barWidth = 0.2
 
# set height of bar for now
AAE = [prop_same_patients_nei_list[3], prop_same_class_nei_list[3], accuracy_list[3]] 
DIFFAE = [prop_same_patients_nei_list[5], prop_same_class_nei_list[5], accuracy_list[5]] 
 
# Set position of bar on X axis 
br1 = [0.5, 1.5, 2.5]
br2 = [x + barWidth for x in br1] 
 
# Make the plot
plt.figure(figsize=(5, 3))
plt.bar(br1, AAE, color =np.array([255,101,101])/255, width = barWidth, 
        edgecolor ='grey', label ='AAE - 1024') 
plt.bar(br2, DIFFAE, color =np.array([146,208,80])/255, width = barWidth, 
        edgecolor ='grey', label ='DIFFAE') 
 
# Adding Xticks 
plt.ylabel('Percentage', fontsize = 14) 
plt.xticks([0.6, 1.6, 2.6], 
        ['Same patient\n neighbor', 'Same grade \n neighbor', 'Classifier \n Accuracy'], fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(fontsize = 14, bbox_to_anchor=(0.0, -0.2, 0.4, 0.0))

plt.savefig('./image/AAE_DIFFAE_latent_metric_v3.png', bbox_inches='tight', dpi =300)
plt.show()     


#%% Compute Reconstruction Accuracy
#plot column for mean of error
from skimage.metrics import structural_similarity as ssim 



save_dir = f'./analysis/test_patients_AAE_lat_vector_n1024.pkl'
with open(save_dir,'rb') as f:
    AAE_reconstruction_data = pickle.load(f)
reconstruction_data = {'org_image': AAE_reconstruction_data['org_image']}
reconstruction_data['rec_AAE_image'] = AAE_reconstruction_data['rec_image']

save_dir = f'./DIFFAE_grade/imgs_manipulated/image_reconstruction.pkl'
with open(save_dir,'rb') as f:
    DIFFAE_reconstruction_data = pickle.load(f)
reconstruction_data['rec_DIFFAE_image'] = DIFFAE_reconstruction_data['rec_image']

# save_dir = file_paths['reconstruction_comparison']
# save_dir = f'./analysis/reconstruction_comparison.pkl'
# with open(save_dir,'rb') as f:
#     reconstruction_data = pickle.load(f)
from math import log10, sqrt 
import cv2 
import numpy as np 
  
def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 
DIFFAE_ssim_skimg_error = []
DIFFAE_PSNR_error = []
DIFFAE_MSE_eror = []
num_sample = len(reconstruction_data['org_image'])

for i in range(num_sample):
    org_img = reconstruction_data['org_image'][i]
    rec_img = reconstruction_data['rec_DIFFAE_image'][i]
    if org_img.max() < 2:
        org_img = (org_img * 255).astype(np.uint8)
    if rec_img.max() < 2:
        rec_img = (rec_img * 255).astype(np.uint8)
    ssim_skimg = ssim(org_img, rec_img, channel_axis=-1, data_range=255)
    DIFFAE_ssim_skimg_error.append(ssim_skimg)
    DIFFAE_PSNR_error.append(PSNR(org_img, rec_img))
    DIFFAE_MSE_eror.append(np.mean((org_img-rec_img)**2))
DIFFAE_ssim_skimg_error = np.array(DIFFAE_ssim_skimg_error)
DIFFAE_PSNR_error = np.array(DIFFAE_PSNR_error)
DIFFAE_MSE_eror = np.array(DIFFAE_MSE_eror)

AAE_ssim_skimg_error = []
AAE_PSNR_error = []
AAE_MSE_error = []
for i in range(num_sample):
    org_img = reconstruction_data['org_image'][i]
    rec_img = reconstruction_data['rec_AAE_image'][i]
    if rec_img.max() < 2:
        rec_img = (rec_img * 255).astype(np.uint8)
    if org_img.max() < 2:
        org_img = (org_img * 255).astype(np.uint8)
    ssim_skimg = ssim(org_img, rec_img, channel_axis=-1, data_range=255)
    AAE_ssim_skimg_error.append(ssim_skimg)
    AAE_PSNR_error.append(PSNR(org_img, rec_img))
    AAE_MSE_error.append(np.mean((org_img-rec_img)**2))
AAE_ssim_skimg_error = np.array(AAE_ssim_skimg_error)
AAE_PSNR_error = np.array(AAE_PSNR_error)
AAE_MSE_error = np.array(AAE_MSE_error)
#plot column for mean of error
DIFFAE_MSE_eror_mean = np.mean(DIFFAE_MSE_eror)
AAE_MSE_error_mean = np.mean(AAE_MSE_error)
DIFFAE_PSNR_error_mean = np.mean(DIFFAE_PSNR_error)
AAE_PSNR_error_mean = np.mean(AAE_PSNR_error)
DIFFAE_ssim_skimg_error_mean = np.mean(DIFFAE_ssim_skimg_error)
AAE_ssim_skimg_error_mean = np.mean(AAE_ssim_skimg_error)

fig, ax = plt.subplots(1,3, figsize=(8,3))
title = ['255-MSE', 'PSNR', 'SSIM']
#change location of barplot to 1 2
positions = (0.1, 0.4)
labels = ['AAE', 'DIFFAE']
colors = [np.array([255,101,101])/255, np.array([146,208,80])/255]
ax[0].bar(positions, [255-AAE_MSE_error_mean, 255-DIFFAE_MSE_eror_mean], color=colors, width=0.1)
ax[0].set_title(title[0], fontsize=16)
ax[0].set_xticks(positions, labels, fontsize=16)
ax[0].set_xlim(0, 0.5)
ax[1].bar(positions, [AAE_PSNR_error_mean, DIFFAE_PSNR_error_mean], color=colors, width=0.1)
ax[1].set_title(title[1], fontsize=16)
ax[1].set_xticks(positions, labels, fontsize=16)
ax[1].set_xlim(0, 0.5)
ax[2].bar(positions, [AAE_ssim_skimg_error_mean, DIFFAE_ssim_skimg_error_mean], color=colors, width=0.1)
ax[2].set_title(title[2], fontsize=16)
ax[2].set_xticks(positions, labels, fontsize=16)
ax[2].set_xlim(0, 0.5)
plt.savefig('./image/compare_reconstruction_error_sigmoidcorrected.png', dpi=300, bbox_inches='tight')


#plot with FID score
import numpy as np, torch
from torchmetrics.image.fid import FrechetInceptionDistance

# numpy (N,224,224,3) -> torch (N,3,H,W) in [0,1]
def to_torch(img_np):
    x = torch.from_numpy(img_np).permute(0,3,1,2).float()
    if x.max() > 1.5: x = x/255.0
    return x.clamp(0,1)

real_images = reconstruction_data['org_image']
fake_images = reconstruction_data['rec_AAE_image']
real_t = to_torch(real_images)
fake_t = to_torch(fake_images)

fid = FrechetInceptionDistance(normalize=True)
fid.update(real_t, real=True)
fid.update(fake_t, real=False)
AAE_FID = fid.compute().item()


real_images = reconstruction_data['org_image']
fake_images = reconstruction_data['rec_DIFFAE_image']
real_t = to_torch(real_images)
fake_t = to_torch(fake_images)
fid = FrechetInceptionDistance(normalize=True)
fid.update(real_t, real=True)
fid.update(fake_t, real=False)
DIFFAE_FID = fid.compute().item()


title = ['MSE', 'SSIM', 'FID']
#change location of barplot to 1 2
positions = (0.1, 0.3)
labels = ['AAE', 'DIFFAE']
colors = [np.array([255,101,101])/255, np.array([146,208,80])/255]
plt.figure(figsize=(6,3))
plt.subplot(1,3,1)
plt.bar(positions, [AAE_MSE_error_mean, DIFFAE_MSE_eror_mean], color=colors, width=0.1)
plt.xlim(0, 0.4)
plt.yticks(fontsize=14)
plt.xticks(positions, labels, fontsize=14)
plt.subplot(1,3,2)
plt.bar(positions, [AAE_ssim_skimg_error_mean, DIFFAE_ssim_skimg_error_mean], color=colors, width=0.1)

plt.xticks(positions, labels, fontsize=14)
plt.xlim(0, 0.4)
plt.yticks(fontsize=14)
plt.subplot(1,3,3)
plt.bar(positions, [AAE_FID, DIFFAE_FID], color=colors, width=0.1)
plt.xlim(0, 0.4)
plt.xticks(positions, labels, fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('./image/compare_reconstruction_error_MSE_SSIM_FID.png', dpi=300, bbox_inches='tight')
