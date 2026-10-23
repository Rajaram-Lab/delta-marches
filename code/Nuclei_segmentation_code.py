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
#%% This code is to segment nuclei from H&E images using the pretrained model
#%%
from configparser import ConfigParser
import os

import numpy as np
from tensorflow.python.keras.models import load_model
import tensorflow as tf
#from tensorflow.python.keras.utils import multi_gpu_model
tf.compat.v1.disable_eager_execution() 
from scipy.sparse import coo_matrix
import cv2
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from Utils.Normalization import StainNormalizer   
import pickle
#%% Load YAML file
yaml_paths = './data/data_dir.yaml'
with open(yaml_paths, 'r') as yaml_file:
    file_paths = yaml.safe_load(yaml_file)
    
# %% Files
pax8_model_file = file_paths['WEIGHTS']['pax8_model_file']
pax8_target = file_paths['WEIGHTS']['pax8_target']
# pax8_model_file = './checkpoints/PAX8_Nuclei_Segmentation/PAX8_Resnet_1WCCE_9Dice_20X_hovernet_masks-30.h5'
# pax8_target = './checkpoints/PAX8_Nuclei_Segmentation/Updated_PAX8_Normalization_Target.npy'
  
pax8_model = load_model(pax8_model_file,compile=False)
pax8Target = np.load(pax8_target)

# Prediction function, model expects 416x416 or multiples of that?
def PadCropPred(img,model):
    rPad=int(32*np.ceil(img.shape[0]/32)-img.shape[0])
    cPad=int(32*np.ceil(img.shape[1]/32)-img.shape[1])
    imgOut=np.pad(img,[[0,rPad],[0,cPad],[0,0]])
    activations=np.squeeze(model.predict(np.expand_dims(imgOut,axis=0)/255))
    predClasses=np.argmax(activations,axis=-1)
    return predClasses

# Overlap function
def OverlapNucleiAndMask(nuclearMask, overlapMask):
    hneNuclearMask = nuclearMask.copy()
    hneNuclearMask.dtype = np.uint8
    
    stainMask = overlapMask.copy()
    stainMask.dtype = np.uint8
    
    nObj, labels, compStats, centroids = \
        cv2.connectedComponentsWithStats(hneNuclearMask, connectivity=8)
    minArea=15 #30
    mapMat=np.zeros(nObj,dtype=np.int32)
    objCounter=1
    for i in range(1,nObj):
        if compStats[i,-1]>=minArea:
            mapMat[i]=objCounter
            objCounter+=1
        else:
            mapMat[i]=0
    labels=mapMat[labels]
    
    confMat=coo_matrix((np.ones(len(labels[labels!=0])),
            (labels[labels!=0],stainMask[labels!=0])),shape=(nObj,2)).toarray()
    
    labelMap=(confMat[:,1]>10)+1
    labelMap[0]=0
    
    hneLabels=labelMap[labels]
    
    return hneLabels

def nuclei_seg_func(hneImg):
    r = hneImg[:,:,0] > 220
    g = hneImg[:,:,1] > 220
    b = hneImg[:,:,2] > 220
    backgroundMask = np.logical_and(np.logical_and(r,g),b)
    tissueMask = np.invert(backgroundMask)

    tissuePixels = np.where(tissueMask)
    imagePixels = hneImg[tissuePixels]
    sourcePixels = np.reshape(imagePixels, (imagePixels.shape[0], 1, 3))

    pax8Normalizer=StainNormalizer('macenko')
    pax8Normalizer.fit(targetSlide=pax8Target,targetMask=None,sourceSlide=sourcePixels,sourceMask=None)
    normalizedHnE = pax8Normalizer.transform(hneImg)
    # End normalization code

    modelPred = PadCropPred(normalizedHnE, pax8_model)

    predictedSegmentation = np.logical_or(modelPred == 1, modelPred == 2)
    predictedPAX8Nuclei = OverlapNucleiAndMask(predictedSegmentation, modelPred == 2)
    return predictedPAX8Nuclei
#%% import data

import pickle

import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "save_dir",
    type=str,
    help="Path to the saved pickle file"
)

args = parser.parse_args()
save_dir = args.save_dir

with open(save_dir, 'rb') as f:
    traverse_data = pickle.load(f)
    
# data_dir = file_paths['synthetic_data']['synthetic_data_grade1to4']
# with open(data_dir,'rb') as f:
#     traverse_data = pickle.load(f)
#%% for different grade transition
Nuc_Seg = []
allimg = traverse_data['img'][traverse_data['full_grade_list'][:,:,0],traverse_data['full_grade_list'][:,:,1],:,:,:]
Nuc_Seg = []
for i in range(allimg.shape[0]):
   nug_seg_i = []
   for j in range(allimg.shape[1]):
       hneImg = allimg[i][j].astype(np.uint8)
       predictedPAX8Nuclei = nuclei_seg_func(hneImg)
       nug_seg_i.append(predictedPAX8Nuclei)
   Nuc_Seg.append(nug_seg_i)
Nuc_Seg = np.array(Nuc_Seg)
traverse_data['full_grade_list_Nuc_seg'] = Nuc_Seg

#%%for grade same grade transition
#allimg = data['patches'][traverse_data['idx']]
allimg = traverse_data['patient_across_img']
# allimg = traverse_data['img'][traverse_data['low_list_unique'][:,0],traverse_data['low_list_unique'][:,1],:,:,:]

Nuc_Seg = []
for i in range(allimg.shape[0]):
   hneImg = allimg[i].astype(np.uint8)
   predictedPAX8Nuclei = nuclei_seg_func(hneImg)
   Nuc_Seg.append(predictedPAX8Nuclei)
Nuc_Seg = np.array(Nuc_Seg)
traverse_data['patient_across_img_Nuc_seg'] = Nuc_Seg

save_dir = args.save_dir
with open(save_dir,'wb') as f:
    pickle.dump(traverse_data,f, protocol= 4)
# data_dir = file_paths['synthetic_data']['synthetic_data_grade1to4']
# with open(data_dir,'wb') as f:
#     pickle.dump(traverse_data,f, protocol= 4)