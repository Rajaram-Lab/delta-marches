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
#%% This code is to produce the vascular mask from the H&E images using the pretrained model

import torch
import pandas as pd
import os,sys


from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image

from configparser import ConfigParser
import numpy as np
device = torch.device('cuda:0')
#%%
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
#import itertools
from torchvision.utils import save_image
print(cuda.is_available())
device = device('cuda:0')
import glob                                                                                                                                      
import os                                                                                                                                        
import pandas as pd                                                                                                                              
import numpy as np                                                                                                                               
import yaml                                                                                                                                      
                                                                                                                                                 
from sklearn.model_selection import KFold                                                                                                                                                                                                                                                   
os.environ["CUDA_VISIBLE_DEVICES"]="0"                                                                                                           
import sys     

import Utils.MixedModel as mm                                                                                                                                                                                       
                                                                                                                                                 
import pickle as pkl                                                                                                                             
import matplotlib.pyplot as plt                                                                                                                  
import matplotlib.patches as patches                                                                                                             
#import seaborn as sns                                                                                                                           
import scipy.sparse                                                                                                                              
                                                                                                                                                 
import torch                                                                                                                                     
from torch.utils.data import Dataset, DataLoader,Sampler                                                                                         
from torchvision import transforms, utils,models    
import torch.nn as nn                                                                                                                            
from tqdm import tqdm                                                                                                                            
from PIL import Image, ImageDraw                                                                                                                 
import progressbar                                                                                                                               
import matplotlib.colors as colors                                                                                                               
import matplotlib.gridspec as gridspec                                                                                                           
from skimage.morphology import remove_small_objects                                                                                              
                                                                                                                                                 
import copy                                                                                                                                      
import cv2                 
import yaml
#%%
#Load YAML file
yaml_paths = './data/data_dir.yaml'
with open(yaml_paths, 'r') as yaml_file:
    file_paths = yaml.safe_load(yaml_file)
    
    
# %% Load Model                                                                                                                                  
transform =   torchvision.transforms.ToTensor() 
cleanSize=40                                                                                                                                    
predictions=[] 
modelEpoch=7                                                                                                                                     
modelDir=modelDir=file_paths['WEIGHTS']['vasculature']                                                                                                           
# modelDir=modelDir='./checkpoints/AngioPaper/Models/MM_PyT_Dice_MultiBatch/'                                                                                                           
modelFile=os.path.join(modelDir,'MixedSimple_UNetResnet_E'+str(modelEpoch)+'.pt')                                                                
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                                                                            
model=mm.MixedModelSimple(mm.ResNetUNet(2))                                                                                                      
model.load_state_dict(torch.load(modelFile))                                                                                                     
model=model.to(device)                                                                                                                           
model.eval()  


def vas_function(img):
    imgResized = cv2.copyMakeBorder(img,96,96,96,96, cv2.BORDER_REFLECT)
    img = transform(imgResized)
    img=img.float().to(device)                                                                                                                           
    predClasses=torch.argmax(model(img.unsqueeze(0))[0],dim=1)                                                                                   
    predClasses=np.squeeze(predClasses.detach().cpu().numpy())                                                                                   
    toClean=predClasses>0                                                                                                                        
    cleanpredClasses=remove_small_objects(toClean, min_size=cleanSize,connectivity=2)                                                    
    cleanpredClasses=cleanpredClasses[96:320,96:320]
    return cleanpredClasses
#%%  

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

#%%
imgList = traverse_data['img'][traverse_data['full_grade_list'][:,:,0],traverse_data['full_grade_list'][:,:,1],:,:,:]                                          
cleanSize=40                                                                                                                                    
predictions=[] 
vas_list = []
for j in range(len(imgList)):
    predictions=[] 
    class_list =[]
    for i in range(len(imgList[j])):                                                                                                              

        img=imgList[j][i].astype(np.uint8)           
        cleanpredClasses = vas_function(img)
        predictions.append(cleanpredClasses)
    predictions = np.array(predictions)
    vas_list.append(predictions)    
traverse_data['full_grade_list_vas'] = np.array(vas_list)
#%% For same class transition

imgList = traverse_data['patient_across_img']
cleanSize = 40
vas_list = []
for j in range(len(imgList)):

    img=imgList[j].astype(np.uint8)                                                 
    cleanpredClasses=vas_function(img)
    vas_list.append(cleanpredClasses)    
vas_list = np.array(vas_list)
traverse_data['patient_across_img_vas'] = np.array(vas_list)


save_dir = args.save_dir
with open(save_dir,'wb') as f:
    pickle.dump(traverse_data,f, protocol= 4)
# save_dir = ''
# with open(save_dir,'wb') as f:
#     pickle.dump(traverse_data,f, protocol= 4)
