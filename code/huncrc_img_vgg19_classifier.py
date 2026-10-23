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
#%% This code is the VGG19 model for classifying grade for synthetic images

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
import yaml
#%% create dataset
# Load YAML file

import pickle
# save_dir = './datasets/HunCRC_data.pkl'
# with open(save_dir,'rb') as f:
#     data = pickle.load(f)
# print(data.keys())

# #used this code to train the model again

# classes_list = ['normal', 'lowgrade_dysplasia', 'highgrade_dysplasia']
# idx = np.where(np.isin(data['class'], classes_list))[0]
# class_dict = {class_name: i for i, class_name in enumerate(classes_list)}
 
# trainPatches = data['img'][idx]
# trainClasses = data['class'][idx]
# trainClasses = np.array([class_dict[class_name] for class_name in trainClasses])
# testPatches = data['img'][idx]
# testClasses = data['class'][idx]
# testClasses = np.array([class_dict[class_name] for class_name in testClasses])
# #%% Define the dataset
# trainPatchClasses = torch.from_numpy(np.uint8(np.array(trainClasses,dtype=float))).type(torch.LongTensor)
# testPatchClasses = torch.from_numpy(np.uint8(np.array(testClasses,dtype=float))).type(torch.LongTensor)

# data_arr = {"train": trainPatches,
#             "val": testPatches}

# target_arr = {"train": trainPatchClasses,
#               "val": testPatchClasses}
# #
# class MyDataset(Dataset):
#     def __init__(self, data, target,  transform=None, preproc=None):
#         self.data = data
#         self.target = target
#         self.transform = transform
#         self.preproc = preproc

#     def __getitem__(self, index):
#         x = self.data[index]
#         y = self.target[index]

#         if self.transform:
#             x = self.transform(images=np.expand_dims(x, 0))

#         if self.preproc:
#             x = self.preproc(Image.fromarray(np.squeeze(x)))
#         return x, y

#     def __len__(self):
#         return len(self.data)

# def reset_weights(m):
#   '''
#     Try resetting model weights to avoid
#     weight leakage.
#   '''
#   for layer in m.children():
#    if hasattr(layer, 'reset_parameters'):
#     print(f'Reset trainable parameters of layer = {layer}')
#     layer.reset_parameters()    
     
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# image_datasets = {"train": MyDataset(data_arr["train"], target_arr["train"],  transform=None, preproc=transform),
#                       "val": MyDataset(data_arr["val"], target_arr["val"], transform=None, preproc=transform)}

# batch_size=32
# train_loader = torch.utils.data.DataLoader(
#         image_datasets['train'], shuffle = True, batch_size=batch_size, num_workers=8, pin_memory = True)
# test_loader = torch.utils.data.DataLoader(
#         image_datasets['val'], shuffle = False, batch_size=batch_size, num_workers=8, pin_memory = True)


# num_classes = len(np.unique(trainClasses))
# num_epochs = 20
# learning_rate = 1e-4

# model = torchvision.models.vgg19(pretrained= True)
# # %% Freeze the convolution layers

# #%% Replace the fc layers

# model.classifier[6] = nn.Linear(in_features = 4096, out_features = num_classes, bias=True)


# for param in model.parameters():
#         param.requires_grad = True
# i=0
# for param in model.parameters():
#     if i<5:
#         param.requires_grad = False
#         i+=1
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  
# model.to(device)
# criterion.to(device)


# classes = trainPatchClasses.numpy()
# from sklearn.utils.class_weight import compute_class_weight# num_classes - int value representing number of classes# trainLabels - np array with all training labelscw = compute_class_weight(class_weight='balanced',
# criterion = nn.CrossEntropyLoss()
# model.to(device)
# #load model 

model = torchvision.models.vgg19(pretrained= True)
# %% Freeze the convolution layers

#%% Replace the fc layers
num_classes = 3
model.classifier[6] = nn.Linear(in_features = 4096, out_features = num_classes, bias=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


yaml_paths = './data/data_dir.yaml'
with open(yaml_paths, 'r') as yaml_file:
    file_paths = yaml.safe_load(yaml_file)
    
save_dir= file_paths['WEIGHTS']['HunCRC_classifier']
# save_dir = './checkpoints/img_classifier/HunCRC_image_vgg19_classifier_3classes.pth'
model.load_state_dict(torch.load(save_dir))
model.to(device)

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
print(traverse_data.keys())

predicted_classes = []
confidence_scores = []
model.eval()

with torch.no_grad():
    for img_batch in traverse_data['traverse_image_list']:
        batch = []
        for img in img_batch:
            img = transform(Image.fromarray(img.astype(np.uint8)))
            batch.append(img)
        batch = torch.stack(batch).to(device)
        output = model(batch)
        confidence_percentage = torch.nn.functional.softmax(output, dim=1).cpu().numpy()
        confidence_scores.append(confidence_percentage)
        output = np.argmax(output.cpu().numpy(), axis=1)
        predicted_classes.append(output)
    predicted_classes = np.array(predicted_classes)
    confidence_scores = np.array(confidence_scores)
traverse_data['predicted_classes'] = {'classes': predicted_classes, 'confidence_scores': confidence_scores}
save_dir = args.save_dir
with open(save_dir,'wb') as f:
    pickle.dump(traverse_data, f, protocol = 4)
