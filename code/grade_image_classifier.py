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
#%% Uncomment the following code to run the training code
##%% create dataset
# Load YAML file
# yaml_paths = './data/data_dir.yaml'
# with open(yaml_paths, 'r') as yaml_file:
#     file_paths = yaml.safe_load(yaml_file)
    
    
# save_dir = file_paths['real_data']['real_data_120000patches']
# with open(save_dir,'rb') as f:
#   data = pickle.load(f)


# trainPatches = data['patches']
# trainClasses = data['classes']
# train_patient_name = data['patients']
# testPatches = data['patches'][:24000] 
# testClasses = data['classes'][:24000]   
# test_patient_name = data['patients'][:24000]   
# #%% Define the dataset
# trainPatchClasses = torch.from_numpy(np.uint8(np.array(trainClasses,dtype=float))).type(torch.LongTensor)
# testPatchClasses = torch.from_numpy(np.uint8(np.array(testClasses,dtype=float))).type(torch.LongTensor)
# trainPatchPatient = torch.from_numpy(np.uint8(np.array(train_patient_name,dtype=float))).type(torch.LongTensor)
# testPatchPatient = torch.from_numpy(np.uint8(np.array(test_patient_name,dtype=float))).type(torch.LongTensor)

# data_arr = {"train": trainPatches,
#             "val": testPatches}

# target_arr = {"train": trainPatchClasses,
#               "val": testPatchClasses}
# patient_arr = {"train": trainPatchPatient,
#               "val": testPatchPatient}
# #
# class MyDataset(Dataset):
#     def __init__(self, data, target,  datadir, transform=None, preproc=None):
#         self.data = data
#         self.target = target
#         self.datadir = datadir
#         self.transform = transform
#         self.preproc = preproc

#     def __getitem__(self, index):
#         x = self.data[index]
#         y = self.target[index]
#         z = self.datadir[index]

#         if self.transform:
#             x = self.transform(images=np.expand_dims(x, 0))

#         if self.preproc:
#             x = self.preproc(Image.fromarray(np.squeeze(x)))
#         return x, y, z

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
     
     
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
# image_datasets = {"train": MyDataset(data_arr["train"], target_arr["train"], patient_arr["train"], transform=None, preproc=transform),
#                       "val": MyDataset(data_arr["val"], target_arr["val"], patient_arr["val"], transform=None, preproc=transform)}

# batch_size=32
# train_loader = torch.utils.data.DataLoader(
#         image_datasets['train'], shuffle = True, batch_size=batch_size, num_workers=8, pin_memory = True)
# test_loader = torch.utils.data.DataLoader(
#         image_datasets['val'], shuffle = False, batch_size=batch_size, num_workers=8, pin_memory = True)


# num_classes = 4
# num_epochs = 50
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

# # %%Loss and optimizer
# #criterion = nn.CrossEntropyLoss()
# #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

# classes = trainPatchClasses.numpy()
# from sklearn.utils.class_weight import compute_class_weight# num_classes - int value representing number of classes# trainLabels - np array with all training labelscw = compute_class_weight(class_weight='balanced',
# class_weights = compute_class_weight(
#                                         "balanced",
#                                         np.unique(classes),
#                                         classes                                                   
#                                     )
# class_weights = dict(zip(np.unique(classes), class_weights)),
# class_weights = class_weights[0]
# class_weights = torch.FloatTensor(list(class_weights.values()))
# criterion = nn.CrossEntropyLoss(weight=class_weights)

# model.to(device)
# criterion.to(device)
# #%%print confusion matrix
# from sklearn.metrics import confusion_matrix
# import seaborn as sn
# # save_dir= os.path.join('checkpoints', '')
# class_names=['Grade1', 'Grade2', 'Grade3', 'Grade4' ]
# def confusion_matrix_plot(epoch,plot=True):
#     with torch.no_grad():
    
#         y_pred = []
#         y_true = []
    
#         # iterate over test data
#         for i, (images, labels, idx) in enumerate(test_loader):
#                 output = model(images.to(device)) # Feed Network
       
#                 output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
#                 y_pred.extend(output) # Save Prediction
                
#                 labels = labels.data.cpu().numpy()
#                 y_true.extend(labels) # Save Truth
        
#         # constant for classes
#         classes = class_names
        
#         # Build confusion matrix
#         cf_matrix = confusion_matrix(y_true, y_pred)
#         df_cm = pd.DataFrame((cf_matrix.T/np.sum(cf_matrix,1)).T, index = [i for i in classes],
#                              columns = [i for i in classes])
#         if plot:
#             plt.figure(figsize = (12,7))

#             plt.title('epoch %d' %epoch)
#             sn.heatmap(df_cm, annot=True)
#             # plt.savefig(save_dir+'confusionmatrix_image_classifier.png')
# # %%Train the model
# total_step = len(train_loader)
# from tqdm import tqdm
# for epoch in range(num_epochs):
#     for i, (images, labels, idx) in enumerate(tqdm(train_loader)): 
#         images = images.to(device)
#         labels = labels.to(device)
        
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print ('Epoch [{}/{}],  Loss: {:.8f}' 
#                    .format(epoch+1, num_epochs, loss.item()))
#     confusion_matrix_plot(epoch+1, plot=True)
#     if epoch >5:
#         torch.save(model.state_dict(),save_dir+'grade_classifier_image_224_weight_alldata.pth')
        
#%%  classify grade for synthetic image, find list of full grade transition, find list of high and low grade

# data_dir = file_paths['synthetic_data']['synthetic_data_grade1to4']

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
    
yaml_paths = './data/data_dir.yaml'
with open(yaml_paths, 'r') as yaml_file:
    file_paths = yaml.safe_load(yaml_file)
    
save_path= file_paths['WEIGHTS']['grade_classifier']

model_cls = torchvision.models.vgg19(pretrained= True)
model_cls.classifier[6] = nn.Linear(in_features = 4096, out_features = 4, bias=True)
model_cls.load_state_dict(torch.load(save_path))
model_cls.eval()
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#classify grade for synthetic image and confidence score
class_list = []
confidence_score_list = []
low_list = []
high_list = []
for i in range(traverse_data['img'].shape[0]):
    class_list_i = []
    batch = []
    print(i)
    for j in range(traverse_data['img'].shape[1]):
        img= transform(traverse_data['img'][i,j])
        batch.append(img)
    batch = torch.stack(batch, axis = 0)
    output = model_cls(batch)
    _, predicted = torch.max(output.data, 1)
    class_list.append(predicted.numpy())
    confidence_score = torch.max(torch.exp(output.data),1)[0]/torch.sum(torch.exp(output.data),1)
    confidence_score_list.append(confidence_score)
    low_list.append([(i,j) for j in range(traverse_data['img'].shape[1]) if predicted[j] < 2 and confidence_score[j] > 0.9])
    high_list.append([(i,j) for j in range(traverse_data['img'].shape[1]) if predicted[j] >= 2 and confidence_score[j] > 0.9])
          
class_list = np.array(class_list)
confidence_score_list = torch.stack(confidence_score_list).numpy()

traverse_data['class'] = class_list
traverse_data['confidence_score'] = confidence_score_list

#find list of full grade transition
full_grade = [i for i in range(traverse_data['class'] .shape[0]) if len(np.unique(traverse_data['class'][i])) == 4]
full_grade_idx = []
for i in full_grade:
    full_grade = []
    for g in range(4):
        grade_idx = np.where(np.array(traverse_data['class'][i])==g)[0]
        idx_max = grade_idx[np.argmax(confidence_score_list[i][grade_idx])]
        full_grade.append([i,idx_max])
    full_grade = np.array(full_grade)
    full_grade_idx.append(full_grade)
full_grade_idx = np.array(full_grade_idx)

#If the transition is from grade 1 to grade 4, we choose real image for grade 1 in full_grade_list
full_grade_idx[:,0,1] = 0
# If the transition is from grade 4 to grade 1, we choose real image for grade 4 in full_grade_list
# full_grade_idx[:,3,1] = 0

traverse_data['full_grade_list'] = full_grade_idx
print('full grade ', len(full_grade_idx))
#from each transition sample take only one sample per low/high group 
low_list_unique = []
high_list_unique = []
for i in range(traverse_data['img'].shape[0]):
    if (len(low_list[i]) > 0 and len(high_list[i]) > 0):
        idx_low = np.random.choice(np.array(low_list[i])[:,1],1)[0]
        low_list_unique.append((i,idx_low))
        idx_high = np.random.choice(np.array(high_list[i])[:,1],1)[0]
        high_list_unique.append((i,idx_high))

low_list_unique = np.array(low_list_unique)
high_list_unique = np.array(high_list_unique)
traverse_data['low_list_unique'] = low_list_unique
traverse_data['high_list_unique'] = high_list_unique


save_dir = args.save_dir
with open(save_dir,'wb') as f:
    pickle.dump(traverse_data,f, protocol= 4)
