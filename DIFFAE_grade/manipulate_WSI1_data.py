"""
Copyright (C) 2026 Rajaram Lab, UT Southwestern Medical Center.

This file is part of the Delta-Marches project.

Delta-Marches is free software: you may redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation.

Delta-Marches is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details, see <https://www.gnu.org/licenses/>.

Author: Thuong Nguyen
Year: 2026
"""

from templates import *
from templates_cls import *
from experiment_classifier import ClsModel
import pickle
from PIL import Image
import torchvision.transforms as transforms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  
import pickle
import torchvision
import cv2
import torchvision.transforms.functional as fn
import numpy as np
from PIL import Image
import torch
device = 'cuda:0'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conf = grade224_autoenc_130M()
# load DIFFAE models
model = LitModel(conf)
state = torch.load(f'checkpoints/grade224_autoenc_130M/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

bunch_dict = {'a': b'1', 'b': b'2', 'c': b'3', 'd': b'4', 'e': b'5'}
import numpy as np
import matplotlib.pyplot as plt

import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "save_dir",
    type=str,
    help="Path to the saved pickle file"
)

parser.add_argument(
    "Patient_name",
    type=str,
    help="Patient name of the images"
)
parser.add_argument(
    "org_bunch",
    type=str,
    help="Original bunch name of the images"
)
parser.add_argument(
    "target_bunch",
    type=str,
    help="Target bunch name of the images"
)
args = parser.parse_args()
save_dir = args.save_dir
Patient_name = args.Patient_name
org_bunch_name = args.org_bunch
# org_slide_name = 'T1-A5-Flip'
#take all org_bunch_name before the last '-'
org_slide_name = org_bunch_name.split('-')[0] + '-' + org_bunch_name.split('-')[1] + '-Flip'
bunch_name = org_bunch_name.split('-')[-1]
import h5py
with h5py.File(save_dir + Patient_name + '-' + org_slide_name + '.hdf5', 'r') as f:
    patch_data = f['patches']['0'][:]
    classes = f['classes'][:]

idx_patches = np.where(classes == bunch_dict[bunch_name])[0]
print(len(idx_patches))
org_patch_data = patch_data[idx_patches]
print(org_patch_data.shape)
# Patient_name = 'KC01888'
target_bunch_name = args.target_bunch
target_slide_name = target_bunch_name.split('-')[0] + '-' + target_bunch_name.split('-')[1] + '-Flip'
bunch_name = target_bunch_name.split('-')[-1]

import h5py
with h5py.File(save_dir + Patient_name + '-' + target_slide_name + '.hdf5', 'r') as f:
    patch_data = f['patches']['0'][:]
    classes = f['classes'][:]
if np.unique(classes).shape[0] == 1:
    print('Only one class in this slide')
    target_patch_data = patch_data
else:
    idx_patches = np.where(classes == bunch_dict[bunch_name])[0]
    print(len(idx_patches))
    target_patch_data = patch_data[idx_patches]
print(target_patch_data.shape)
num_traverse_step = 11
transform =   transforms.ToTensor()
  #Find the random patches not from the same patient as the org sample but having same class to swap
idx_target = np.arange(target_patch_data.shape[0])
np.random.shuffle(idx_target)
org_img_list =[]
target_img_list = []
for i in range(250):
    org_img_list.append(transform(org_patch_data[i])*2-1)
    target_img_list.append(transform(target_patch_data[i])*2-1)
org_img_list = torch.stack(org_img_list)
target_img_list = torch.stack(target_img_list)
num_traverse_step = 11
image_list = []
from tqdm import tqdm
xT_list = []
# for j in tqdm(range(3)):
for j in tqdm(range(250)):
      
  idx1 = j
  idx2 = idx_target[j]  
  org_batch = org_img_list[idx1].unsqueeze(0)
  target_batch = target_img_list[idx2].unsqueeze(0)
  
  #Image Converting
  org_cond = model.encode(org_batch.to(device))
  target_cond = model.encode(target_batch.to(device))
  interpolate_cond = []
  for alpha in np.linspace(0, 1, num_traverse_step):
      interpolate_cond.append((1-alpha)*org_cond + alpha*target_cond)
  interpolate_cond = torch.concatenate(interpolate_cond)
  # batch_size = [num_traverse_step, 3, 224, 224]
  xT =  model.encode_stochastic(org_batch.to(device), org_cond, T=250)
  xT = torch.stack([xT.squeeze().detach().cpu()]*num_traverse_step).to(device)
  img = model.render(xT, interpolate_cond, T=100)
  img = np.uint8(255*(img.detach().cpu().numpy()))
  image_list.append(img) 

image_list = np.array(image_list)
# xT_list = np.array(xT_list)
traverse_data = {}
traverse_data['traverse_image']  = {'random_seed': idx_target, 'traverse_img_with_encoded_xT': image_list, 'xT': xT_list}
save_dir = './analysis/interpolate_syn_img_from_WSI1Data_orgslide_' + org_slide_name + '_' + org_bunch_name + '_targetslide_' + target_slide_name + '_'+ target_bunch_name+ str(num_traverse_step) + 'step.pkl'
with open(save_dir, 'wb') as f:
    pickle.dump(traverse_data, f, protocol= 4)