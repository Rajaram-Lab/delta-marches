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
import math
import torch
#device = 'cuda:3'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conf = grade224_autoenc_130M()
# load DIFFAE models
model = LitModel(conf)
state = torch.load(f'checkpoints/grade224_autoenc_130M/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)
cls_conf = grade224_autoenc_cls()
cls_model = ClsModel(cls_conf)
state = torch.load(f'checkpoints/grade224_autoenc_cls/last.ckpt',
                    map_location='cpu')
print('latent step:', state['global_step'])
cls_model.load_state_dict(state['state_dict'], strict=False)
cls_model.to(device)

#load pretrained grade classifier for image 
class_img_save_dir = ''
model_cls = torchvision.models.vgg19(pretrained= True)
model_cls.classifier[6] = torch.nn.Linear(in_features = 4096, out_features = 4, bias=True)
model_cls.load_state_dict(torch.load(class_img_save_dir))
model_cls.eval()
model_cls.to(device)

#%% Transition from grade a to b
# Enter input -output class
org_class = 'grade_1'
target_class = 'grade_4'


#%% Load data
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
with open(save_dir,'rb') as f:
  data = pickle.load(f)

    
idx_list = np.arange(24000)

#load class from 
transform =    transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

org_img_list =[]
from PIL import Image
for i in idx_list:
   org_img_list.append(transform(data['patches'][i]))
num_pic =10

num_sample = len(idx_list)
image_class_list = np.zeros((num_sample,num_pic+1))
image_list_full =[]
batch_size = 100

#Transition Procedure
for idx in range(int(num_sample/batch_size)):
    print(f'batch {idx}')
    batch = torch.stack(org_img_list[idx*batch_size:(idx+1)*batch_size])
    cond = model.encode(batch.to(device))
    xT = model.encode_stochastic(batch.to(device), cond, T=250)
    
    img = model.render(xT, cond, T=100)
    imgResized=img.to(device)
    a, predicted = torch.max(model_cls(imgResized),1)
    img = np.uint8(255*img.detach().cpu().numpy())
    image_list_full.append(img.transpose(0,2,3,1)) 
image_list_full = np.concatenate(image_list_full, axis = 0)

#Save data
img_list = {'rec_image': image_list_full, 'idx': idx_list}
save_dir = f'imgs_manipulated/image_reconstruction.pkl'
with open(save_dir,'wb') as f:
    pickle.dump(img_list, f, protocol=4)
    
   