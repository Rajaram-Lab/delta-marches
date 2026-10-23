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

parser.add_argument(
    "org_class",
    type=str,
    help="Original class of the images"
)
parser.add_argument(
    "target_class",
    type=str,
    help="Target class of the images"
)

args = parser.parse_args()
save_dir = args.save_dir
org_class = args.org_class
target_class = args.target_class

# Load data
with open(save_dir, "rb") as f:
    data = pickle.load(f)
    
num_sample =200
class_dict = {'grade_1': 0, 'grade_2': 1,'grade_3': 2, 'grade_4': 3}
import random

idx_list = np.where(data['classes'] == class_dict[org_class])[0]
np.random.shuffle(idx_list)
idx_list = idx_list[:num_sample]

#load class from 
transform =    transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

org_img_list =[]
from PIL import Image
for i in idx_list:
   org_img_list.append(transform(data['patches'][i]))
num_pic =10

image_class_list = np.zeros((num_sample,num_pic+1))
image_list_full =[]
batch_size = 100

#Transition Procedure
for idx in range(int(num_sample/batch_size)):
    print(f'batch {idx}')
    image_list = []
    image_list_n = []
    batch = torch.stack(org_img_list[idx*batch_size:(idx+1)*batch_size])
    img =np.uint8(255*(batch.detach().cpu().numpy()+1)/2)
    image_list.append(img)
    cond = model.encode(batch.to(device))
    xT = model.encode_stochastic(batch.to(device), cond, T=250)
    cls_id = CelebAttrDataset.cls_to_id[target_class]
    cond2 = cls_model.normalize(cond)
    scale_r =1
    cond3 = cond2 + scale_r* math.sqrt(512) * F.normalize(cls_model.classifier.weight[cls_id][None, :], dim=1)
    cond3 = cls_model.denormalize(cond3)
    
    # Find the step size scale_r for each image to reach the target class based on classifier model
    for i in range(batch.shape[0]):
        scale_r = 2
        cond3[i] = cond2[i] + scale_r* math.sqrt(512) * F.normalize(cls_model.classifier.weight[cls_id][None, :], dim=1)
        cond3[i] = cls_model.denormalize(cond3[i])
        img = model.render(xT[None,i,:,:,:], cond3[None,i,:], T=100)
        img = transform(img[0].cpu().numpy().transpose(1,2,0))[None,:,:,:].to(device)
        a, predicted = torch.max(model_cls(img),1)
        
        # Increase the step size scale_r in transition until reaching the target class
        while scale_r <8 and predicted[0] !=class_dict[target_class]:
            scale_r +=2
            cond3[i] = cond2[i] + scale_r* math.sqrt(512) * F.normalize(cls_model.classifier.weight[cls_id][None, :], dim=1)
            cond3[i] = cls_model.denormalize(cond3[i])
            
            img = model.render(xT[None,i,:,:,:], cond3[None,i,:], T=100)
            img = transform(img[0].cpu().numpy().transpose(1,2,0))[None,:,:,:].to(device)
            a, predicted = torch.max(model_cls(img),1)
    step_int = scale_r/num_pic  
    
    # Interpolate to generate the intermediate images 
    for i in range(num_pic):
        print(f'interpolate step {i}')
        cond2 = (1-i/(num_pic-1))*cond + i/(num_pic-1)*cond3
        img = model.render(xT, cond2, T=100)
        imgResized=img.to(device)
        a, predicted = torch.max(model_cls(imgResized),1)
        img = np.uint8(255*img.detach().cpu().numpy())
        image_list.append(img) 
    image_list_n = np.stack(image_list)
    image_list_full.append(image_list_n.transpose(1,0,3,4,2))
image_list_full = np.concatenate(image_list_full, axis = 0)

#Save data
img_list = {'img': image_list_full, 'idx': idx_list}
save_dir = f'imgs_manipulated/img_transition_from{org_class}to{target_class}_{num_sample}pat.pkl'
with open(save_dir,'wb') as f:
    pickle.dump(img_list, f, protocol=4)
    
import matplotlib.pyplot as plt
for i in range(10):   
    fig, ax = plt.subplots(1, num_pic+1, figsize=(5*(num_pic+1), 5))
    for j in range(image_list_full[i].shape[0]):
        ax[j].imshow(image_list_full[i][j])
        ax[j].axis('off')
    plt.savefig(f'imgs_manipulated/img_transition_{num_sample}pat_{i}_'+org_class +'_to_'+ target_class+'.png')
   