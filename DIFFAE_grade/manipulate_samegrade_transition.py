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

# %%

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

#load trained grade classifier for image 
class_img_save_dir = ''
model_cls = torchvision.models.vgg19(pretrained= True)
model_cls.classifier[6] = torch.nn.Linear(in_features = 4096, out_features = 4, bias=True)
model_cls.load_state_dict(torch.load(class_img_save_dir))
model_cls.eval()
model_cls.to(device)

transform =   transforms.ToTensor()
transform_img_op =    transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#%% Same grade random interpolation

import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "save_dir",
    type=str,
    help="Path to the saved real pickle file"
)
parser.add_argument(
    "save_dir_traversal",
    type=str,
    help="Path to the saved traversal pickle file"
)

parser.add_argument(
    "org_class",
    type=str,
    help="Original class of the images"
)
args = parser.parse_args()
save_dir = args.save_dir
save_dir_traversal = args.save_dir_traversal
org_class = args.org_class

# Load data
with open(save_dir, "rb") as f:
    data = pickle.load(f)

# The patients are divided into 3 groups based on the origin of data to avoid some slides taken from same patients and this information is not shared in the data
patient_groups = [['10683', '10688', '10693', '10703', '198224', '198225', '198226',
       '198227', '198230', '198231', '198232', '198233', '198234',
       '198235', '198236', '198237', '198238', '198239', '198240',
       '198241'],
        ['5587', '5588', '5590', '5591', '5595', '5596', '5597',
       '5598', '5599', '5607', '5608', '5612', '5615', '5617', '5707',
       '5708', '5709', '5711', '5712', '5713', '5714', '5715', '5716',
       '5717', '5718', '5720', '5721', '5722', '5723', '5724', '5725',
       '5727', '5740', '5741', '5742', '5743', '5745', '5746', '5752',
       '5760', '5785', '5789', '5817', '5866', '5868', '5869', '5870',
       '5871', '5873', '5875', '5876', '5877', '5878', '5879', '5880',
       '5881', '5882', '5883', '5884', '5885', '5899', '5902', '5967',
       '5983', '6051', '6052', '6053', '6054', '6055', '6056', '6057',
       '6088', '6090', '6091', '6092', '6095', '6096', '6097', '6100',
       '6102'],
        [ '18472', '18475', '18477',
       '18480', '18482', '18484', '18485', '18501', '18509', '18522',
       '18911', '18917', '18918', '18920', '18921', '18922', '18923',
       '18927', '18928', '18930', '18931', '18932', '18933', '19255',
       '19278', '19424', '19696', '19725']]

class_dict = {'grade_1': 0, 'grade_2': 1,'grade_3': 2, 'grade_4': 3}
# class_dict = {'grade_1': 0}
# class_dict = {'grade_4': 3}
image_class_list =[]
image_list_full =[]

batch_size = 50
num_sample = 500
org_idx_list = []
target_idx_list = []
convert_list = []
img_full_list = []

# for org_class in class_dict.keys():
#   print('converting ', org_class)
  
  #load data
  #load origin idx
  # idx_list_full  = np.where(data['classes']==class_dict[org_class] )[0]
  # np.random.shuffle(idx_list_full)
  # idx_list = idx_list_full[:num_sample]
  # org_class = 'grade_1'
  
  #load origin idx
save_dir_traversal = args.save_dir_traversal
with open(save_dir_traversal,'rb') as f:
  traverse_data = pickle.load(f)
idx_list = traverse_data['idx'][traverse_data['full_grade_list'][:,0,0]]
  
#Find the random patches not from the same patient as the org sample but having same class to swap
idx_target = []
for j in range(num_sample):
  #find the sample idx such that it is not the same patient as the org sample but having same class
  org_patient =  data['patients'][idx_list[j]]
  # remain_patient = np.where(data['patients']!=org_patient)[0]
  #find the list in patient_groups having org_patient
  for group in patient_groups:
    if org_patient in group:
      # remove all the patients in the same group
      remain_patient = [idx for idx, patient_name in enumerate(data['patients']) if patient_name not in group]
  sameclass_idx = np.where(data['classes'][remain_patient]==class_dict[org_class])[0]
  np.random.shuffle(sameclass_idx)
  idx_target.append(remain_patient[sameclass_idx[0]])
idx_target = np.array(idx_target)  

org_img_list =[]
target_img_list = []
for i in range(num_sample):
    org_img_list.append(transform(data['patches'][idx_list[i]])*2-1)
    target_img_list.append(transform(data['patches'][idx_target[i]])*2-1)
org_img_list = torch.stack(org_img_list)
target_img_list = torch.stack(target_img_list)

#Image Converting
image_list = []
imgNp_class = []
for idx in range(int(num_sample/batch_size)):
    
    org_batch = org_img_list[idx*batch_size:(idx+1)*batch_size]
    target_batch =target_img_list[idx*batch_size:(idx+1)*batch_size]
    cond = model.encode(org_batch.to(device))
    xT = model.encode_stochastic(org_batch.to(device), cond, T=250)
    cond2 = model.encode(target_batch.to(device))
    #swap cond and cond2
    img = model.render(xT, cond2, T=100)
    img = np.uint8(255*(img.detach().cpu().numpy()))
    image_list.append(img) 

image_list = np.concatenate(image_list, axis = 0)
image_list = image_list.transpose(0,2,3,1)
target_idx_list = idx_target

traverse_data['patient_across_img'] = image_list
traverse_data['patient_across_img_idx'] = target_idx_list
with open(save_dir_traversal,'wb') as f:
    pickle.dump(traverse_data, f, protocol=4)

# img_list = {'org_img_idx': org_idx_list, 'target_img_idx': target_idx_list, 'convert_img': img_full_list, 'convert_class': convert_list}

# save_dir = '/imgs_manipulated/img_interpolate_samegrade_{num_sample}patpergrade.pkl'
# with open(save_dir,'wb') as f:
#     pickle.dump(img_list, f, protocol=4)
