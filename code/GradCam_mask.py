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
#This code is used to obtain the GraCam heatmap for transition and real data
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image
from torch import nn
import pickle
from PIL import Image
import torchvision.transforms as transforms
yaml_paths = './data/data_dir.yaml'

import yaml
yaml_paths = './data/data_dir.yaml'
with open(yaml_paths, 'r') as yaml_file:
    file_paths = yaml.safe_load(yaml_file)
    
    
save_dir = file_paths['real_data']['real_data_4000patches']
# save_dir = './dataset/grade_data_4000.pkl'
with open(save_dir,'rb') as f:
  data = pickle.load(f)

save_path = file_paths['WEIGHTS']['grade_classifier']
num_classes = 4
num_epochs = 50
learning_rate = 1e-4

model = models.vgg19(pretrained= True)
model.classifier[6] = nn.Linear(in_features = 4096, out_features = num_classes, bias=True)
model.load_state_dict(torch.load(save_path))
model.eval()

target_layers =  [model.features[-1]]

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


grad_cam_list = []
img_list = data['img'][:,None,:,:,:]


data['Gradcam'] = {}

with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
  # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
  # grayscale_cam = cam(input_tensor=img)
  for i in range(img_list.shape[0]):
    print(i)
    grad_cam_list_i = []
    for j in range(img_list.shape[1]):
      target_category = [ClassifierOutputTarget(j)]
      img = transform(img_list[i,j])
      img = img.unsqueeze(0)
      grayscale_cam = cam(input_tensor=img, targets=target_category)
      grad_cam_list_i.append(grayscale_cam)
    grad_cam_list_i = np.squeeze(grad_cam_list_i)
    grad_cam_list.append(grad_cam_list_i)
grad_cam_list = np.squeeze(grad_cam_list)


data['Gradcam']['Gradcampp_mask'] = grad_cam_list

save_dir = file_paths['real_data']['real_data_4000patches']
with open(save_dir,'wb') as f:
    pickle.dump(data,f, protocol= 4)