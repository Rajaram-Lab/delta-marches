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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda:0'
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
# load pretrained grade classifier for image 
model_cls = torchvision.models.vgg19(pretrained= True)
model_cls.classifier[6] = torch.nn.Linear(in_features = 4096, out_features = 3, bias=True)

save_dir= 'trained_models'
model_cls.load_state_dict(torch.load(os.path.join(save_dir, 'HunCRC_image_vgg19_classifier_3classes.pth')))

model_cls.eval()
model_cls.to(device)
#%% Transition from class a to b
save_dir = os.path.join('datasets', 'HunCRC_data.pkl')
with open(save_dir,'rb') as f:
    data = pickle.load(f)
    
org_class = 'normal'
target_class = 'highgrade_dysplasia'
org_idx = np.where(data['class'] == org_class)[0]
np.random.shuffle(org_idx)
org_class = 'normal'
target_class = 'highgrade_dysplasia'
org_idx = np.where(data['class'] == org_class)[0]
org_cond = data['img'][org_idx]
ratio = 30
cls_id = CelebAttrDataset.cls_to_id[target_class]

classes_3_list = ['normal', 'lowgrade_dysplasia', 'highgrade_dysplasia']
all3_idx = np.where(np.isin(data['class'], classes_3_list))[0]
all3_class_dict = {class_name: i for i, class_name in enumerate(classes_3_list)}
confidence_score = np.array(data['confidence_vgg19'])
#find in all3_idx the indices of samples that belong to org_class and have confidence_score > 0.9
org_idx = all3_idx[(data['class'][all3_idx] == org_class)]
confidence_score_org = confidence_score[(data['class'][all3_idx] == org_class)]
sorted_idx = org_idx[np.argsort(confidence_score_org[:,0])]
selected_idx = sorted_idx[:1000]

from tqdm import tqdm
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
transform_syn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
org_img_list =[]
from PIL import Image
for i in selected_idx:
   org_img_list.append(transform(Image.fromarray(data['img'][i])))
num_pic = 10
num_sample = len(selected_idx)
image_list_full =[]
batch_size = 25
start_point = 0
print(start_point)
step_size_list = []
xT_list = []
#Transition Procedure
for idx in range(math.ceil(len(selected_idx)/batch_size)):
    print(f'batch {idx}')
    image_list = []
    image_list_n = []
    batch = torch.stack(org_img_list[idx*batch_size:(idx+1)*batch_size])
    img =np.uint8(255*(batch.detach().cpu().numpy()+1)/2)
    image_list.append(img)
    cond = model.encode(batch.to(device))
    # xT = torch.randn_like(batch).to(device)
    xT = model.encode_stochastic(batch.to(device), cond, T=250)
    cls_id = CelebAttrDataset.cls_to_id[target_class]
    cond2 = cls_model.normalize(cond)
    num_step =1
    cond3 = cond2 
    
    step_size = []
    # Find the step size num_step for each image to reach the target class based on classifier model
    for i in range(batch.shape[0]):
        num_step = 1
        cond3[i] = cond2[i] + num_step* math.sqrt(512) * F.normalize(cls_model.classifier.weight[cls_id][None, :], dim=1)
        cond3[i] = cls_model.denormalize(cond3[i])
        img = model.render(xT[None,i,:,:,:], cond3[None,i,:], T=100)
        img = transform_syn(img[0].cpu().numpy().transpose(1,2,0))[None,:,:,:].to(device)
        
        output = model_cls(img)
        _, predicted = torch.max(output.data, 1)
        confidence_score = torch.max(torch.exp(output.data),1)[0]/torch.sum(torch.exp(output.data),1)

        # Increase the step size num_step in transition until reaching the target class
        while num_step <3 and predicted[0] !=2:
            num_step +=1
            cond3[i] = cond2[i] + num_step* math.sqrt(512) * F.normalize(cls_model.classifier.weight[cls_id][None, :], dim=1)
            cond3[i] = cls_model.denormalize(cond3[i])
            
            img = model.render(xT[None,i,:,:,:], cond3[None,i,:], T=100)
            img = transform_syn(img[0].cpu().numpy().transpose(1,2,0))[None,:,:,:].to(device)
            a, predicted = torch.max(model_cls(img),1)
        step_size.append(num_step)
    step_int = num_step/num_pic  
    step_size = np.array(step_size)
    step_size_list.append(step_size)
    # Interpolate to generate the intermediate images 
    for i in range(num_pic+1):
        print(f'interpolate step {i}')
        cond2 = (1-i/(num_pic))*cond + i/(num_pic)*cond3
        img = model.render(xT, cond2, T=100)
        imgResized=img.to(device)
        a, predicted = torch.max(model_cls(imgResized),1)
        img = np.uint8(255*img.detach().cpu().numpy())
        image_list.append(img) 
    image_list_n = np.stack(image_list)
    image_list_full.append(image_list_n.transpose(1,0,3,4,2))
    
    #Save data
    img_list = {'img': image_list_full, 'idx': selected_idx, 'step_size': step_size_list}
    save_dir = f'imgs_manipulated/img_transition_from{org_class}to{target_class}_{num_sample}pat.pkl'
    with open(save_dir,'wb') as f:
        pickle.dump(img_list, f, protocol=4)        

   