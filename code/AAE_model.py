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
#%% This code is to train and test AAE model for tumor grade dataset. Training data has all 120000 patches taken from 120 patients with 128 slides and 4 grades.
#%%
import torch
import pandas as pd
import os,sys
from PIL import Image
import time
from torch import nn, cuda, device
from torch.autograd import Variable
import torchvision
from torch.utils.data import Dataset# data management
import numpy as np
import torch.optim as optim # Optimization algorithm
import matplotlib.pyplot as plt
import torchvision as tv
import itertools
import pickle
from tqdm import tqdm
import random
#import itertools
from torchvision.utils import save_image
print(cuda.is_available())
device = device('cuda:0')
#device = device('cpu')
print(device)

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
import yaml
#%% create directory to dataset
# Load YAML file
# yaml_paths = './data/data_dir.yaml'
# with open(yaml_paths, 'r') as yaml_file:
#     file_paths = yaml.safe_load(yaml_file)
    
    
# save_dir = file_paths['real_data']['real_data_120000patches']

import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--full_data",
    type=str,
    default= './data/datasets/grade_data_120000.pkl',
    help="Path to the saved 120000 patches pickle file"
)

parser.add_argument(
    "--latent_dim",
    type=int,
    default = 1024,
    help="Latent dimension for the AAE model"
)

parser.add_argument(
    "--subset_data",
    type=str,
    default= './data/datasets/grade_data_4000.pkl',
    help="Path to the subset 4000 patches pickle file"
)

args = parser.parse_args()
save_dir = args.full_data
save_dir_4000 = args.subset_data
latent_dim = args.latent_dim
with open(save_dir,'rb') as f:
  data = pickle.load(f)
with open(save_dir_4000,'rb') as f:
    data_4000 = pickle.load(f)


def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()    
     
trainPatches = data['patches'] 
trainClasses = data['classes']   
train_patient_name = data['patients']
testPatches = data['patches'][:24000]
testClasses = data['classes'][:24000]
test_patient_name = data['patients'][:24000]   
testPatches = data['patches'][data_4000['idx']]
testClasses = data['classes'][data_4000['idx']]
test_patient_name = data['patients'][data_4000['idx']]


#%% Define the dataset
trainPatchClasses = torch.from_numpy(np.uint8(np.array(trainClasses,dtype=float))).type(torch.LongTensor)
testPatchClasses = torch.from_numpy(np.uint8(np.array(testClasses,dtype=float))).type(torch.LongTensor)
trainPatchpatient = torch.from_numpy(np.uint8(np.array(train_patient_name,dtype=float))).type(torch.LongTensor)
testPatchpatient = torch.from_numpy(np.uint8(np.array(test_patient_name,dtype=float))).type(torch.LongTensor)

data_arr = {"train": trainPatches,
            "val": testPatches}

target_arr = {"train": trainPatchClasses,
              "val": testPatchClasses}
patient_arr = {"train": trainPatchpatient,
              "val": testPatchpatient}

preproc_op = torchvision.transforms.ToTensor()
                
class MyDataset(Dataset):
    def __init__(self, data, target,  datadir, transform=None, preproc=None):
        self.data = data
        self.target = target
        self.datadir = datadir
        self.transform = transform
        self.preproc = preproc

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        z = self.datadir[index]

        if self.transform:
            x = self.transform(images=np.expand_dims(x, 0))

        if self.preproc:
            x = self.preproc(np.squeeze(x))
        return x, y, z

    def __len__(self):
        return len(self.data)
    
image_datasets = {"train": MyDataset(data_arr["train"], target_arr["train"], patient_arr["train"], transform=None, preproc=preproc_op),
                      "val": MyDataset(data_arr["val"], target_arr["val"], patient_arr["val"], transform=None, preproc=preproc_op)}

batch_size=32
train_loader = torch.utils.data.DataLoader(
        image_datasets['train'], shuffle = True, batch_size=batch_size, num_workers=8, pin_memory = True)
test_loader = torch.utils.data.DataLoader(
        image_datasets['val'], shuffle = False, batch_size=batch_size, num_workers=8, pin_memory = True)


#%%Hyperparameters
# latent_dim = 2048
num_epochs = 50
nz = latent_dim
learning_rate = 1e-4
learning_rate_D = 5e-5
#%% Design AAE model

b1=0.5
b2 = 0.999
class Encoder(nn.Module):
    def __init__(self, nc, nf, nz):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, nf, 5, 2, 2, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf, nf*2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*2, nf*4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),            
            
            nn.Conv2d(nf*4, nf*4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(nf*4, nf*4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(in_features=7*7*nf*4, out_features=nz),
        )

    def forward(self, input):
        y =  self.net(input)
        return y

    
class Unflatten(nn.Module):
    def __init__(self, shape):
        super(Unflatten, self).__init__()
        self.shape = shape
        
    def forward(self, input):
        return input.view(len(input), self.shape[0], self.shape[1], self.shape[2])
    
class Decoder(nn.Module):
    def __init__(self, nc, nz, nf):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=nz, out_features=7*7*nf*4),
            Unflatten((nf*4, 7, 7)),
            
            nn.ConvTranspose2d(nf*4, nf*4, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(nf*4, nf*4, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(nf*4, nf*2, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(nf*2, nf, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(nf, int(nf/2), 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(int(nf/2)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(int(nf/2), nc, 5, 1, 2, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.net(input)

# Discriminator for output
class Discriminator_output(nn.Module):
    def __init__(self, nc):
        super(Discriminator_output, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, 64, 4 , 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4 , 2,1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4 , 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),            
            nn.Conv2d(256, 512, 4 , 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4 , 2, 1, 1, bias=False),
            
            nn.Flatten(),
            nn.Linear(49, 1),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.net(input)
# Discriminator for lattent
class Discriminator(nn.Module):
    def __init__(self, nc, nz):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(nz, 1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1000, 1),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.net(input)
    
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
  # For fold results
results = {}
  
  # Set fixed random number seed
torch.manual_seed(42)
  # Start print
print('--------------------------------')


# Init the neural network
enc = Encoder(nc=3, nz=nz, nf=256).apply(reset_weights)
dec = Decoder(nc=3, nz=nz, nf=256).apply(reset_weights)
dis = Discriminator(nc=3, nz=nz).apply(reset_weights)
dis_out = Discriminator_output(nc=3).apply(reset_weights)
beta = 0.5
loss_list = []
# loss
adversarial_loss = nn.BCELoss()
adversarial_loss_out = nn.BCELoss()
reconstruction_loss = nn.MSELoss()

# optimizer
optimizer_G = torch.optim.Adam(itertools.chain(enc.parameters(), dec.parameters()), lr=learning_rate)
optimizer_D_out = torch.optim.Adam(dis_out.parameters(), lr=learning_rate)
optimizer_D = torch.optim.Adam(dis.parameters(), lr=learning_rate_D, betas=(b1, b2))

#if cuda:
enc.to(device)
dec.to(device)
dis.to(device)
dis_out.to(device)
adversarial_loss.to(device)
reconstruction_loss.to(device)
adversarial_loss_out.to(device)
# Run the training loop for defined number of epochs
for epoch in range(num_epochs):

    loss_list_mean = 0
    n_loss = 0
    mean_rec_error = 0
    mean_fake_out_error = 0
    mean_fake_error = 0
    for i, (x, idx, pu) in enumerate(tqdm(train_loader)):

        valid = Variable(Tensor(x.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(Tensor(x.shape[0], 1).fill_(0.0), requires_grad=False).to(device)

        x = x.to(device)
        
        img_shape = x[0].shape

        # reconstruction + generator loss
        for param in dis_out.parameters():
            param.requires_grad = False
        optimizer_G.zero_grad()
        fake_z = enc(x) #this change to make it faster but should check again with old code
        decoded_x = dec(fake_z)
        # fake_lat_vec = enc(decoded_x)
        validity_fake_z = dis(fake_z)      
        """Calculate GAN and L1 loss for the generator"""
        pred_fake = dis_out(decoded_x)
        fake_loss = adversarial_loss(pred_fake, valid)
        loss_list_mean += reconstruction_loss(decoded_x, x).detach()
        n_loss+=1
        mean_rec_error += (reconstruction_loss(decoded_x, x)/batch_size).detach()
        mean_fake_out_error += (fake_loss).detach()
        mean_fake_error += (adversarial_loss(validity_fake_z, valid)).detach()
        
        G_loss =0.001*adversarial_loss(validity_fake_z, valid) + 0.001*fake_loss +  reconstruction_loss(decoded_x, x)
        G_loss.backward()
        optimizer_G.step()
        
        #1)  update D_out       
        for param in dis_out.parameters():
            param.requires_grad = True
        optimizer_D_out.zero_grad()
        #fake
        # fake_AB = dec(enc(x))
        pred_fake = dis_out(decoded_x.detach())
        fake_loss = adversarial_loss(pred_fake, fake)
        
        # Real
        pred_real = dis_out(x)
        real_loss = adversarial_loss(pred_real, valid)
        D_out = 0.5*(fake_loss+real_loss)
        D_out.backward()
        optimizer_D_out.step()

        

        # 2) discriminator loss
        optimizer_D.zero_grad()
        real_z = Variable(Tensor(np.random.normal(0,1,(x.shape[0], latent_dim)))).to(device)
        real_loss = adversarial_loss(dis(real_z), valid)
        fake_loss = adversarial_loss(dis(fake_z.detach()), fake)
        D_loss = 0.5*(real_loss + fake_loss)
        D_loss.backward()
        optimizer_D.step()
    if n_loss>0:
        loss_list_mean = loss_list_mean/n_loss
    loss_list.append([epoch, loss_list_mean, mean_rec_error, mean_fake_out_error, mean_fake_error])
    print(
            "[Epoch %d/%d] [G loss: %f]  \n [mean_rec_error: %f] \n [mean_fake_out_error: %f] \n [mean_fake_error: %f] "
            % (epoch, num_epochs, G_loss.item(), mean_rec_error, mean_fake_out_error, mean_fake_error)
        
            )

    save_path = f'./checkpoints/AAE_model/AAE_model'
    torch.save(enc.state_dict(),save_path+f'_n{latent_dim}_enc.pth')
    torch.save(dec.state_dict(),save_path+f'_n{latent_dim}_dec.pth')
print('Starting testing')

batch_size=32
train_loader = torch.utils.data.DataLoader(
        image_datasets['train'], shuffle = False, batch_size=batch_size, num_workers=8, pin_memory = True)
test_loader = torch.utils.data.DataLoader(
        image_datasets['val'], shuffle = False, batch_size=batch_size, num_workers=8, pin_memory = True)


nz=latent_dim
enc_Path = f'./checkpoints/AAE_model/AAE_model_n{latent_dim}_enc.pth'
dec_Path = f'./checkpoints/AAE_model/AAE_model_n{latent_dim}_dec.pth'
enc = Encoder(nc=3, nf=256, nz=nz)
dec = Decoder(nc=3, nz=nz, nf=256)
enc.load_state_dict(torch.load(enc_Path))
enc = enc.to(device)
enc.eval()
dec.load_state_dict(torch.load(dec_Path))
dec = dec.to(device)
dec.eval()
from tqdm import tqdm
result = {'lat_vector':[], 'classes':[],'patients':[]}
with torch.no_grad():
    for i, (x, idx, y) in enumerate(tqdm(train_loader)):
#    if i<60:
        x = x.to(device)
        fake_vec  = enc(x)
        result['lat_vector'].append(fake_vec.detach().cpu().numpy())
        result['classes'].append(idx.numpy())
        result['patients'].append(y.numpy())
result['lat_vector'] = np.concatenate(result['lat_vector'], axis = 0)
result['classes'] = np.concatenate(result['classes'], axis = 0)
result['patients'] = np.concatenate(result['patients'], axis = 0)
save_dir = f'./analysis/all_patients_AAE_lat_vector_n{latent_dim}.pkl'
with open(save_dir,'wb') as f:
    pickle.dump(result, f, protocol=4)
    
    
result = {'org_image':[], 'rec_image':[], 'lat_vector':[], 'classes':[],  'patients':[]}
with torch.no_grad():
    for i, (x, idx, y) in enumerate(tqdm(test_loader)):
#    if i<60:
        x = x.to(device)
        fake_vec  = enc(x)
        result['org_image'].append(x.cpu().permute(0,2,3,1).numpy())
        result['rec_image'].append(dec(fake_vec).detach().cpu().permute(0,2,3,1).numpy())
        result['lat_vector'].append(fake_vec.detach().cpu().numpy())
        result['classes'].append(idx.numpy())
        result['patients'].append(y.numpy())
result['org_image'] = np.concatenate(result['org_image'], axis = 0)
result['rec_image'] = np.concatenate(result['rec_image'], axis = 0)
result['lat_vector'] = np.concatenate(result['lat_vector'], axis = 0)
result['classes'] = np.concatenate(result['classes'], axis = 0)
result['patients'] = np.concatenate(result['patients'], axis = 0)
save_dir = f'./analysis/test_patients_AAE_lat_vector_n{latent_dim}.pkl'
with open(save_dir,'wb') as f:
    pickle.dump(result, f, protocol=4)
