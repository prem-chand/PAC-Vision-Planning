#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 22:40:15 2021

@author: rhaegar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(101)

device = torch.device("cuda:0")
# device = torch.device("cpu")

# load matlab data
# matdata = loadmat('/home/rhaegar/Github/3D_biped_updated/train_data.mat', struct_as_record=False, squeeze_me = True)
matdata = loadmat('/home/rhaegar/Github/3D_biped_updated/training_data.mat', struct_as_record=False, squeeze_me = True)
# train_data = matdata['b1']

state = matdata['state']
action = matdata['action']

# labels = torch.tensor(train_data.a, dtype=torch.long)
# outputs = np.array(train_data.a, dtype='int16')
# data = torch.tensor(train_data.state, dtype=torch.float32)

labels = torch.tensor(action, dtype=torch.long)
outputs = np.array(action, dtype='int16')
data = torch.tensor(state, dtype=torch.float32)

data_mean = torch.mean(data, dim = 0)
data_var = torch.var(data, dim = 0)

# normalized data
inputs = (data - data_mean)/torch.sqrt(data_var)

prim_list = [5*num-45 for num in range(19)]
label_to_idx = dict()

for i, action in enumerate(prim_list):
    label_to_idx[action] = torch.Tensor([i])
    
for i, label in enumerate(outputs):
    labels[i] = label_to_idx[label]
    

NUM_PRIM = 19
STATE_DIM = inputs.shape[0]

#  split data ito train and validation sets; 20% validation
n_samples = inputs.shape[1]
n_val = int(0.3 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

train_ip = inputs.T[train_indices]
train_op = labels[train_indices]

val_ip = inputs.T[val_indices]
val_op = labels[val_indices]

class Classifier(nn.Module):
    def __init__(self, state_dim=20, num_prim=19):
        # state-dim = 18(robot-state) + 2(F_x,F_y)
        super(Classifier,self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(state_dim,10),
            nn.ELU(),
            nn.Linear(10,10),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(10,num_prim),
            nn.LogSoftmax(dim=1),            
            )
        
    def forward(self, x): 
        return self.pipe(x)

# class Classifier(nn.Module):
    
#     def __init__(self, state_dim = 20, num_prim = 13):
        
#         super(Classifier, self).__init__()
#         self.fc1 = nn.Linear(state_dim,20)
#         self.fc2 = nn.Linear(20,20)
#         self.fc3 = nn.Linear(20,num_prim)
        
#     def forward(self, x):
#         x = torch.tanh(self.fc1(x))
#         x = torch.tanh(self.fc2(x))
#         x = F.log_softmax(self.fc3(x),dim=1)
#         return x

def opt(n_epochs, model, optimizer, loss_function, train_ip, train_op, val_ip, val_op, args, PATH):
    running_trn_loss = args['running_trn_loss']
    running_val_loss = args['running_val_loss']
    epoch = args['epoch']
    for epoch in range(epoch+1, epoch+n_epochs+1):
        
        
        train_log_probs = model(train_ip.to(device))
        train_loss = loss_function(train_log_probs, train_op.to(device))
        running_trn_loss.append(train_loss.item())
        
        with torch.no_grad():
            val_log_probs = model(val_ip.to(device))
            val_loss = loss_function(val_log_probs, val_op.to(device))
            running_val_loss.append(val_loss.item())
            assert val_loss.requires_grad == False
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if epoch%1000 == 0:
            print('iter: {}, training loss: {:.3f}, validation loss: {:.3f}'.format(epoch, train_loss.item(), val_loss.item()))
            
    plt.plot(running_trn_loss, 'bo')
    plt.plot(running_val_loss ,'go')
    plt.legend(['training loss','validation loss'])
    
    
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_rn_loss': running_trn_loss,
            'val_rn_loss': running_val_loss,
            # 'train_loss': train_loss
            }, PATH)
   
N_EPOCHS = 10000
use_saved = 0
LR = 1e-3
SAVE_PATH = '/home/rhaegar/Github/PAC-Vision-Planning/model.pt'
if use_saved:
    model = Classifier(STATE_DIM, NUM_PRIM)
    model.to(device)
    
    # num_params = sum(p.numel() for p in model.parameters())
    # print('Number of Neural Network Parameters:', num_params)    
    
    checkpoint = torch.load(SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    running_trn_loss = checkpoint['train_rn_loss']
    running_val_loss = checkpoint['val_rn_loss']
    
    args = {"epoch": epoch,
            "running_trn_loss": running_trn_loss,
            "running_val_loss": running_val_loss}
    
    model.eval()
    
else:
    model = Classifier(STATE_DIM, NUM_PRIM)
    model.to(device)
    
    # num_params = sum(p.numel() for p in model.parameters())
    # print('Number of Neural Network Parameters:', num_params)   
    
    optimizer = optim.Adam(model.parameters(), lr = LR)
    
    args = {"epoch":0,
            "running_trn_loss": [],
            "running_val_loss": []}
    
    # epoch = 0
    # running_trn_loss = []
    # running_val_loss = []
    
loss_function = nn.CrossEntropyLoss()
num_params = sum(p.numel() for p in model.parameters())
print('Number of Neural Network Parameters:', num_params)
opt(N_EPOCHS, model, optimizer, loss_function, train_ip, train_op, val_ip, val_op, args, SAVE_PATH)

# model = Classifier(STATE_DIM, NUM_PRIM)
# model.to(device)




# optimizer = optim.Adam(model.parameters(), lr = 1e-4)



# checkpoint = torch.load(SAVE_PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# running_trn_loss = checkpoint['train_rn_loss']
# running_val_loss = checkpoint['val_rn_loss']


# model.eval()



# opt(N_EPOCHS, model, optimizer, loss_function, train_ip, train_op, val_ip, val_op, SAVE_PATH)
# opt(N_EPOCHS, model, optimizer, loss_function, train_ip, train_op, val_ip, val_op, SAVE_PATH)








