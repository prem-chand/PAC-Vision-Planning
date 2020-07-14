#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 20:41:37 2020

@author: premchand
"""

import torch
import torch.nn as nn
from os.path import join
from policy.biped_policy import Policy
import matlab.engine
eng = matlab.engine.start_matlab()

num_steps = 25

eng.cd('/Users/premchand/Downloads/GitHub/3D-biped/old-code/executable/To python')
data = eng.get_init_data(nargout=11)
t0 = data[0]
x0 = data[1]
# beta = data[2]
# vL = data[3]
# pL_start = data[4]
p_stance_foot = data[5]
# dt = data[6]
# w = data[7]
# DP_Dbeta = data[8]
int_F_x = torch.as_tensor([[data[9]]])
int_F_y = torch.as_tensor([[data[10]]])

# inputs = torch.cat((x0,int_F_x,int_F_y),dim = 0)
# inputs = torch.transpose(inputs,0,1)
policy = Policy()
# num_params = sum(p.numel() for p in policy.parameters())
# print('Number of Neural Network Parameters:', num_params)

# Initialize the posterior distribution
# mu = nn.ParameterList([nn.Parameter(torch.zeros(num_params))])
# logvar = nn.ParameterList([nn.Parameter(torch.log(torch.ones(num_params)*4))])
# res = policy(inputs)[0]
# prim = int(res.max(0).indices + 1)

# t0 = torch.tensor([[0]])

cost0 = 0
is_fallen = False
for i in range(num_steps):
        # if i == 0:
        #     t0 = 0
        #     p_stance_foot = matlab.double([[0],[0],[0]])
    
    x1 = torch.as_tensor(x0)
    inputs = torch.cat((x1,int_F_x,int_F_y),dim = 0)
    inputs = torch.transpose(inputs,0,1)
    res = policy(inputs)[0]
    prim_idx = int(res.max(0).indices + 1)
        
    step_data = eng.simulate_step(t0,x0,prim_idx,p_stance_foot,nargout = 7)
    next_state = step_data[0]
    F_x = step_data[1]
    F_y = step_data[2]
    p_stance_foot = step_data[3]
    te = step_data[4]
    cost1 = step_data[5]
    is_fallen = step_data[6]
    if is_fallen:
        break
    cost0 = max(torch.as_tensor(cost0),torch.as_tensor(cost1))
    
    
    t0 = te
    x0 = next_state
    int_F_x = torch.as_tensor([[F_x]])
    int_F_y = torch.as_tensor([[F_y]])


# def get_next_state(t0,x0,beta,vL,pL_start,p_stance_foot,dt,w):
#     out = eng.get_next_state(t0,x0,beta,vL,pL_start,p_stance_foot,dt,w,nargout = 6)
#     next_state = out[0]
#     te1 = out[1]
#     te2 = out[2]
#     int_F_x = out[3]
#     int_F_y = out[4]
#     p_st_foot = out[5]
    
#     return(out[0],out[1],out[2],out[3],out[4],out[5])


