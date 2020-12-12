#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 11:04:02 2020

@author: premchand
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, state_dim=18, num_prim=13):
        # state-dim = 18(robot-state) + 2(F_x,F_y)
        super(Policy,self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(state_dim,32),
            nn.ELU(),
            nn.Linear(32,24),
            nn.ELU(),
            nn.Linear(24,num_prim),
            nn.Softmax(dim=1),            
            )
        
    def forward(self, x): 
        return self.pipe(x)
    
def load_policy(policy, policy_params):

    count = 0
    for p in policy.parameters():
        num_params_p = p.data.numel()
        p.data = policy_params[count:count+num_params_p].view(p.data.shape)
        count+=num_params_p

    return policy