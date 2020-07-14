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
    def __init__(
            self, 
            state_dim=20, 
            num_prim=3
            ):
        # state-dim = 18(robot-state) + 2(F_x,F_y)
        super(Policy,self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(state_dim,64),
            nn.ReLU(),
            nn.Linear(64,num_prim),
            nn.Softmax(dim=1),            
        )
        
    def forward(self, x): 
        return self.pipe(x)
    
# if __name__ == "__main__": 
#     net = BipedPolicy(state_dim=20, num_prim=3) 
#     v = torch.FloatTensor(torch.zeros([1, 20], dtype=torch.float)) 
#     out = net(v) 
#     print(net) 
#     print(out)    
    
    
    
    
    
    
    
    
    
    