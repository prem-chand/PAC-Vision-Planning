#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:30:56 2020

@author: premchand
"""

# import torch
# import torch.nn as nn
# import math
# from os.path import join
# import scipy.io as io
import numpy as np
# from policy.biped_policy import Policy
import matlab.engine
eng = matlab.engine.start_matlab()

eng.cd('/Users/premchand/Downloads/GitHub/PAC_biped_matlab')   
eng.addpath(eng.genpath(eng.pwd()))

# eng.workspace['s'] = 0
eng.sim0(nargout = 0)

data = []
data0 = eng.workspace

data.append(data0)

while (eng.workspace['i'] <= data0['num_steps']):
    # print("int_F_x = ",eng.workspace['int_F_x'])
    # print("int_F_y = ",eng.workspace['int_F_y'])
    switch_count = int(eng.workspace['switch_count'])
    if eng.workspace['i'] == eng.workspace['vel_switch'][0][switch_count-1]:
        print(eng.workspace['t_switch_vel'])
        print(eng.workspace['t_start'])
        eng.workspace['t_switch_vel'] = eng.workspace['t_switch_vel'] + eng.workspace['t_start']
        eng.workspace['t_start'] = matlab.double([0])
        eng.workspace['pL_start'] = eng.workspace['simout']['lead'][-1]
        
        switch_count = int(eng.workspace['switch_count'])
        eng.workspace['switch_count'] = switch_count + 1
        vL = (np.array(eng.workspace['vL_desired'])[:,eng.workspace['switch_count'] - 1]).tolist()
        eng.workspace['vL'] = matlab.double(vL)

    if eng.workspace['i'] > 1:
        l = eng.atan2(eng.workspace['int_F_y'],eng.workspace['int_F_x'])
        m = eng.workspace['ye2_u'][0]
        turn = (np.double([l]) - np.double(m))*0.2*180/np.pi
        eng.workspace['turn'] = matlab.double(turn.tolist())
        
        if np.abs(eng.workspace['turn']) < 2:
            eng.workspace['turn'] = matlab.double([0])
        else:
            eng.workspace['turn'] = matlab.double([np.sign(eng.workspace['turn'])*30])
        
        # eng.workspace['betta'] = eng.pinv(eng.workspace['DP_DBetta'])*matlab.double()
            
            
            
            
            
            
    eng.sim1(nargout = 0)
    data.append(eng.workspace)
    eng.workspace['i'] = eng.workspace['i'] + 1
    