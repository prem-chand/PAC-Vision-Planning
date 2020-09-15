#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 20:41:37 2020

@author: premchand
"""

import torch
import torch.nn as nn
from os.path import join
import scipy.io as io
import numpy as np
from policy.biped_policy import Policy
import matlab.engine
eng = matlab.engine.start_matlab()

num_steps = 25

simout = {}
simout['t'] = []
simout['stf'] = []
simout['stk'] = []
simout['swh'] = []
simout['sth'] = []
simout['swk'] = []
simout['swf'] = []
simout['head'] = []
simout['Fext'] = []
simout['CM'] = []
simout['lead'] = []
simout['t_step_end'] = []
simout['z'] = []
simout['z_step_end'] = []

eng.cd('/Users/premchand/Downloads/GitHub/PAC_biped_matlab')   
eng.addpath(eng.genpath(eng.pwd()))
lib_speed = io.loadmat('/Users/premchand/Downloads/GitHub/PAC_biped_matlab/Fixed_Point_Lib_Speed/fps_lib1.mat')

data = lib_speed['fps_lib']

X_star1 = data[10,5:-1]
X_star1 = matlab.double(X_star1.tolist())

out1 = eng.fcn_alpha_red_leg1(X_star1)
out2 = eng.fcn_alpha_red_leg2(X_star1)

Dpoinc = io.loadmat('/Users/premchand/Downloads/GitHub/PAC_biped_matlab/precompute/Dp_Dbeta.mat')
DP_DBetta = Dpoinc['x']
DP_DBetta = np.delete(DP_DBetta,1,0)

Dspeed = io.loadmat('/Users/premchand/Downloads/GitHub/PAC_biped_matlab/precompute/fixed_0_G.mat')
G = Dspeed['G']

vL = np.array([[0.65],[0],[0]])

x2p = eng.impact_map_leg2(X_star1)

pL_start = eng.fcn_position_head1(eng.reshape(x2p,1,18))
p_stance_foot = matlab.double([[0],[0],[0]])

head = pL_start
time = 0

yi2 = matlab.double(np.zeros([7,1]).tolist())
ydoti2 = matlab.double(np.zeros([7,1]).tolist())

thetadot_plus2 = matlab.double([0])

S = np.diag(np.array([-1, 1, -1, 1, 1, -1, -1, 1, 1]))
T = np.bmat([[S,np.zeros([9,9])],[np.zeros([9,9]),S]])

X_star2 = T*np.array(X_star1).T
X_star2 = matlab.double(X_star2.tolist())

tau1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],[0, -1, 0, -1/2, 0, 0, 0, 0, 0]])
tau2 = np.bmat([[tau1,np.zeros([2,9])],[np.zeros([2,9]),tau1]])

X_star_u1 = tau2*np.array(X_star1).T
X_star_u2 = tau2*X_star2

ye2_u = X_star_u1
ye2 = eng.reshape(X_star1,1,18)

i = 1
turn = 0
beta = matlab.double(np.zeros([7,1]).tolist())
t_start = 0
t_switch_vel = 0
vel_switch = [5, 10, 15, 20, 0]
switch_count = 1
vL_des = np.array([[0.65, 0.55, 0.65, 0.55, 0.65],[0, -0.35, 0, -0.35, 0],[0, 0, 0, 0, 0]])

vL = vL_des[:,0]
prev_turn = 0

































































