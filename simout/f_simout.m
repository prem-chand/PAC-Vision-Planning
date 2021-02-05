function simout = f_simout(seed)


simout.t = []; simout.p_stfoot = []; simout.p_stknee = []; simout.p_swhip = []; simout.p_sthip = [];
simout.p_swknee = []; simout.p_swfoot = []; simout.p_head = []; simout.Fext = []; simout.p_CM = [];
simout.pL = []; simout.t_step_end = []; simout.z = []; simout.z_step_end = [];
simout.pLdot = []; simout.pEdot = []; simout.yc = []; simout.ycdot = [];

simout_file = strcat('simout/simout',num2str(seed),'.mat');
save(simout_file,'simout')

end