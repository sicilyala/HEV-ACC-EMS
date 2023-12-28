"""
    3 car-following models in SUMO;
    V1: "Krauss", V2: "IDM", V3: "ACC";
"""
# import os
import scipy.io as scio
from common.utils import get_driving_cycle
from sumo_env import *


def sumo_main(scenario, model_sel):
    speed_list = get_driving_cycle(scenario)
    total_steps = len(speed_list)
    start_sumo()

    step_info = {'spd': [], 'acc': [], 'distance': [],
                 'follow_x': [], 'leading_x': []}
    model_name = {'V1': 'Krauss', 'V2': 'IDM', 'V3': 'ACC'}
    # model_sel = 'V3'
    datadir = './sumo_CF/'+scenario
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    file_v = '_v1'      # file name version
    dataname = datadir + '/'+model_name[model_sel]+file_v+'.mat'
    for i in range(total_steps):    # just one episode
        spd_p = speed_list[i]
        sumo_out = run_sumo(i, model_sel, spd_p)
        for key in step_info.keys():
            step_info[key].append(sumo_out[key])

    scio.savemat(dataname, mdict=step_info)
    

if __name__ == '__main__':
    scenario = "mix2"
    sumo_main(scenario, 'V3')   # model_name = {'V1': 'Krauss', 'V2': 'IDM', 'V3': 'ACC'}
