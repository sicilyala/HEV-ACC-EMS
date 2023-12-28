# 用训练好的样本初始化记忆池
import scipy.io as scio
import numpy as np
from utils2 import get_driving_cycle, get_acc_limit
from arguments import get_args


args = get_args()
SPEED_LIST = get_driving_cycle(cycle_name=args.scenario_name)  # speed of leading car
ACC_LIST = get_acc_limit(SPEED_LIST, output_max_min=False)  # acceleration of leading car
MILEAGE = sum(SPEED_LIST)/1000  # km

def build_memory():
    data_file = 'E:/SEU2/Program1/MADDPG-program/evaluate/mix2__/model10_mix2_v12_g_436/episode_data/data_ep0.mat'
    data = scio.loadmat(data_file)
    pointer = 0
    state, all_actions, reward, state_next = [], [], [], []
    
    # normalization
    speed_limit = 120/3.6
    acc_limit = 2.5
    s_dim0 = 5
    s_dim1 = 4
    
    s0 = np.zeros(s_dim0, dtype=np.float32)
    s1 = np.zeros(s_dim1, dtype=np.float32)
    a0 = np.zeros(1, dtype=np.float32)
    a1 = np.zeros(1, dtype=np.float32)
    s0_next = np.zeros(s_dim0, dtype=np.float32)
    s1_next = np.zeros(s_dim1, dtype=np.float32)
    
    # agent Driver, id==0
    # aa = data['spd']
    # a = data['spd'][pointer]
    s0[0] = data['spd'][0][pointer] / speed_limit
    s0[1] = data['acc'][0][pointer]/acc_limit
    s0[2] = data['distance'][0][pointer]/450
    s0[3] = SPEED_LIST[pointer] / speed_limit
    s0[4] = ACC_LIST[pointer]/acc_limit
    a0[0] = data['acc'][0][pointer]/acc_limit
    r0 = float(data['driver_reward'][0][pointer])
    s0_next[0] = data['spd'][0][pointer+1] / speed_limit
    s0_next[1] = data['acc'][0][pointer+1]/acc_limit
    s0_next[2] = data['distance'][0][pointer+1]/450
    s0_next[3] = SPEED_LIST[pointer+1] / speed_limit
    s0_next[4] = ACC_LIST[pointer+1]/acc_limit
    
    # agent energy id ==1
    s1[0] = data['SOC'][0][pointer]
    s1[1] = (data['tep_a'][0][pointer]-25)/35
    s1[2] = data['I'][0][pointer]/50
    s1[3] = data['P_mot'][0][pointer] / 150
    # s1[4] = data['follow_x'][0][pointer]/MILEAGE
    a0[0] = data['P_eng'][0][pointer]/62
    r1 = float(data['energy_reward'][0][pointer])
    s1_next[0] = data['SOC'][0][pointer+1]
    s1_next[1] = (data['tep_a'][0][pointer+1]-25)/35
    s1_next[2] = data['I'][0][pointer+1]/50
    s1_next[3] = data['P_mot'][0][pointer+1] / 150
    # s1_next[4] = data['follow_x'][0][pointer+1]/MILEAGE

    state.append(s0)
    state.append(s1)
    all_actions.append(a0)
    all_actions.append(a1)
    reward.append(r0)
    reward.append(r1)
    state_next.append(s0_next)
    state_next.append(s1_next)
    pointer += 1
    return state, all_actions, reward, state_next, pointer

if __name__ == '__main__':
    # args = get_args()
    state, all_actions, reward, state_next, pointer = build_memory()
    print(state)
    print(all_actions)
    print(reward)
    print(state_next)
    print(pointer)