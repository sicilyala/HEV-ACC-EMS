from tqdm import tqdm
import numpy as np
import torch
import os
import scipy.io as scio
from common.arguments import get_args
from dqn_model import DQN_model, Memory
from common.agentEMS_DQN import EMS_DQN
# from common.SHEV import SHEV_model
# from common.Cell import CellModel_2
from env_dqn import make_env

# def main_ACC(args):
#     save_path = args.save_dir+'/'+args.scenario_name
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     save_path_episode = save_path+'/episode_data'
#     if not os.path.exists(save_path_episode):
#         os.makedirs(save_path_episode)
#
#     # action_space = [-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5]
#     action_space = [-2.0, -1.0, -0.5, 0, 0.5, 1.0, 2.0]
#     # action_space = [0.5, 0, -0.5]
#     driver = Driver(action_num=len(action_space))
#     memory = Memory(memory_size=args.buffer_size, batch_size=args.batch_size)
#     dqn_agent = DQN_model(args, s_dim=driver.obs_num, a_dim=driver.action_num)
#
#     average_reward = []  # average_reward of each episode
#     DONE = {}
#     average_loss = []
#     initial_epsilon = 0.90
#     finial_epsilon = 0.01
#     epsilon_decent = (initial_epsilon-finial_epsilon)/500
#     epsilon = initial_epsilon
#
#     for episode in tqdm(range(500)):
#         state = driver.reset_obs()  # ndarray
#
#         rewards = []
#         loss = []
#         episode_info = {'spd': [], 'acc': [], 'distance': [], 'leading_x': [], 'follow_x': [],
#                         'collision': [], 'TTC': [], 'jerk_value': [],
#                         'r_jerk': [], 'r_speed': [], 'r_safe': [], 'driver_reward': []}
#         for episode_step in range(args.episode_step):
#             with torch.no_grad():
#                 action_id, epsilon_using = dqn_agent.e_greedy_action(state, epsilon)
#                 action = action_space[action_id]  # float
#             next_state = driver.execute(action)
#             reward = driver.get_reward()
#             done = driver.get_done()
#             info = driver.get_info()
#
#             memory.store_trasition(state, action_id, reward, next_state)
#             state = next_state
#
#             rewards.append(reward)
#             if done:
#                 # print('SOC failure in step %d of episode %d'%(episode_step, episode))
#                 if episode not in DONE.keys():
#                     DONE.update({episode: episode_step})
#             for key in episode_info.keys():
#                 episode_info[key].append(info[key])
#
#             if memory.current_size > 100*args.batch_size:
#                 minibatch = memory.uniform_sample()
#                 dqn_agent.train(minibatch)
#                 loss.append(dqn_agent.loss)
#
#             # save data in .mat
#             if episode_step+1 == args.episode_step:
#                 datadir = save_path_episode+'/data_ep%d.mat'%episode
#                 # episode_info.update(episode_info)
#                 colli_times = sum(episode_info['collision'])
#                 f_travel = info['follow_x']/1000
#                 l_travel = info['leading_x']/1000
#                 # feul_cost = info[1]['fuel_consumption']  # in L
#                 # feul_cost = 100*feul_cost/f_travel
#                 episode_info.update({'colli_times': int(colli_times)})
#                 scio.savemat(datadir, mdict=episode_info)
#
#                 # end inside an episode
#                 mean_r = np.mean(rewards)
#                 mean_loss = np.mean(loss)
#                 average_reward.append(mean_r)
#                 average_loss.append(mean_loss)
#                 print('\n'+'episode %d: l_travel %.3fkm, f_travel %.3fkm, '
#                            'collision %d, reward %.2f, loss %.2f, epsilon %.4f'
#                       % (episode, l_travel, f_travel, colli_times, mean_r, mean_loss, epsilon_using))
#             # episode_step += 1
#
#         epsilon -= float(epsilon_decent)
#
#     scio.savemat(save_path+'/ep_mean_r.mat', mdict={'ep_r': average_reward})
#     scio.savemat(save_path+'/ep_mean_loss.mat', mdict={'loss': average_loss})
#
#     print('buffer counter:', memory.counter)
#     print('buffer current size:', memory.current_size)
#     print('replay ratio: %.3f'%(memory.counter/memory.current_size)+'\n')
#     print('done:', DONE)

def main_EMS(args):
    save_path = args.save_dir+'/'+args.scenario_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path_episode = save_path+'/episode_data'
    if not os.path.exists(save_path_episode):
        os.makedirs(save_path_episode)
    
    # action_space = [-5.0, 0, 5.0]
    action_num = 63
    action_space = np.arange(0, action_num, 1, dtype=np.float32)
    ems = EMS_DQN()
    memory = Memory(memory_size=args.buffer_size, batch_size=args.batch_size)
    dqn_agent = DQN_model(args, s_dim=4, a_dim=action_num)
    
    average_reward = []  # average_reward of each episode
    DONE = {}
    average_loss = []
    fuel = []
    electric = []
    money = []
    lr = []
    initial_epsilon = 1.0
    finial_epsilon = 0.01
    epsilon_decent = (initial_epsilon-finial_epsilon)/450
    epsilon = initial_epsilon
    
    datapath = "E:/SEU2/Program1/MADDPG-program/model10/" \
                    "mix2_v12_PER_v4/" \
                    "episode_data/data_ep495.mat"
    data = scio.loadmat(datapath)
    
    SPD_LIST = data['spd'][0]
    ACC_LIST = data['acc'][0]
    episode_step_num = SPD_LIST.shape[0]  # 4619
    MILE = np.sum(SPD_LIST)/1000
    print('mileage: %.3fkm' % MILE)

    for episode in tqdm(range(500)):
        state = ems.reset_obs()  # ndarray
        
        rewards = []
        loss = []
        episode_info_EMS = {'W_eng': [], 'T_eng': [], 'P_eng': [], 'fuel_eff': [],
                             'W_ISG': [], 'T_ISG': [], 'P_ISG': [], 'Gen_eff': [],
                             'W_mot': [], 'T_mot': [], 'P_mot': [], 'Mot_eff': [],
                             'fuel_consumption': [], 'T_axle': [], 'W_axle': [],
                             'SOC': [], 'SOH': [], 'cell_power_out': [], 'P_batt': [],
                             'cell_OCV': [], 'cell_Vt': [], 'cell_V_3': [],
                             'I': [], 'I_c': [], 'tep_a': [], 'dsoh': [],
                             'fuel_cost': [], 'battery_reward': [], 'energy_reward': [],
                             'delta_SOC': [], 'cost_SOC': [], 'cost_SOH': [], 'cost_power': []}
        for episode_step in range(episode_step_num):
            with torch.no_grad():
                action_id, epsilon_using = dqn_agent.e_greedy_action(state, epsilon)
            action = action_space[action_id]  # float
            spd = SPD_LIST[episode_step]
            acc = ACC_LIST[episode_step]
            next_state = ems.execute(action, spd, acc)
            reward = ems.get_reward()
            done = ems.get_done()
            info = ems.get_info()
            
            memory.store_trasition(state, action_id, reward, next_state)
            state = next_state
            
            rewards.append(reward)
            if done:
                # print('SOC failure in step %d of episode %d'%(episode_step, episode))
                if episode not in DONE.keys():
                    DONE.update({episode: episode_step})
            for key in episode_info_EMS.keys():
                episode_info_EMS[key].append(info[key])
            
            if memory.current_size > 100*args.batch_size:
                minibatch = memory.uniform_sample()
                dqn_agent.train(minibatch)
                loss.append(dqn_agent.loss)
            
            # end of an episode: sava model params, save data, print info
            if episode_step+1 == args.episode_step:
                dqn_agent.save_model(episode)
                datadir = save_path_episode+'/data_ep%d.mat'%episode

                # h2_equal = float(sum(episode_info_EMS['h2_equal']))
                h2_equal = float(sum(episode_info_EMS['fuel_consumption']))
                h2_g_100km = h2_equal/MILE*100
                episode_info_EMS.update({'fuel_100km': h2_g_100km})
                fuel.append(h2_g_100km)

                SOC_0 = episode_info_EMS['SOC'][0]
                SOC_end = episode_info_EMS['SOC'][-1]
                elec_100km = 111.5*(SOC_0-SOC_end)/MILE*100
                electric.append(elec_100km)
                episode_info_EMS.update({'elec_kWh_100km': elec_100km})

                # money_epi = float(sum(episode_info_EMS['total_money_revised']))
                # money_100km = money_epi/MILE*100
                # episode_info_EMS.update({'money_100km': money_100km})
                # money.append(money_100km)

                # print('episode %d: SOC %.3f, h2_100km %.3fg, electric_100km %.3fkWh,'
                #       ' money_100km ￥%.3f'
                #       % (episode, SOC_end, h2_g_100km, elec_100km, money_100km))
                print('episode %d: SOC %.3f, h2_100km %.3fg, electric_100km %.3fkWh'
                      % (episode, SOC_end, h2_g_100km, elec_100km))
                
                scio.savemat(datadir, mdict=episode_info_EMS)
                
                mean_r = np.mean(rewards)
                mean_loss = np.mean(loss)
                average_reward.append(mean_r)
                average_loss.append(mean_loss)
                print('\n'+'episode %d: reward %.6f, loss %.6f, epsilon %.6f'
                      % (episode, mean_r, mean_loss, epsilon_using))
        
        epsilon -= float(epsilon_decent)
        lr0 = dqn_agent.scheduler_lr.get_last_lr()[0]
        print('episode %d: lr %.6f' % (episode, lr0))
        lr.append(lr0)
        dqn_agent.scheduler_lr.step()
    
    scio.savemat(save_path+'/ep_mean_r.mat', mdict={'ep_r': average_reward})
    scio.savemat(save_path+'/ep_mean_loss.mat', mdict={'loss': average_loss})
    scio.savemat(save_path+'/fuel.mat', mdict={'fuel': fuel})
    scio.savemat(save_path+'/electric.mat', mdict={'electric': electric})
    scio.savemat(save_path+'/money.mat', mdict={'money': money})
    scio.savemat(save_path+'/lr.mat', mdict={'lr': lr})
    
    print('buffer counter:', memory.counter)
    print('buffer current size:', memory.current_size)
    print('replay ratio: %.3f'%(memory.counter/memory.current_size)+'\n')
    print('done:', DONE)

if __name__ == '__main__':
    args = get_args()
    args = make_env(args)
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Random seeds have been set to %d!" % seed)
    
    print('cycle name: ', args.scenario_name)
    
    # main_ACC(args)
    main_EMS(args)