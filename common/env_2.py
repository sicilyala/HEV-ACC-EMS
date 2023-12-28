from common.car_agents_2 import Driver, Energy
from common.utils2 import get_driving_cycle
import torch


class Env:
    def __init__(self):
        self.agents = self._init_agents()
        self.agent_num = len(self.agents)
    
    @staticmethod
    def _init_agents():
        driver = Driver()
        energy = Energy()
        agents = [driver, energy]
        return agents
    
    def reset(self):
        obs_all = []
        for i in range(self.agent_num):
            observation = self.agents[i].reset_obs()
            obs_all.append(observation)
        return obs_all
    
    def step(self, actions):
        obs_all = []
        reward_all = []
        done_all = []
        info_all = []
        for i in range(self.agent_num):
            obs_all.append(self.agents[i].execute(actions[i]))
            # reward_all.append(self.agents[i].get_reward())
            reward_all.append([self.agents[i].get_reward()])        # for PER
            done_all.append(self.agents[i].get_done())
            info_all.append(self.agents[i].get_info())
            
        # print('Driver reward: %.2f; Energy reward: %.2f.' % (reward_all[0], reward_all[1]))
        return obs_all, reward_all, done_all, info_all,


def make_env(args):
    env = Env()
    args.n_agents = env.agent_num
    args.obs_shape = [env.agents[i].obs_num for i in range(args.n_agents)]
    args.action_shape = [env.agents[i].action_num for i in range(args.n_agents)]
    args.high_action = 1
    args.low_action = -1
    speed_list = get_driving_cycle(cycle_name=args.scenario_name)
    args.episode_steps = len(speed_list)  # cycle length, be equal to args.episode_steps
    # if args.cuda:
    #     args.device = torch.device("cuda:0")
    # else:
    #     args.device = torch.device("cpu")
    # print('device: %s'%args.device)
    return env, args
