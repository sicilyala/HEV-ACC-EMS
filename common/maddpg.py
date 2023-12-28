import os
# os.environ["to_VISIBLE_DEVICES"] = "0"  # GPU编号

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print('device: %s'%device)

class MADDPG:
    def __init__(self, args, agent_id, load_or_not):
        # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        # self.train_step = 0
        self.c_loss = 0
        self.a_loss = 0
        
        # create the network
        self.actor_network = Actor(args, agent_id).to(device)
        self.critic_network = Critic(args).to(device)
        print('actor device: ', list(self.actor_network.parameters())[0].device)
        print('critic device: ', list(self.critic_network.parameters())[0].device)
        
        # build up the target network
        self.actor_target_network = Actor(args, agent_id).to(device)
        self.critic_target_network = Critic(args).to(device)
        print('actor target device: ', list(self.actor_target_network.parameters())[0].device)
        print('critic target device: ', list(self.critic_target_network.parameters())[0].device)
        
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        
        # create the optimizer
        # lr_a = [args.lr_actor_0, args.lr_actor_1]
        # lr_c = [args.lr_critic_0, args.lr_critic_1]
        lr_a = [0.001, 0.0005]
        lr_c = [0.005, 0.0005]
        base_lr_a = [0.00001, 0.00001]
        base_lr_c = [0.0001, 0.00001]
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=lr_a[self.agent_id])
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=lr_c[self.agent_id])
        
        # learning rate scheduler
        # if self.agent_id == 0:
        self.scheduler_lr_a = torch.optim.lr_scheduler.CyclicLR(self.actor_optim,  base_lr=base_lr_a[self.agent_id],
                                                                max_lr=lr_a[self.agent_id], step_size_up=50,
                                                                mode="triangular2", cycle_momentum=False)
        self.scheduler_lr_c = torch.optim.lr_scheduler.CyclicLR(self.critic_optim, base_lr=base_lr_c[self.agent_id],
                                                                max_lr=lr_c[self.agent_id], step_size_up=50,
                                                                mode="triangular2", cycle_momentum=False)
        # if self.agent_id == 1:
        #     self.scheduler_lr_a = torch.optim.lr_scheduler.CyclicLR(self.actor_optim, base_lr=base_lr_a[self.agent_id],
        #                                                             max_lr=lr_a[self.agent_id], step_size_up=50,
        #                                                             mode="triangular", cycle_momentum=False)
        #     self.scheduler_lr_c = torch.optim.lr_scheduler.CyclicLR(self.critic_optim, base_lr=base_lr_c[self.agent_id],
        #                                                             max_lr=lr_c[self.agent_id], step_size_up=50,
        #                                                             mode="triangular", cycle_momentum=False)

        # load model to evaluate
        if load_or_not is True:
            load_path = self.args.load_dir+'/'+self.args.load_scenario_name+'/'+'agent_%d'%agent_id
            actor_pkl = '/actor_params_ep%d.pkl'%self.args.load_episode
            critic_pkl = '/critic_params_ep%d.pkl'%self.args.load_episode
            load_a = load_path+actor_pkl
            load_c = load_path+critic_pkl
            if os.path.exists(load_a):
                self.actor_network.load_state_dict(torch.load(load_a))
                self.critic_network.load_state_dict(torch.load(load_c))
                print('Agent {} successfully loaded actor_network: {}'.format(agent_id, load_a))
                print('Agent {} successfully loaded critic_network: {}'.format(agent_id, load_c))
        else:
            # create the directory to store the model
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = self.args.save_dir+'/'+self.args.scenario_name
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
            self.model_path = self.model_path+'/'+'agent_%d'%agent_id
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
    
    # soft update for a single agent
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1-self.args.tau)*target_param.data+self.args.tau*param.data)
        
        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1-self.args.tau)*target_param.data+self.args.tau*param.data)
    
    # update the network
    def train(self, transitions, other_agents):
        for key in transitions.keys():
            transitions[key] = torch.as_tensor(transitions[key], dtype=torch.float32)
            transitions[key] = Variable(transitions[key])
        o, u, o_next, r0 = [], [], [], []  # 用来装每个agent经验中的各项
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d'%agent_id])  # [ [64, 4], [64, 2] ,[64, 3] ]
            u.append(transitions['u_%d'%agent_id])  # [ [64, 1], [64, 1] ,[64, 1] ]
            o_next.append(transitions['o_next_%d'%agent_id])
            r0.append(transitions['r_%d'%agent_id])  # [ [64, ], [64, ] ,[64, ] ]
        # # way 1
        r = transitions['r_%d'%self.agent_id]  # 训练时只需要自己的reward, np.array (64, )
        # # way 2
        # if self.agent_id == 0:
        #     r = r0[0]
        # else:
        #     r = r0[1]+r0[2]
        # # way 3
        # r = r0[0] + r0[1] + r0[2]
        r = r.to(device)
        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            # 得到下一个状态对应的动作
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target_network(o_next[agent_id]))
                else:
                    # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
                    # 可以直接输入所有agents，遍历输出动作
                    u_next.append(other_agents[index].policy.actor_target_network(o_next[agent_id]))
                    index += 1
            q_next = self.critic_target_network(o_next, u_next).detach()
            target_q = (r.unsqueeze(1)+self.args.gamma*q_next).detach()  # shape [64, 1]
            # unsqueeze()  增加维度
        # the q loss
        q_value = self.critic_network(o, u)  # shape [64, 1]
        critic_loss = (target_q-q_value).pow(2).mean()  # tensor ()     # TD-error: (target_q - q_value)
        self.c_loss = critic_loss.data
        
        # the actor loss
        # 更新当前agent的动作，其他agent的动作不变
        u[self.agent_id] = self.actor_network(o[self.agent_id])
        actor_loss = - self.critic_network(o, u).mean()  # tensor ()  # Q-value: the more the better
        self.a_loss = -actor_loss.data
        # update the network
        # if self.agent_id == 1:
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        self._soft_update_target_network()
        # if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
        #     self.save_model(self.train_step)
        # self.train_step += 1
    
    def save_model(self, save_episode):
        """save the model parameters at specified episode"""
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d'%self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path+'/actor_params_ep%d.pkl'%save_episode)
        torch.save(self.critic_network.state_dict(), model_path+'/critic_params_ep%d.pkl'%save_episode)

# define the actor network
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.high_action
        self.agent_id = agent_id
        # self.fc1 = nn.Linear(args.obs_shape[agent_id], 64)
        # self.fc2 = nn.Linear(64, 128)
        # self.fc3 = nn.Linear(128, 256)        # 全连接层    200,200
        # self.fc4 = nn.Linear(256, 128)
        # self.fc5 = nn.Linear(128, 64)
        # try easy network
        # self.fc1 = nn.Linear(args.obs_shape[agent_id], 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 64)
        # self.action_out = nn.Linear(64, args.action_shape[agent_id])
        # try easy network
        self.fc1 = nn.Linear(args.obs_shape[agent_id], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.action_out = nn.Linear(32, args.action_shape[agent_id])
    
    def forward(self, x):
        # print('here in agent %s Actor_forward()' % self.agent_id)
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        action = self.max_action*torch.tanh(self.action_out(x))  # tanh value section: [-1, 1]
        return action

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        # self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 64)
        # self.fc2 = nn.Linear(64, 128)
        # self.fc3 = nn.Linear(128, 256)
        # self.fc4 = nn.Linear(256, 128)
        # self.fc5 = nn.Linear(128, 64)
        # easier
        # self.fc1 = nn.Linear(sum(args.obs_shape)+sum(args.action_shape), 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 64)
        # self.q_out = nn.Linear(64, 1)
        # easier
        self.fc1 = nn.Linear(sum(args.obs_shape)+sum(args.action_shape), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.q_out = nn.Linear(32, 1)
    
    def forward(self, state, action):
        # print('here in Critic_forward()')
        state = torch.cat(state, dim=1)
        state = state.to(device)
        for i in range(len(action)):
            action[i] /= self.max_action
            action[i] = action[i].to(device)
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        q_value = self.q_out(x)
        return q_value