import numpy as np
import torch
from common.maddpg import MADDPG


class Brain:
    def __init__(self, agent_id, args, load_or_not):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id, load_or_not=load_or_not)
    
    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:  # epsilon greedy    以小概率进行探索
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi = self.policy.actor_network(inputs).squeeze(0)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            noise = noise_rate*self.args.high_action*np.random.randn(*u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, -self.args.high_action, self.args.high_action)  # clip 截取
        return u.copy()
    
    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)
    
    def select_action2(self, o):
        inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)  # 增加一个维度：shape(2,3) -> shape(1,2,3)
        pi = self.policy.actor_network(inputs).squeeze(0)  # 减少一个维度
        # u = pi.detach()
        # u = u.cpu().numpy()        # from GPU tensor to CPU numpy
        uu = pi.detach().numpy()
        return uu
    
    def save_model_net(self, save_episode):
        self.policy.save_model(save_episode)
    
    def lr_scheduler(self, episode):
        lra = self.policy.scheduler_lr_a.get_last_lr()[0]
        lrc = self.policy.scheduler_lr_c.get_last_lr()[0]
        print('\n'+'episode %d: agent_%d, lr_a %.6f, lr_c %.6f'%(episode, self.agent_id, lra, lrc))
        idd_a = 'lra_%d' % self.agent_id
        idd_c = 'lrc_%d' % self.agent_id
        # if self.agent_id == 1:
        self.policy.scheduler_lr_a.step()
        self.policy.scheduler_lr_c.step()
        return idd_a, idd_c, lra, lrc