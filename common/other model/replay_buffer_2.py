# replay buffer on GPU

import threading
# import numpy as np
import torch as t
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

class Buffer:
    def __init__(self, args):
        self.size = args.buffer_size
        # number of transitions can be stored in buffer, default: 5e5
        self.args = args
        # memory management
        self.current_size = 0
        self.counter = 0
        # create the buffer to store info
        self.buffer = dict()
        for i in range(self.args.n_agents):
            self.buffer['o_%d' % i] = t.empty([self.size, self.args.obs_shape[i]]).to(device)      # state
            self.buffer['u_%d' % i] = t.empty([self.size, self.args.action_shape[i]]).to(device)       # action
            self.buffer['r_%d' % i] = t.empty([self.size]).to(device)                                  # reward
            self.buffer['o_next_%d' % i] = t.empty([self.size, self.args.obs_shape[i]]).to(device)     # next state
        # thread lock   线程锁
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, o, u, r, o_next):
        # idxs = self._get_storage_idx(inc=1)  # 以transition的形式存，每次存inc条经验
        idxs = self.counter % self.size         # store one transition every time
        for i in range(self.args.n_agents):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = o[i]
                self.buffer['u_%d' % i][idxs] = u[i]
                self.buffer['r_%d' % i][idxs] = r[i]
                self.buffer['o_next_%d' % i][idxs] = o_next[i]
        self.counter += 1
        self.current_size = min(self.size, self.current_size+1)
    
    # sample the data from the replay buffer randomly
    def sample(self, batch_size):
        temp_buffer = {}
        # idx = np.random.randint(0, self.current_size, batch_size)
        idx = t.randint(0, self.current_size, (batch_size, ))
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)   # array([0])
        elif self.current_size < self.size:           # it's used when inc>1
            print('elif self.current_size < self.size')
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)              # "half-open" interval [low, high)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx
