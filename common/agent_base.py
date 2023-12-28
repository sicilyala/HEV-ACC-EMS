import time

class BaseAgent:
    def __init__(self):
        self.number = ''  # str
        self.name = ''  # str
        self.obs_num = 0
        self.action_num = 0
        self.info = {}
        self.done = False
        time_v = time.strftime("%H_%M_%S", time.gmtime())
        self.driving_data = "common/data/driving_data_%s.npy" % time_v
        
    def reset_obs(self):
        """reset observation"""
        raise NotImplementedError()

    def execute(self, action):
        """execute action from network, then observe, return obs_next"""
        raise NotImplementedError()

    def get_reward(self):
        """return reward"""
        raise NotImplementedError()

    def get_done(self):
        """return done"""
        return self.done

    def get_info(self):
        """return information"""
        return self.info
    
    # def communicate(self, *args):
    #     """self agent obtain information from other agents"""
    #     pass
    