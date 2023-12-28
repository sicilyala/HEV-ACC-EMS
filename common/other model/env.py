### Designed by Jingda Wu, PhD student in MAE, NTU

import numpy as np
import math
import pickle

class Battery:

    def __init__(self, init_temperature, init_soc, init_soh, target_soc=0.9, max_soc = 0.99, min_soc = 0.1, max_current_scale = 6, min_cell_voltage = 2, max_cell_voltage = 6.6, trainable = True):
        self.tf = init_temperature
        self.init_soc = init_soc
        self.init_soh = init_soh
        self.soh_scale = 0
        self.target_soc = target_soc
        self.max_soc = max_soc
        self.min_soc = min_soc
        self.max_v_cell = max_cell_voltage
        self.min_v_cell = min_cell_voltage
        
        self.max_current_scale = max_current_scale
        
        self.previous_action = 0

        self.restart()

    def restart(self):
        self.count = 0 # time counting

        self.current = 0 # unit, A
        self.crate = 0
        self.voltage = 0 # unit, V
        self.power_out = self.current * self.voltage # unit, W
        
        ran = False # determine whether the initial soc is random from 0.3-0.9 or determined init_soc value
        self.soc = np.random.randint(3,9)/10 if ran else self.init_soc
        self.soh = self.init_soh
        self.soh_deriv = 0
        
        self.previous_action = 0

        self.v1 = 0
        self.v2 = 0
        self.ts = self.tf # inital surface temperature. unit, Calsius
        self.tc = self.tf # inital central temperature. unit, Calsius
        self.ta = self.tf # inital average temperature. unit, Calsius

        # the ocv is retrived from the paper Wu2018Continuous 
        self.ocv_list  = pickle.load(open('B.pkl','rb'))

        # the battery heat model parameters should be replaced with your parameters
        self.r0_list = 'xxx'
        self.r1_list = 'xxx'
        self.r2_list = 'xxx'
        self.c1_list = 'xxx'
        self.c2_list = 'xxx'
        self.b_list = 'xxx'

        # used in heating model
        self.Cn=2.3 
        self.Cc=62.7
        self.Cs=4.5
        self.Ru=8
        self.Rc=1.94

        self.z = 0.55 # used in ageing model
        self.ideal_gas_constant = 8.31


    def run_step(self,action):
        self.current = action * self.max_current_scale * self.Cn

        self.ocv = self.ocv_list(self.soc)
        self.r0 = self.r0_list(self.ta)
        self.r1 = self.r1_list(self.ta,self.soc)
        self.r2 = self.r2_list(self.ta,self.soc)
        self.c1 = self.c1_list(self.ta,self.soc)
        self.c2 = self.c2_list(self.ta,self.soc)
        self.b = self.b_list(self.soc)

        cell_heat = self.current * (self.v1 + self.v2 + self.r0 * self.current)

        total_Qbatt = self.ocv * self.Cn

        self.crate = abs(self.current / self.Cn)

        self.voltage = self.ocv + self.v1 + self.v2 + self.r0 * self.current

        self.power_out = self.voltage * self.current

        soc_deriv = self.current / 3600 / self.Cn
        v1_deriv = -self.v1 / self.r1 / self.c1 + self.current / self.c1
        v2_deriv = -self.v2 / self.r2 / self.c2 + self.current / self.c2
        tc_deriv = ( (self.ts - self.tc) / self.Rc + cell_heat ) / self.Cc
        ts_deriv = ( (self.tc - self.ts) / self.Rc + (self.tf - self.ts) / self.Ru) / self.Cs

        self.soc = self.soc + soc_deriv
        self.v1 = self.v1 + v1_deriv
        self.v2 = self.v2 + v2_deriv
        self.tc = self.tc + tc_deriv
        self.ts = self.ts + ts_deriv
        self.ta = (self.tc + self.ts) / 2

        B = self.b_list(self.crate)
        E = 31700 - 370.3 * self.crate
        T = self.tc + 273
        A = ( 20 / B / math.exp(-E / self.ideal_gas_constant / T ) ) ** (1 / self.z)
        N = self.ocv * A / total_Qbatt

        self.soh_deriv = abs( self.power_out ) / 1000 / 2 / N / total_Qbatt
        
        N1 = 3600 * A / self.Cn
        dsoh = abs(self.current/2/N1/self.Cn)

        self.soh = self.soh - dsoh

        next_observation = self.obtain_observation()

        other_variants = {'ta':self.ta,'tc':self.tc,'v1':self.v1,'v2':self.v2,'v1_deriv':v1_deriv,'v2_deriv':v2_deriv,
                          'crate':self.crate,'soh_deriv':self.soh_deriv,'v':self.voltage,
                          'tc_deriv':tc_deriv,'ts_deriv':ts_deriv,'cell_heat':cell_heat,'A':A,'N':N1,
                          'soc_deriv':soc_deriv,'ocv':self.ocv,'r0':self.r0,'soh':self.soh}        
        
        reward_safety_voltage = - 0.7*(self.voltage>3.6) - 0.55*(self.voltage<3.5)*(self.soc<0.93)
        reward_safety_heat = 0.03*(45 - self.ta)*(self.ta>45) - 0.0*(self.ta>45) + 0.03*(self.ta - 10)*(self.ta<10)
        reward_soc = -2 *abs(self.target_soc - self.soc)*(self.soc<=self.target_soc) - 500*(self.soc-self.target_soc)**2*(self.soc>self.target_soc)
        reward_deriv = (abs(self.previous_action - self.current)>0.01)*(-0.02)
        reward_soh = -self.soh_deriv*self.soh_scale
        reward = 0.1 + reward_safety_voltage + reward_safety_heat + reward_soc + reward_deriv + reward_soh

        
        fail = self.soc > self.max_soc or self.soc < self.min_soc or self.voltage < self.min_v_cell or self.voltage > self.max_v_cell
        
        self.count += 1
        
        self.previous_action = self.current if self.count > 0 else 0
        
        finish = self.count > 2000
        
        done =  finish or fail

        return next_observation, reward, done, self.count, other_variants
        

    def obtain_observation(self):
        observations = [self.soc,self._sigmoid(self.tc/50),(self.voltage-2)/4]
        return observations
    
    def _sigmoid(self, x, sigma=2):
        return 2 / (1 + np.exp(-sigma * x) ) - 1

