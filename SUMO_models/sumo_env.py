import traci
import os
import sumolib
import sys

spdMin = 0
accMin = -2.5
disMin = 0
car_id = {'V1': ['001', '002'],
          'V2': ['003', '004'],
          'V3': ['005', '006']}     # preceding car: 1, 3, 5; host car: 2, 4, 6

def start_sumo():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    if_sumo_gui = True
    if not if_sumo_gui:
        sumoBinary = sumolib.checkBinary('sumo')
    else:
        sumoBinary = sumolib.checkBinary('sumo-gui')
    sumocfg_file = "E:/SEU2/Program1/MADDPG-program/SUMO_models/sumo_cfgs/car_follow_model.sumocfg"
    traci.start([sumoBinary, "-c", sumocfg_file])
    print("start SUMO successful!")

def reset_sumo():
    traci.load(["-c", "E:/SEU2/Program1/MADDPG-program/SUMO_models/sumo_cfgs/car_follow_model.sumocfg"])
    traci.simulationStep(1)

def run_sumo(simu_step, model_sel, spd_p):
    traci.simulationStep(simu_step+1)

    p_id = car_id[model_sel][0]
    h_id = car_id[model_sel][1]
    traci.vehicle.setSpeed(p_id, spd_p)
    p_x = traci.vehicle.getDistance(p_id)   # distance travelled
    h_acc = traci.vehicle.getAcceleration(h_id)
    h_v = traci.vehicle.getSpeed(h_id)
    h_x = traci.vehicle.getDistance(h_id)
    h_acc, h_v, h_x = limitMin(h_acc, h_v, h_x)
    distance = 14 + p_x - h_x  # 5 is car body length,m
    sumo_out = {'spd': h_v, 'acc': h_acc, 'distance': distance,
                'follow_x': h_x, 'leading_x': p_x}
    return sumo_out

   
def limitMin(h_acc, h_v, h_x):
    h_acc = max(h_acc, accMin)
    h_v = max(h_v, spdMin)
    h_x = max(h_x, disMin)
    return h_acc, h_v, h_x

