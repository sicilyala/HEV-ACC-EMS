from utils2 import get_driving_cycle, get_acc_limit
import matplotlib.pyplot as plt
from arguments import get_args

if __name__ == '__main__':
    args = get_args()
    data_dir = args.save_dir+"/"+args.scenario_name+"/"
    data_name = 'acceleration of leading car'
    speed_list = get_driving_cycle(args.scenario_name)
    acc_list, acc_max, acc_min = get_acc_limit(speed_list, True)
    print('maximal acceleration: ', acc_max)
    print('minimal acceleration: ', acc_min)
    plt.plot(acc_list)
    plt.xlabel('time step')
    plt.ylabel(data_name)
    plt.title('%s' % data_name)
    # plt.savefig(data_dir+data_name+'.jpg', format='jpg')
    print(len(speed_list))      # 241
    print(len(acc_list))        # 241