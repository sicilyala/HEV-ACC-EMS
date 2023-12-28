import argparse

"""
Here are the parameters for training
dataset use:
    debug: Standard_IM240, 241s
    train: WVU_mix, 4713s; mix2, 4619s; mix3, 2431s;
    validation: Standard_ChinaCity (CTUDC), 1314s; CLTC_P, 1800s; mix_valid; Standard_HWFET
"""

def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--max-episodes", type=int, default=500, help="number of episodes ")
    parser.add_argument("--episode-steps", type=int, default=None, help="number of time steps in a sngle episode")
    # Core training parameters
    parser.add_argument("--lr-actor-0", type=float, default=1e-3, help="learning rate of actor")
    parser.add_argument("--lr-critic-0", type=float, default=1e-2, help="learning rate of critic")
    parser.add_argument("--lr-actor-1", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic-1", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--init_noise", type=float, default=0.25, help="initial noise rate for sampling from a standard normal distribution ")
    # parser.add_argument("--init_noise_1", type=float, default=0.25)
    parser.add_argument("--noise_discount_rate", type=float, default=0.999)
    parser.add_argument("--gamma", type=float, default=0.975, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(1e5),
                        help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=64, help="number of episodes to optimize at the same time")
    # parser.add_argument("--miu", type=float, default=0.8, help="running mean and std")
    # save model under training     # Standard_ChinaCity  Standard_IM240 Standard_WVUCITY(1408)
    parser.add_argument("--save_dir", type=str, default="./model12",
                        help="directory in which saves training data and model")
    parser.add_argument("--scenario_name", type=str, default="mix2", help="name of driving cycle data")
    parser.add_argument("--file_v", type=str, default='_v1_RIS', help="每次训练都须重新指定")
    # load learned model to train new model or evaluate
    parser.add_argument("--load_or_not", type=bool, default=False, help="load learned model to train new model")
    parser.add_argument("--load_episode", type=int, default=458)
    parser.add_argument("--load_scenario_name", type=str, default="mix2_v12_PER_v4")
    parser.add_argument("--load_dir", type=str, default="./model10")
    # evaluate
    parser.add_argument("--evaluate", type=bool, default=False)
    parser.add_argument("--evaluate_episode", type=int, default=1)
    parser.add_argument("--eva_dir", type=str, default="./evaluate",
                        help="directory in which saves evaluation result")
    # data process
    # parser.add_argument("--data_episode", type=int, default=497)
    # parser.add_argument("--data_scenario", type=str, default="Standard_WVUCITY", help="name of driving cycle data")
    # parser.add_argument("--data_dir", type=str, default="./model8", help="load data to process")
    # device
    # parser.add_argument("--cuda", type=bool, default=True, help="True for GPU, False for CPU")
    # all above
    args = parser.parse_args()
    return args
