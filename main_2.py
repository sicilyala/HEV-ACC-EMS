# main program of training and evaluating
import sys
# from runner_2 import Runner
# from runner_2_PER import Runner           # noe memory for all agents， 用td_error_up更新，一次训练更新两次权重
# from runner_2_PER2 import Runner        # one memory for one agent
from common.runner_2_PER_v2 import Runner      # shared memory, 用td_error_up_mean更新，一次训练只更新一次权重
# from runner_2_fix import Runner
from common.arguments import get_args
from common.env_2 import make_env
from common.utils2 import Logger
from common.evaluate import Evaluator
# from data_process import data_process_new, draw_loss

if __name__ == '__main__':
    args = get_args()
    env, args = make_env(args)
    if args.evaluate:
        args.scenario_name += '__'+args.load_dir+'_'+args.load_scenario_name + '_%d'%args.load_episode
        sys.stdout = Logger(filepath=args.eva_dir+"/"+args.scenario_name+"/", filename='evaluate_log.log')
        print('max_episodes: ', args.evaluate_episode)
    else:
        args.scenario_name += args.file_v
        sys.stdout = Logger(filepath=args.save_dir+"/"+args.scenario_name+"/", filename='train_log.log')
        print('max_episodes: ', args.max_episodes)
    print('cycle name: ', args.scenario_name)
    print('episode_steps: ', args.episode_steps)
    print('n_agents: ', args.n_agents)
    print('obs_shape: ', args.obs_shape)
    print('action_shape: ', args.action_shape)
    
    if args.evaluate:
        print("-----Start evaluating!-----")
        evaluator = Evaluator(args, env)
        evaluator.evaluate()
        print("-----Evaluating is finished!-----")
        print('-----Data saved in: <%s>-----' % (args.eva_dir+"/"+args.scenario_name))
    else:
        print("-----Start training-----")
        runner = Runner(args, env)
        runner.run()
        print("-----Training is finished!-----")
        print('-----Data saved in: <%s>-----' % (args.save_dir+"/"+args.scenario_name))
        # draw_loss(args.save_dir, args.scenario_name)
        # data_process_new(args.save_dir, args.scenario_name, args.data_episode,
        #                  save_picture=True, show_max=True)
