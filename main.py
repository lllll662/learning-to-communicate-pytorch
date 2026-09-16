import copy, argparse, csv, json, datetime, os
from functools import partial
from pathlib import Path

from utils.dotdic import DotDic
from arena import Arena
from agent import CNetAgent
from switch.switch_game import SwitchGame
from switch.switch_cnet import SwitchCNet

"""
Play communication games
"""

# configure opts for Switch game with 3 DIAL agents
def init_action_and_comm_bits(opt):
    # 判断是否启用通信
    opt.comm_enabled = opt.game_comm_bits > 0 and opt.game_nagents > 1
    
    # 如果没指定，根据是否用DIAL方法来决定
    if opt.model_comm_narrow is None:
        opt.model_comm_narrow = opt.model_dial
    
    # 如果用RIAL（不是DIAL），需要将比特数转成类别数
    # 例如：2 bits → 2^2=4 种消息
    if not opt.model_comm_narrow and opt.game_comm_bits > 0:
        opt.game_comm_bits = 2 ** opt.game_comm_bits
    
    # 计算总的动作空间
    if opt.comm_enabled:
        opt.game_action_space_total = opt.game_action_space + opt.game_comm_bits
    else:
        opt.game_action_space_total = opt.game_action_space
    
    return opt


def init_opt(opt):
    if not opt.model_rnn_layers:
        opt.model_rnn_layers = 2              # 默认RNN有2层
    if opt.model_avg_q is None:
        opt.model_avg_q = True                # 默认计算平均Q值
    if opt.eps_decay is None:
        opt.eps_decay = 1.0                   # 默认epsilon不衰减
    opt = init_action_and_comm_bits(opt)      # 计算通信相关参数
    return opt

def create_game(opt):
	game_name = opt.game.lower()
	if game_name == 'switch':
		return SwitchGame(opt) # 创建游戏实例
	else:
		raise Exception('Unknown game: {}'.format(game_name))

def create_cnet(opt):
	game_name = opt.game.lower()
	if game_name == 'switch':
		return SwitchCNet(opt)
	else:
		raise Exception('Unknown game: {}'.format(game_name))

def create_agents(opt, game):
    agents = [None]  # 1-index，agents[0]不使用
    cnet = create_cnet(opt)         # 创建第一个神经网络
    cnet_target = copy.deepcopy(cnet)  # 创建目标网络（用于稳定Q学习）
    
    for i in range(1, opt.game_nagents + 1):  # i=1,2,...,N
		# 每个agent有两个网络：model（学习用）和target（稳定用）
        agents.append(CNetAgent(opt, game=game, model=cnet, target=cnet_target, index=i))
        
        # 如果不共享参数，各agent有自己的网络
        if not opt.model_know_share:
            cnet = create_cnet(opt)
            cnet_target = copy.deepcopy(cnet)
    
    return agents

def save_episode_and_reward_to_csv(file, writer, e, r):
	writer.writerow({'episode': e, 'reward': r})
	file.flush()

def run_trial(opt, result_path=None, verbose=False):
	# Initialize action and comm bit settings
	opt = init_opt(opt)

	game = create_game(opt)       		# 创建Switch游戏
	agents = create_agents(opt, game)   # 创建N个智能体
	arena = Arena(opt, game)        	# 创建训练竞技场


	test_callback = None
	if result_path:
		os.makedirs(os.path.dirname(result_path), exist_ok=True)  # 创建目录
		result_out = open(result_path, 'w')
		csv_meta = '#' + json.dumps(opt) + '\n'  # 第一行：配置元数据
		result_out.write(csv_meta)
		writer = csv.DictWriter(result_out, fieldnames=['episode', 'reward'])
		writer.writeheader()  # 写列名：episode, reward
		# 创建回调函数，每次测试时调用
		test_callback = partial(save_episode_and_reward_to_csv, result_out, writer)

	arena.train(agents, verbose=verbose, test_callback=test_callback)  # 开始训练

	if result_path:
		result_out.close()

if __name__ == '__main__':
	# 创建一个空白解析器（准备定义参数）
	parser = argparse.ArgumentParser()
	# 往解析器里面添加各个命令行参数（-c、-r、-v这些）
	parser.add_argument('-c', '--config_path', type=str, help='path to existing options file')
	parser.add_argument('-r', '--results_path', type=str, help='path to results directory')
	parser.add_argument('-n', '--ntrials', type=int, default=1, help='number of trials to run')
	parser.add_argument('-s', '--start_index', type=int, default=0, help='starting index for trial output')
	parser.add_argument('-v', '--verbose', action='store_true', help='prints training epoch rewards if set')
	# 解析终端输入，把参数打包到args
	args = parser.parse_args()

	# 打开JSON文件 → 读取配置 → 转成Python字典 → 包装成DotDic对象
	opt = DotDic(json.loads(open(args.config_path, 'r').read()))

	# 构建结果路径
	result_path = None
	if args.results_path:
		result_path = args.config_path and os.path.join(args.results_path, Path(args.config_path).stem) or \
			os.path.join(args.results_path, 'result-', datetime.datetime.now().isoformat())

	# 运行多个Trial循环：每个trial是独立的（随机初始化不同，但配置相同）
	for i in range(args.ntrials):
		trial_result_path = None
		if result_path:
			trial_result_path = result_path + '_' + str(i + args.start_index) + '.csv'
		trial_opt = copy.deepcopy(opt)
		run_trial(trial_opt, result_path=trial_result_path, verbose=args.verbose)

