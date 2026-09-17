"""
Switch game

This class manages the state of the Switch game among multiple agents.

RIAL Actions:

1 = Nothing
2 = Tell
3 = On
4 = Off
"""

import numpy as np
import torch

from utils.dotdic import DotDic 

class SwitchGame:

	def __init__(self, opt):
		self.game_actions = DotDic({
			'NOTHING': 1,
			'TELL': 2
		})

		self.game_states = DotDic({
			'OUTSIDE': 0,
			'INSIDE': 1,
		})

		self.opt = opt

		# Set game defaults
		opt_game_default = DotDic({
			'game_action_space': 2,
			'game_reward_shift': 0,
			'game_comm_bits': 1,
			'game_comm_sigma': 2
		})
		for k in opt_game_default:
			if k not in self.opt:
				self.opt[k] = opt_game_default[k]

		self.opt.nsteps = 4 * self.opt.game_nagents - 6

		self.reward_all_live = 1
		self.reward_all_die = -1

		self.reset()

	def reset(self):
		# Step count 当前步数
		self.step_count = 0		

		# Rewards 所有game的奖励
		self.reward = torch.zeros(self.opt.bs, self.opt.game_nagents)

		# Who has been in the room? 谁去过房间
		self.has_been = torch.zeros(self.opt.bs, self.opt.nsteps, self.opt.game_nagents)

		# Terminal state 哪些game已结束
		self.terminal = torch.zeros(self.opt.bs, dtype=torch.long)

		# Active agent 每步谁是active的
		self.active_agent = torch.zeros(self.opt.bs, self.opt.nsteps, dtype=torch.long) # 1-indexed agents
		 # 随机分配每一步的active agent
		for b in range(self.opt.bs):
			for step in range(self.opt.nsteps):
				agent_id = 1 + np.random.randint(self.opt.game_nagents)
				self.active_agent[b][step] = agent_id
				self.has_been[b][step][agent_id - 1] = 1

		return self

	def get_action_range(self, step, agent_id):
		"""
		Return 1-indexed indices into Q vector for valid actions and communications (so 0 represents no-op)

		返回这个agent在这一步可以执行哪些动作的索引

		例如：
			快速测试配置中：
			game_action_space = 2
			game_comm_bits = 1
			game_action_space_total = 2 + 1 = 3

			agent1是active的：
			action_range = [1, 2]         → 可以选择动作1或2
			comm_range = [3, 3]           → 可以选择通信bit（或不通信）

			agent2不是active的：
			action_range = [1, 1]         → 只能选择1（NOTHING）
			comm_range = []               → 没有通信权
		"""
		opt = self.opt
		action_dtype = torch.long
		action_range = torch.zeros((self.opt.bs, 2), dtype=action_dtype)
		comm_range = torch.zeros((self.opt.bs, 2), dtype=action_dtype)
		for b in range(self.opt.bs): 
			if self.active_agent[b][step] == agent_id:
				# 这个agent是active的，可以执行所有动作+通信
				action_range[b] = torch.tensor([1, opt.game_action_space], dtype=action_dtype)
				comm_range[b] = torch.tensor(
					[opt.game_action_space + 1, opt.game_action_space_total], dtype=action_dtype)
			else:
				# 这个agent不是active的，只能选择"什么都不做"
				action_range[b] = torch.tensor([1, 1], dtype=action_dtype)

		return action_range, comm_range

	def get_comm_limited(self, step, agent_id):
		"""
		如果game_comm_limited=true，agent只能接收前一步的active agent的消息

		例如：
			Step 0: Agent2是active → 无人可接收消息
			Step 1: Agent1是active → Agent1可接收Agent2的消息
			Step 2: Agent3是active → Agent3可接收Agent1的消息
			···
		"""
		if self.opt.game_comm_limited:
			comm_lim = torch.zeros(self.opt.bs, dtype=torch.long)
			for b in range(self.opt.bs):
				if step > 0 and agent_id == self.active_agent[b][step]:
					# 这个agent是当前step的active
                	# 它只能接收前一步agent的消息
					comm_lim[b] = self.active_agent[b][step - 1]
			return comm_lim
		return None

	def get_reward(self, a_t):
		"""
		a_t: 所有agent在当前步的动作，形状[bs, nagents]

		每一步：
			1. 获取当前active agent
			2. 检查它的动作
			3. 如果是TELL：
				- 检查所有agent是否都去过
				- YES → reward = +1 ✅
				- NO → reward = -1 ❌
			4. 游戏结束（terminal=1）
			5. 或达到最大步数，游戏结束
		"""

		# Return reward for action a_t by active agent
		for b in range(self.opt.bs):
			# 谁是active的
			active_agent_idx = self.active_agent[b][self.step_count].item() - 1

			# 如果active agent执行了TELL动作，且当前步不是终端状态
			if a_t[b][active_agent_idx].item() == self.game_actions.TELL and not self.terminal[b].item():
				# 检查是否所有agent都去过房间
				has_been = self.has_been[b][:self.step_count + 1].sum(0).gt(0).sum(0).item()
				if has_been == self.opt.game_nagents:
					self.reward[b] = self.reward_all_live
				else:
					self.reward[b] = self.reward_all_die
				self.terminal[b] = 1	 # 游戏结束

			# 如果达到最大步数但还没TELL
			elif self.step_count == self.opt.nsteps - 1 and not self.terminal[b]:
				self.terminal[b] = 1	 # 游戏结束
				# reward保持为0（没有显式设置）

		return self.reward.clone(), self.terminal.clone()

	def step(self, a_t):
		"""
		执行一步
		"""
		reward, terminal = self.get_reward(a_t)
		self.step_count += 1

		return reward, terminal

	def get_state(self):
		"""
		返回当前状态：每个agent是否在房间里

		例如：
		state[b] = [0, 1, 0]  表示在batch b中：
		- agent1不在房间里（0）
		- agent2在房间里（1）
		- agent3不在房间里（0）

		这是agents可以观察到的部分可观察信息
		"""
		state = torch.zeros(self.opt.bs, self.opt.game_nagents, dtype=torch.long)

		# Get the state of the game
		for b in range(self.opt.bs):
			for a in range(1, self.opt.game_nagents + 1):
				if self.active_agent[b][self.step_count] == a:
					state[b][a - 1] = self.game_states.INSIDE

		return state	# 形状[bs, nagents]，值为0或1

	def god_strategy_reward(self, steps):
		"""
		最优策略奖励：
		如果采用最优策略（所有agent都去过一次），能获得多少奖励
		用来归一化训练奖励
		"""
		reward = torch.zeros(self.opt.bs)
		for b in range(self.opt.bs):
			has_been = self.has_been[b][:self.opt.nsteps].sum(0).gt(0).sum().item()
			if has_been == self.opt.game_nagents:
				reward[b] = self.reward_all_live

		return reward

	def naive_strategy_reward(self):
		pass

	def get_stats(self, steps):
		stats = DotDic({})
		stats.god_reward = self.god_strategy_reward(steps)
		return stats

	def describe_game(self, b=0):
		print('has been:', self.has_been[b])
		print('num has been:', self.has_been[b].sum(0).gt(0).sum().item())
		print('active agents: ', self.active_agent[b])
		print('reward:', self.reward[b])

