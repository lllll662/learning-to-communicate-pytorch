"""
Create agents for communication games
"""
import copy

import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from utils.dotdic import DotDic
from modules.dru import DRU

class CNetAgent:
	def __init__(self, opt, game, model, target, index):
		self.opt = opt
		self.game = game
		self.model = model				# 学习网络（会被更新）
		self.model_target = target		# 目标网络（用于计算目标值，稳定学习）

		# 冻结target网络的梯度（不需要更新）
		for p in self.model_target.parameters():
			p.requires_grad = False

		self.episodes_seen = 0
		self.dru = DRU(opt.game_comm_sigma, opt.model_comm_narrow, opt.game_comm_hard)		# 通信的离散化单元
		self.id = index
		self.optimizer = optim.RMSprop(
			params=model.get_params(), lr=opt.learningrate, momentum=opt.momentum)			# 优化器

	def reset(self):
		self.model.reset_parameters()				# 重置学习网络的参数
		self.model_target.reset_parameters()		# 重置目标网络的参数
		self.episodes_seen = 0

	def _eps_flip(self, eps):
		"""
		生成一个长度等于`batch_size(bs)`的布尔数组，每个位置为 True 的概率是 eps，
		用来做 ε-greedy：判断这个 batch 里每一局，要不要随机探索（选随机动作）
		"""
		# Sample Bernoulli with P(True) = eps
		return np.random.rand(self.opt.bs) < eps

	def _random_choice(self, items):
		"""
		从 items 中随机选择一个元素
		"""
		return torch.from_numpy(np.random.choice(items, 1)).item()

	def select_action_and_comm(self, step, q, eps=0, target=False, train_mode=False):
		"""
		q: 神经网络输出的Q值，形状[bs, game_action_space_total]
    	eps: 探索率（0=贪心，1=完全随机）
		target: 是否使用目标网络计算目标值
		train_mode: 是否在训练模式下
		"""
		# eps-Greedy action selector
		if not train_mode:
			eps = 0
		opt = self.opt
		# 从Game得到这个agent这一步可以做哪些动作
		action_range, comm_range = self.game.get_action_range(step, self.id)
		action = torch.zeros(opt.bs, dtype=torch.long)
		action_value = torch.zeros(opt.bs)
		comm_action = torch.zeros(opt.bs).int()
		comm_vector = torch.zeros(opt.bs, opt.game_comm_bits)
		comm_value = None
		if not opt.model_dial:
			comm_value = torch.zeros(opt.bs)

		should_select_random_comm = None
		# 返回[batch_size]的布尔值，True表示这个游戏需要探索动作
		should_select_random_a = self._eps_flip(eps)
		if not opt.model_dial:
			# 返回[batch_size]的布尔值，True表示这个游戏需要探索通信
			should_select_random_comm = self._eps_flip(eps)

		# Get action + comm
		# 对每个并行游戏，选择动作和通信
		for b in range(opt.bs):
			q_a_range = range(0, opt.game_action_space)
			# 当前 agent 允许选的动作区间
			a_range = range(action_range[b, 0].item() - 1, action_range[b, 1].item())
			if should_select_random_a[b]:
				# 随机选择一个动作
				action[b] = self._random_choice(a_range)
				action_value[b] = q[b, action[b]]
			else:
				# 贪心：选择Q值最高的动作
				action_value[b], action[b] = q[b, a_range].max(0)
			action[b] = action[b] + 1

			q_c_range = range(opt.game_action_space, opt.game_action_space_total)
			if comm_range[b, 1].item() > 0:
				# 当前 agent 允许选的通信区间
				c_range = range(comm_range[b, 0].item() - 1, comm_range[b, 1].item())
				if not opt.model_dial:
					if should_select_random_comm[b]:
						comm_action[b] = self._random_choice(c_range)
						comm_value[b] = q[b, comm_action[b]]
						comm_action[b] = comm_action[b] - opt.game_action_space
					else:
						comm_value[b], comm_action[b] = q[b, c_range].max(0)
					# 转换成one-hot向量
					comm_vector[b][comm_action[b]] = 1
					comm_action[b] = comm_action[b] + 1
				else:	# DIAL
					# 通信通过DRU处理Q值，生成连续向量
					comm_vector[b] = self.dru.forward(q[b, q_c_range], train_mode=train_mode) # apply DRU
			elif (not opt.model_dial) and opt.model_avg_q and target:
				comm_value[b], _ = q[b, q_a_range].max(0)

		"""
		action: 选择的游戏动作 [bs]
		action_value: 这个动作的Q值 [bs]
		comm_vector: 发送的通信消息 [bs, comm_bits]
		comm_action: 通信动作索引（RIAL用）
		comm_value: 通信的Q值（RIAL用）
		"""
		return (action, action_value), (comm_vector, comm_action, comm_value)

	def episode_loss(self, episode):
		"""
		计算整个episode的总损失
		"""
		opt = self.opt
		total_loss = torch.zeros(opt.bs)

		for b in range(opt.bs):
			b_steps = episode.steps[b].item()
			
			# 对这个游戏的每一步
			for step in range(b_steps):
				record = episode.step_records[step]

				# 对每个agent
				for i in range(opt.game_nagents):
					 # 计算TD误差（Temporal Difference Error）
					td_action = 0
					td_comm = 0
					r_t = record.r_t[b][i]          # 当前时刻，智能体i拿到的即时奖励
					q_a_t = record.q_a_t[b][i]      # 当前网络预测：动作a对应的Q值
					q_comm_t = 0                    # 通信动作的Q，初始化为0

					# 普通动作的 TD 误差 td_action
					if record.a_t[b][i].item() > 0:
						# a_t>0 代表这个agent执行了有效动作（不是空动作）
						if record.terminal[b].item() > 0:
							# terminal=True，游戏结束，没有下一步
							td_action = r_t - q_a_t
						else:
							# 游戏未结束，拿下一步的最大Q
							next_record = episode.step_records[step + 1]
							q_next_max = next_record.q_a_max_t[b][i]

							# model_avg_q开启：动作Q和通信Q取平均作为目标值
							if not opt.model_dial and opt.model_avg_q:
								q_next_max = (q_next_max + next_record.q_comm_max_t[b][i])/2.0
							
							# TD公式：目标Q = r + γ * maxQ_next
							td_action = r_t + opt.gamma * q_next_max - q_a_t

					# 通信动作 TD 误差 td_comm（只有非 dial 模型才走这里）
					if not opt.model_dial and record.a_comm_t[b][i].item() > 0:
						q_comm_t = record.q_comm_t[b][i]
						if record.terminal[b].item() > 0:
							td_comm = r_t - q_comm_t
						else:
							next_record = episode.step_records[step + 1]
							q_next_max = next_record.q_comm_max_t[b][i]
							if opt.model_avg_q: 
								q_next_max = (q_next_max + next_record.q_a_max_t[b][i])/2.0
							td_comm = r_t + opt.gamma * q_next_max - q_comm_t

					if not opt.model_dial:
						loss_t = (td_action ** 2 + td_comm ** 2)
					else:
						loss_t = (td_action ** 2)
					total_loss[b] = total_loss[b] + loss_t

		loss = total_loss.sum()
		# 所有 batch 所有智能体 loss 求和，求平均，得到整局的总 loss
		loss = loss/(self.opt.bs * self.opt.game_nagents)
		return loss

	def learn_from_episode(self, episode):
		# Step 1: 清零梯度
		self.optimizer.zero_grad()
		# Step 2: 计算损失
		loss = self.episode_loss(episode)
		# Step 3: 反向传播更新参数(retain_graph：如果agents共享参数，保留计算图)
		loss.backward(retain_graph=not self.opt.model_know_share)
		# Step 4: 梯度裁剪（防止梯度爆炸）
		clip_grad_norm_(parameters=self.model.get_params(), max_norm=10)
		# Step 5: 更新权重
		self.optimizer.step()

		# Step 6: 定期更新目标网络
		self.episodes_seen = self.episodes_seen + 1
		if self.episodes_seen % self.opt.step_target == 0:
			# 每step_target个episode，把当前网络复制到目标网络
			self.model_target.load_state_dict(self.model.state_dict())


