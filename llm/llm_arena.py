"""
用 LLM 代替 DQN 跑 Switch 游戏

和原 Arena 的区别：
	- 没有 RNN 隐状态 / target 网络 / ε-greedy，agent 的记忆由每局的 history 显式记录并写进 prompt
	- 不在房间的 agent 只能选 NOTHING，所以每一步只需要为当前 active 的 agent 调用一次 LLM，
	  bs 局游戏的 prompt 拼成一个 batch 一起前向
	- llm_train=false：train() 里每个 episode 都是贪心评估，不更新 LLM 参数
	- llm_train=true：每个 episode 先采样一批游戏、用 REINFORCE 更新 LoRA，每 step_test 轮贪心评估一次

断点续训（llm_train=true 且设置了 llm_save_path 时）：
	- 每 llm_ckpt_every 个 episode（默认 = step_test）把 LoRA、优化器、随机数状态、进度、累计用时
	  存到 llm_save_path/checkpoint.pt
	- 启动时如果 checkpoint.pt 已存在，自动从下一个 episode 接着训（llm_resume=false 则从头开始）
	- 每个 episode 的训练/测试指标和时间记录在 llm_save_path/train_log.csv
"""
import csv
import datetime
import os
import random
import time

import numpy as np
import torch

from arena import Arena
from llm import prompt_builder as pb
from utils.dotdic import DotDic

CKPT_NAME = 'checkpoint.pt'
LOG_NAME = 'train_log.csv'
LOG_FIELDS = ['episode', 'phase', 'avg_reward', 'tell_rate', 'win', 'lose', 'loss', 'elapsed_s', 'time']
# 续训时这些配置项允许和 checkpoint 不同（比如把 nepisodes 改大接着训）
RESUME_FREE_KEYS = {'nepisodes', 'step_test', 'llm_ckpt_every', 'llm_resume'}


def now():
	return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def fmt_duration(seconds):
	s = int(seconds)
	return f'{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}'


class LLMArena(Arena):
	def __init__(self, opt, game):
		super(LLMArena, self).__init__(opt, game)
		# 灯泡语义：进门看到的是前一天 active agent 留下的 bit，对应 game_comm_limited=true
		assert opt.game_comm_limited, 'LLM agent assumes game_comm_limited=true'

	def run_episode(self, policy, train_mode=False):
		opt = self.opt
		game = self.game
		game.reset()
		episode = self.create_episode()
		episode.tell = torch.zeros(opt.bs).int()	# 这局是否有人 TELL
		episode.transcripts = [[] for _ in range(opt.bs)]
		episode.decisions = []		# 训练用：每个 LLM 决策 (b, messages, option)

		bulb = torch.zeros(opt.bs, dtype=torch.long)	# 第 1 天灯是 OFF
		# history[b][agent_idx]: 这个 agent 在第 b 局的访问记录 [(day, 看到的bit, option), ...]
		history = [[[] for _ in range(opt.game_nagents)] for _ in range(opt.bs)]

		step = 0
		while step < opt.nsteps and episode.ended.sum() < opt.bs:
			# 原 create_step_record 里有 RNN 隐状态和 Q 值，LLM 用不上，只记录动作和奖励
			record = DotDic({})

			# 1. 为每局还没结束的游戏，给当天进房间的 agent 构造 prompt
			live = [b for b in range(opt.bs) if not episode.ended[b]]
			active = [game.active_agent[b][step].item() for b in live]	# 1-indexed
			messages = [
				pb.build_messages(opt, a, step + 1, bulb[b].item(), history[b][a - 1])
				for b, a in zip(live, active)]

			# 2. LLM 一次前向给出所有局的选项
			logits = policy.option_logits(messages)
			options = policy.select(logits, sample=train_mode)

			# 3. 选项 → 游戏动作 + 灯泡 bit（不在房间的 agent 默认 NOTHING）
			a_t = torch.full((opt.bs, opt.game_nagents), game.game_actions.NOTHING, dtype=torch.long)
			for b, a, m, option in zip(live, active, messages, options.tolist()):
				episode.decisions.append((b, m, option))
				history[b][a - 1].append((step + 1, bulb[b].item(), option))
				episode.transcripts[b].append((step + 1, a, bulb[b].item(), option))
				if option == pb.OPTION_TELL:
					a_t[b][a - 1] = game.game_actions.TELL
					episode.tell[b] = 1
				else:
					bulb[b] = option

			# 4. 环境执行一步，累计奖励和结束标记（和原 Arena 一致）
			record.a_t = a_t
			record.r_t, record.terminal = game.step(a_t)
			for b in range(opt.bs):
				if not episode.ended[b]:
					episode.steps[b] = episode.steps[b] + 1
					episode.r[b] = episode.r[b] + record.r_t[b]
				if record.terminal[b]:
					episode.ended[b] = 1
			episode.step_records.append(record)
			step = step + 1

		episode.game_stats = game.get_stats(episode.steps)
		return episode

	def describe_episode(self, episode, b=0):
		print(f'  game {b}: visitors', [a for _, a, _, _ in episode.transcripts[b]],
			'reward', episode.r[b][0].item())
		for day, a, seen, option in episode.transcripts[b]:
			print(f'    day {day}: prisoner {a} saw bulb {pb.bulb_str(seen)} -> {pb.describe_option(option)}')

	def episode_metrics(self, episode):
		team_r = episode.r[:, 0]
		return {
			'avg_steps': episode.steps.float().mean().item(),
			'avg_reward': self.average_reward(episode),
			'tell_rate': episode.tell.float().mean().item(),
			'win': (team_r > 0).float().mean().item(),
			'lose': (team_r < 0).float().mean().item(),
		}

	def episode_stats(self, m):
		return (f'avg steps: {m["avg_steps"]:.2f} avg reward: {m["avg_reward"]:.3f} '
			f'| tell rate: {m["tell_rate"]:.3f} win: {m["win"]:.3f} lose: {m["lose"]:.3f}')

	# ---------------- 断点续训 ----------------

	def save_checkpoint(self, policy, save_dir, e, test_log, elapsed):
		state = policy.state_dict()
		state.update({
			'episode': e,				# 已完成的最后一个 episode
			'test_log': test_log,		# [(episode, 归一化奖励), ...]
			'elapsed': elapsed,			# 累计训练用时（秒），不含中断的时间
			'opt': dict(self.opt),
			'rng': {
				'torch': torch.get_rng_state(),
				'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
				'numpy': np.random.get_state(),		# SwitchGame 用 numpy 抽每天进房间的人
				'random': random.getstate(),
			},
		})
		# 先写临时文件再替换：保存到一半被打断，也不会损坏上一个 checkpoint
		path = os.path.join(save_dir, CKPT_NAME)
		torch.save(state, path + '.tmp')
		os.replace(path + '.tmp', path)

	def load_checkpoint(self, policy, save_dir):
		path = os.path.join(save_dir, CKPT_NAME)
		if not os.path.exists(path):
			return None
		if self.opt.llm_resume is False:
			print(f'llm_resume=false: ignoring {path}, training from scratch (it will be overwritten)')
			return None
		state = torch.load(path, map_location=policy.device, weights_only=False)
		changed = sorted(k for k in set(state['opt']) | set(self.opt)
			if k not in RESUME_FREE_KEYS and state['opt'].get(k) != self.opt.get(k))
		if changed:
			print('WARNING: config differs from checkpoint in', changed)
		policy.load_state_dict(state)
		rng = state['rng']
		torch.set_rng_state(rng['torch'])
		if rng['cuda'] is not None and torch.cuda.is_available() and len(rng['cuda']) == torch.cuda.device_count():
			torch.cuda.set_rng_state_all(rng['cuda'])
		np.random.set_state(rng['numpy'])
		random.setstate(rng['random'])
		print(f'Resumed from {path}: episode {state["episode"]} done, '
			f'elapsed {fmt_duration(state["elapsed"])}')
		return state

	def open_log(self, save_dir, start_e):
		"""打开 train_log.csv；续训时丢掉 checkpoint 之后的行（那部分会重新训练）"""
		path = os.path.join(save_dir, LOG_NAME)
		rows = []
		if start_e > 0 and os.path.exists(path):
			with open(path, newline='') as f:
				rows = [r for r in csv.DictReader(f) if int(r['episode']) < start_e]
		log_file = open(path, 'w', newline='')
		writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
		writer.writeheader()
		writer.writerows(rows)
		log_file.flush()
		return log_file, writer

	# ---------------- 训练主循环 ----------------

	def train(self, policy, reset=True, verbose=False, test_callback=None):
		opt = self.opt
		save_dir = opt.llm_save_path if opt.llm_train else None
		ckpt_every = opt.llm_ckpt_every or opt.step_test

		start_e, test_log, elapsed_before = 0, [], 0.0
		log_file = writer = None
		if save_dir:
			os.makedirs(save_dir, exist_ok=True)
			state = self.load_checkpoint(policy, save_dir)
			if state:
				start_e, test_log, elapsed_before = state['episode'] + 1, state['test_log'], state['elapsed']
				# main.py 以 'w' 打开结果 CSV，把 checkpoint 之前的测试结果补写回去
				if test_callback:
					for e, r in test_log:
						test_callback(e, r)
			log_file, writer = self.open_log(save_dir, start_e)

		session_start = time.time()
		print(f'[{now()}] start: episodes {start_e}..{opt.nepisodes - 1}')

		def progress(e):
			"""累计用时 + 按本次会话的平均速度估计剩余时间"""
			spent = time.time() - session_start
			eta = spent / (e - start_e + 1) * (opt.nepisodes - e - 1)
			return f'elapsed {fmt_duration(elapsed_before + spent)} ETA {fmt_duration(eta)}'

		def log(e, phase, m, loss=None):
			if writer:
				writer.writerow({'episode': e, 'phase': phase, 'avg_reward': m['avg_reward'],
					'tell_rate': m['tell_rate'], 'win': m['win'], 'lose': m['lose'],
					'loss': '' if loss is None else loss,
					'elapsed_s': round(elapsed_before + time.time() - session_start, 1), 'time': now()})
				log_file.flush()

		for e in range(start_e, opt.nepisodes):
			# 1. 训练：按 π_mix 采样一批游戏，REINFORCE 更新 LoRA
			if opt.llm_train:
				episode = self.run_episode(policy, train_mode=True)
				loss = policy.learn_from_episode(episode)
				m = self.episode_metrics(episode)
				log(e, 'train', m, loss)
				if verbose:
					print('train epoch:', e, self.episode_stats(m), f'loss: {loss:.4f} | {progress(e)}')

			# 2. 测试：贪心策略评估（不训练时每个 episode 都评估）
			if not opt.llm_train or e % opt.step_test == 0:
				episode = self.run_episode(policy, train_mode=False)
				m = self.episode_metrics(episode)
				test_log.append((e, m['avg_reward']))
				log(e, 'test', m)
				if test_callback:
					test_callback(e, m['avg_reward'])
				print(f'[{now()}] TEST EPOCH: {e}', self.episode_stats(m), f'| {progress(e)}')
				if verbose:
					self.describe_episode(episode)

			# 3. 定期保存 checkpoint，最后一个 episode 一定保存
			if save_dir and ((e + 1) % ckpt_every == 0 or e == opt.nepisodes - 1):
				self.save_checkpoint(policy, save_dir, e, test_log,
					elapsed_before + time.time() - session_start)

		if log_file:
			log_file.close()
		total = elapsed_before + time.time() - session_start
		print(f'[{now()}] finished, total training time {fmt_duration(total)}')
		rewards = [r for _, r in test_log]
		if rewards:
			print('MEAN TEST REWARD over', len(rewards), 'evaluations:', sum(rewards) / len(rewards))
		if save_dir:
			policy.save(save_dir)
			print('LoRA adapter saved to', save_dir)
			try:
				from llm.plot_training import plot_run
				print('training curves saved to', plot_run(save_dir, nagents=opt.game_nagents, nsteps=opt.nsteps))
			except Exception as err:	# 画图失败不影响训练结果，之后可以用 python -m llm.plot_training 重画
				print('WARNING: failed to plot training curves:', err)
		return rewards
