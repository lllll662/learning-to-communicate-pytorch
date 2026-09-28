"""
用 LLM 代替 DQN 跑 Switch 游戏

和原 Arena 的区别：
	- 没有 RNN 隐状态 / target 网络 / ε-greedy，agent 的记忆由每局的 history 显式记录并写进 prompt
	- 不在房间的 agent 只能选 NOTHING，所以每一步只需要为当前 active 的 agent 调用一次 LLM，
	  bs 局游戏的 prompt 拼成一个 batch 一起前向
	- 目前只做推理（演示阶段），train() 里每个 episode 都是评估，不更新 LLM 参数
"""
import torch

from arena import Arena
from llm import prompt_builder as pb
from utils.dotdic import DotDic


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
			for b, a, option in zip(live, active, options.tolist()):
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

	def train(self, policy, reset=True, verbose=False, test_callback=None):
		opt = self.opt
		rewards = []
		for e in range(opt.nepisodes):
			episode = self.run_episode(policy, train_mode=False)
			norm_r = self.average_reward(episode)
			rewards.append(norm_r)
			if test_callback:
				test_callback(e, norm_r)

			team_r = episode.r[:, 0]
			print('TEST EPOCH:', e, 'avg steps:', episode.steps.float().mean().item(), 'avg reward:', norm_r,
				'| tell rate:', episode.tell.float().mean().item(),
				'win:', (team_r > 0).float().mean().item(),
				'lose:', (team_r < 0).float().mean().item())
			if verbose:
				self.describe_episode(episode)

		print('MEAN REWARD over', opt.nepisodes, 'episodes:', sum(rewards) / len(rewards))
		return rewards
