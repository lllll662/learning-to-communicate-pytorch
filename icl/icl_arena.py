"""
上下文学习（ICL）版 Switch：不训练参数，智能体靠 protocol.md + 各自的 memory.md 完成通信

每个 agent 是独立的 LLM 实例（icl/backend.py）。一次实验由若干"轮"组成，每轮是：
	1. 协议：第 0 轮用 icl_protocol_file（E1，人写的协议），或者赛前商定（E2）；
	   之后每轮如果 icl_reflect=true，就带着上一轮的协议、复盘和经验重新商定（E3）
	2. 比赛：bs 局并行，每一步只有当天进房间的 agent 决策，只能通过 1 bit 灯泡通信
	3. 复盘：统计 + 失败对局的完整记录写进 debrief.md；E3 还会让 1 号总结经验写进 lessons.md

自然语言交流只发生在比赛外（商定 / 复盘），对应"集中学习、分散执行"。
每轮结束后保存 state.json，被打断后用同样的命令重新运行即可从下一轮接着跑（icl_resume=false 则从头开始）。
"""
import csv
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from icl import prompts as P
from icl import protocol_check as PC
from icl.diagnose import analyze
from icl.memory import Workspace
from llm import prompt_builder as pb
from llm.llm_arena import LLMArena, now, fmt_duration
from utils.dotdic import DotDic

METRICS_NAME = 'icl_metrics.csv'
# 每轮的评价指标（icl_metrics.csv）：
#	reward_se：归一化 reward 的标准误（bootstrap）
#	announce_*：宣布次数、宣布的胜率、宣布时灯是 ON 的比例（正确的协议只在灯 ON 时宣布）
#	oracle_acc / acc_R1..R4：和最优协议的一致率（3 人时），E1 中就是执行正确率
#	protocol_*：协议检查——完美执行时的分数、说错的比例、是否用到灯泡、各 agent 理解不一致的比例
METRIC_FIELDS = ['avg_reward', 'reward_se', 'win', 'lose', 'tell_rate', 'parse_fail',
	'announce_n', 'announce_win_rate', 'announce_bulb_on', 'oracle_acc', 'acc_R1', 'acc_R2', 'acc_R3', 'acc_R4',
	'protocol_value', 'protocol_lose', 'protocol_uses_bulb', 'protocol_disagree', 'protocol_compile_fail']


class ICLArena(LLMArena):
	def __init__(self, opt, game):
		super(ICLArena, self).__init__(opt, game)
		self.ws = Workspace(opt.icl_run_dir)
		self.temperature = opt.icl_temperature or 0.0
		self.max_tokens = opt.icl_max_tokens or 512
		self.check_min = opt.icl_check_min if opt.icl_check_min is not None else 0.65	# 不通信也能拿到 0.65
		self.pool = ThreadPoolExecutor(max_workers=opt.game_nagents)

	# ---------------- 调用后端 ----------------

	def query(self, backends, requests, temperature=None, max_tokens=None):
		"""
		requests: {agent_id: [request, ...]} -> {agent_id: [回复, ...]}
		不同 agent 的请求并发发给各自的后端（hf 后端在同一进程里，按顺序跑）
		"""
		t = self.temperature if temperature is None else temperature
		mt = max_tokens or self.max_tokens
		requests = {a: reqs for a, reqs in requests.items() if reqs}
		if self.opt.icl_backend == 'hf':
			return {a: backends[a].chat(reqs, t, mt) for a, reqs in requests.items()}
		futures = {a: self.pool.submit(backends[a].chat, reqs, t, mt) for a, reqs in requests.items()}
		return {a: f.result() for a, f in futures.items()}

	def ask(self, backends, agent_id, messages, phase):
		"""比赛外的单次调用（讨论、整理协议、复盘），温度和长度单独设置"""
		t = self.opt.icl_plan_temperature if self.opt.icl_plan_temperature is not None else 0.7
		req = {'messages': messages, 'meta': {'phase': phase}}
		return self.query(backends, {agent_id: [req]}, t, self.opt.icl_plan_max_tokens or 768)[agent_id][0]

	# ---------------- 比赛 ----------------

	def run_episode(self, backends, protocol):
		opt = self.opt
		game = self.game
		n = opt.game_nagents
		game.reset()
		episode = self.create_episode()
		episode.tell = torch.zeros(opt.bs).int()
		episode.transcripts = [[] for _ in range(opt.bs)]	# (day, agent, 看到的bit, option)，和 LLMArena 一致
		episode.decisions = []		# 写进 decisions.jsonl
		episode.parse_fail = 0

		bulb = torch.zeros(opt.bs, dtype=torch.long)	# 第 1 天灯是 OFF
		# facts[b][k]: 第 b 局 agent k+1 的客观访问记录；notes[b][k]: 它自己写的笔记
		facts = [[[] for _ in range(n)] for _ in range(opt.bs)]
		notes = [['' for _ in range(n)] for _ in range(opt.bs)]

		step = 0
		while step < opt.nsteps and episode.ended.sum() < opt.bs:
			record = DotDic({})
			day = step + 1

			# 1. 为每局还没结束的游戏，给当天进房间的 agent 构造 prompt，按 agent 分组
			pending = {}		# agent_id -> [(b, request), ...]
			for b in range(opt.bs):
				if episode.ended[b]:
					continue
				a = game.active_agent[b][step].item()
				seen = bulb[b].item()
				messages = P.build_decision_messages(opt, a, day, seen, facts[b][a - 1], notes[b][a - 1], protocol)
				meta = {'phase': 'decide', 'day': day, 'bulb': seen, 'facts': list(facts[b][a - 1])}
				pending.setdefault(a, []).append((b, {'messages': messages, 'meta': meta}))

			# 2. 各 agent 的后端并发生成；解析失败的再试一次（提高温度，避免贪心解码给出同样的输出）
			replies = self.query(backends, {a: [q for _, q in lst] for a, lst in pending.items()})
			results = {}		# b -> [agent, request, 原始输出, option, 笔记]
			retry = {}
			for a, lst in pending.items():
				for (b, q), text in zip(lst, replies[a]):
					option, nt = P.parse_decision(text)
					results[b] = [a, q, text, option, nt]
					if option is None:
						retry.setdefault(a, []).append(b)
			if retry:
				again = self.query(backends, {a: [results[b][1] for b in bs_] for a, bs_ in retry.items()},
					temperature=max(self.temperature, 0.7))
				for a, bs_ in retry.items():
					for b, text in zip(bs_, again[a]):
						option, nt = P.parse_decision(text)
						results[b][2] += '\n\n[retry]\n' + text
						results[b][3], results[b][4] = option, nt

			# 3. 选项 → 游戏动作 + 灯泡 bit（不在房间的 agent 默认 NOTHING），同时更新各自的 memory
			a_t = torch.full((opt.bs, n), game.game_actions.NOTHING, dtype=torch.long)
			for b in sorted(results):
				a, q, text, option, nt = results[b]
				parsed = option is not None
				if not parsed:		# 两次都解析失败：按 OFF 处理，笔记保持不变
					episode.parse_fail += 1
					option, nt = pb.OPTION_OFF, notes[b][a - 1]
				seen = bulb[b].item()
				episode.decisions.append({
					'game': b, 'day': day, 'agent': a, 'bulb_seen': pb.bulb_str(seen),
					'action': {v: k for k, v in P.ACTIONS.items()}[option], 'parsed': parsed, 'notes': nt,
					'prompt': q['messages'][1]['content'], 'output': P.prefill_of(q['messages']) + text})
				facts[b][a - 1].append((day, seen, option))
				notes[b][a - 1] = nt
				episode.transcripts[b].append((day, a, seen, option))
				if option == pb.OPTION_TELL:
					a_t[b][a - 1] = game.game_actions.TELL
					episode.tell[b] = 1
				else:
					bulb[b] = option

			# 4. 环境执行一步，累计奖励和结束标记（和 LLMArena 一致）
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
		episode.memories = [[P.render_memory(facts[b][k], notes[b][k], opt.icl_notes is not False) for k in range(n)]
			for b in range(opt.bs)]
		episode.active_agent = game.active_agent.clone()
		return episode

	# ---------------- 赛前商定协议 ----------------

	def plan(self, backends, r, context=None):
		"""
		所有 agent 轮流发言 icl_plan_turns 轮 → 1 号整理成 protocol.md → 其他人审阅 → 有意见时 1 号改一次
		context（E3）：上一轮的协议、复盘和累积的经验
		"""
		opt = self.opt
		n = opt.game_nagents
		turns = opt.icl_plan_turns or 2
		transcript = []
		for t in range(turns):
			for k in range(1, n + 1):	# 按顺序发言，每个人都能看到前面所有人说的话
				msgs = P.build_discussion_messages(opt, k, t, turns, transcript, context)
				transcript.append((k, P.clean_markdown(self.ask(backends, k, msgs, 'discuss'))))

		draft = P.clean_markdown(self.ask(backends, 1, P.build_draft_messages(opt, transcript, context), 'draft'))
		review_reqs = {k: [{'messages': P.build_review_messages(opt, k, draft), 'meta': {'phase': 'review'}}]
			for k in range(2, n + 1)}
		t = opt.icl_plan_temperature if opt.icl_plan_temperature is not None else 0.7
		replies = self.query(backends, review_reqs, t, opt.icl_plan_max_tokens or 768)
		reviews = [(k, P.clean_markdown(replies[k][0])) for k in range(2, n + 1)]
		objections = [(k, text) for k, text in reviews if not P.is_agree(text)]
		final = draft
		if objections:
			msgs = P.build_draft_messages(opt, transcript, context, draft, objections)
			final = P.clean_markdown(self.ask(backends, 1, msgs, 'draft')) or draft

		self.ws.write('discussion.md', '\n\n'.join([
			'# Discussion', P.render_transcript(transcript),
			'# Draft protocol (prisoner 1)', draft,
			'# Reviews', '\n'.join(f'- prisoner {k}: {text}' for k, text in reviews),
			'# Final protocol', final if objections else '(same as the draft: everyone agreed)']), r)
		return final or '(no protocol was agreed)'

	# ---------------- 协议检查 ----------------

	def compile_protocol(self, backends, protocol):
		"""每个 agent 用自己的 LLM 把协议翻译成决策表：{agent_id: {局面: option}}，外加解析失败数"""
		opt = self.opt
		sits = PC.situations(opt.nsteps)
		reqs = {a: [{'messages': P.build_compile_messages(opt, protocol, s, a), 'meta': dict(s, phase='compile')}
			for s in sits] for a in range(1, opt.game_nagents + 1)}
		replies = self.query(backends, reqs, temperature=0.0)
		tables, fails = {}, 0
		for a, texts in replies.items():
			tables[a] = {}
			for s, text in zip(sits, texts):
				option, _ = P.parse_decision(text)
				fails += option is None
				tables[a][PC.key(s)] = pb.OPTION_OFF if option is None else option
		return tables, fails

	def check_protocol(self, backends, protocol, r, tag=''):
		"""编译 + 穷举评估；报告存成 round_xx/protocol_check{tag}.md，返回 (指标, 报告文字)"""
		opt = self.opt
		tables, fails = self.compile_protocol(backends, protocol)
		res = PC.evaluate(tables, opt.game_nagents, opt.nsteps)
		res['compile_fail'] = fails
		text = PC.report(res, tables, opt.nsteps, self.check_min)
		self.ws.write(f'protocol_check{tag}.md', text, r)
		self.ws.write_json(f'protocol_table{tag}.json', PC.table_to_json(tables), r)
		print(f'  protocol check{tag}: value if executed perfectly {res["value"]:.2f}, '
			f'uses bulb {res["uses_bulb"]}, agents disagree on {res["disagree"]:.0%} of situations')
		return res, text

	def checked_protocol(self, backends, protocol, r, planned, context=None):
		"""
		icl_check_protocol = "report"：只检查、记录；"revise"：没通过（完美执行的分数 < icl_check_min）就
		把报告交给 1 号修改，最多 icl_check_retries 次，用分数最高的那一版。人写的协议（E1）只检查不修改
		"""
		mode = self.opt.icl_check_protocol
		if not mode:
			return protocol, None
		res, text = self.check_protocol(backends, protocol, r)
		best = (res['value'], protocol, res)
		if mode == 'revise' and planned:
			for i in range(self.opt.icl_check_retries if self.opt.icl_check_retries is not None else 2):
				if best[0] >= self.check_min:
					break
				fixed = P.clean_markdown(self.ask(backends, 1, P.build_fix_messages(self.opt, protocol, text, context),
					'fix')) or protocol
				self.ws.write(f'protocol_v{i + 1}.md', fixed, r)
				res, text = self.check_protocol(backends, fixed, r, f'_v{i + 1}')
				protocol = fixed
				if res['value'] > best[0]:
					best = (res['value'], fixed, res)
		return best[1], best[2]

	# ---------------- 赛后复盘 ----------------

	def build_debrief(self, episode, m):
		"""统计 + 几局失败对局的完整记录（上帝视角），给下一轮商定协议用"""
		opt = self.opt
		god = episode.game_stats.god_reward
		outcome = {}
		for b in range(opt.bs):
			r = episode.r[b][0].item()
			outcome[b] = 'win' if r > 0 else 'lose' if r < 0 else 'missed' if god[b] > 0 else 'silent'
		count = {k: sum(v == k for v in outcome.values()) for k in ('win', 'lose', 'missed', 'silent')}
		lines = [
			f"Results over {opt.bs} games: normalized reward {m['avg_reward']:.2f}.",
			f"- correct announcement (win): {count['win']}",
			f"- wrong announcement (lose): {count['lose']}",
			f"- everyone had visited by day {opt.nsteps} but nobody announced: {count['missed']}",
			f"- not everyone visited and nobody announced (nothing could be done): {count['silent']}",
		]
		k = opt.icl_debrief_games or 4
		shown = ([b for b in range(opt.bs) if outcome[b] == 'lose'] +
			[b for b in range(opt.bs) if outcome[b] == 'missed'])[:k]
		if len(shown) < k:
			shown += [b for b in range(opt.bs) if outcome[b] == 'win'][:1]
		by_game = {}
		for d in episode.decisions:
			by_game.setdefault(d['game'], []).append(d)
		for b in shown:
			visited = set()
			lines += ['', f"## Game {b}: {self._outcome_text(b, outcome[b], episode)}"]
			for d in by_game.get(b, []):
				visited.add(d['agent'])
				lines.append(f"- day {d['day']}: prisoner {d['agent']} entered (visited so far: "
					f"{sorted(visited)}), saw the bulb {d['bulb_seen']}, chose {d['action']}. "
					f"Notes: \"{d['notes'].strip()}\"")
		return '\n'.join(lines)

	def _outcome_text(self, b, outcome, episode):
		n = self.opt.game_nagents
		if outcome in ('win', 'lose'):
			day, a, _, _ = episode.transcripts[b][-1]
			seen = set(episode.active_agent[b][:day].tolist())
			missing = sorted(set(range(1, n + 1)) - seen)
			return (f"WIN: prisoner {a} announced correctly on day {day}" if outcome == 'win' else
				f"LOSE: prisoner {a} announced on day {day}, but prisoner(s) {missing} had not visited yet")
		return 'MISSED: everyone had visited, but nobody announced'

	# ---------------- 主循环 ----------------

	def train(self, backends, reset=True, verbose=False, test_callback=None):
		opt = self.opt
		ws = self.ws
		rounds = opt.icl_rounds or 1

		state = ws.load_state() if opt.icl_resume is not False else None
		if state:
			start, protocol, lessons, debrief = state['round'] + 1, state['protocol'], state['lessons'], state['debrief']
			test_log, elapsed_before = state['test_log'], state['elapsed']
			metrics_log, check = state.get('metrics', []), state.get('check', {})
			rng = state['rng']
			np.random.set_state((rng['numpy'][0], np.array(rng['numpy'][1], dtype=np.uint32), *rng['numpy'][2:]))
			random.setstate((rng['random'][0], tuple(rng['random'][1]), rng['random'][2]))
			print(f'Resumed from {ws.run_dir}: round {state["round"]} done, elapsed {fmt_duration(elapsed_before)}')
			if test_callback:		# main.py 以 'w' 打开结果 CSV，把之前的结果补写回去
				for e, rew in test_log:
					test_callback(e, rew)
		else:
			start, protocol, lessons, debrief, test_log, elapsed_before = 0, None, '', '', [], 0.0
			metrics_log, check = [], {}
			if opt.icl_seed is not None:
				np.random.seed(opt.icl_seed)
				random.seed(opt.icl_seed)
		log_file, writer = self.open_log(ws.run_dir, start)

		session_start = time.time()
		print(f'[{now()}] start: rounds {start}..{rounds - 1}, backend {opt.icl_backend}, run dir {ws.run_dir}')

		def elapsed():
			return elapsed_before + time.time() - session_start

		for r in range(start, rounds):
			# 1. 本轮的协议：E1 读文件；E2 第 0 轮商定；E3 每轮带着复盘重新商定。新协议先检查（icl_check_protocol）
			context = None
			if protocol is None:
				planned = not opt.icl_protocol_file
				if planned:
					protocol = self.plan(backends, r)
				else:
					with open(opt.icl_protocol_file, encoding='utf-8') as f:
						protocol = f.read()
			elif opt.icl_reflect:
				planned = True
				context = {'protocol': protocol, 'debrief': debrief, 'lessons': lessons}
				protocol = self.plan(backends, r, context)
			else:
				planned = None		# 沿用上一轮的协议和检查结果
			if planned is not None:
				protocol, res = self.checked_protocol(backends, protocol, r, planned, context)
				check = self.check_metrics(res)
			ws.write('protocol.md', protocol, r)

			# 2. 比赛
			episode = self.run_episode(backends, protocol)
			m = self.episode_metrics(episode)
			m['parse_fail'] = episode.parse_fail / max(1, len(episode.decisions))
			m['reward_se'] = self.reward_se(episode)
			a = analyze(episode.decisions, opt.game_nagents)
			m.update({k: v for k, v in a.items() if k in METRIC_FIELDS})
			m.update(check)
			test_log.append([r, m['avg_reward']])
			metrics_log.append(dict({k: m.get(k, '') for k in METRIC_FIELDS}, round=r))
			self.write_metrics(metrics_log)
			writer.writerow({'episode': r, 'phase': 'test', 'avg_reward': m['avg_reward'], 'tell_rate': m['tell_rate'],
				'win': m['win'], 'lose': m['lose'], 'loss': '', 'elapsed_s': round(elapsed(), 1), 'time': now()})
			log_file.flush()
			if test_callback:
				test_callback(r, m['avg_reward'])
			eta = (time.time() - session_start) / (r - start + 1) * (rounds - r - 1)
			print(f'[{now()}] ROUND {r}:', self.episode_stats(m), f'+/- {m["reward_se"]:.3f} (s.e.)',
				f'parse fail: {m["parse_fail"]:.3f} | elapsed {fmt_duration(elapsed())} ETA {fmt_duration(eta)}')
			print(f'  announcements {m["announce_n"]} (win rate {m["announce_win_rate"]:.2f}, '
				f'bulb ON {m["announce_bulb_on"]:.2f})' + (f', oracle agreement {m["oracle_acc"]:.2f} '
				f'(R1 {m["acc_R1"]:.2f} R2 {m["acc_R2"]:.2f} R3 {m["acc_R3"]:.2f} R4 {m["acc_R4"]:.2f})'
				if 'oracle_acc' in m else ''))
			if verbose:
				self.describe_episode(episode)

			# 3. 记录本轮的外部记忆
			ws.write_jsonl('decisions.jsonl', episode.decisions, r)
			for b in range(opt.bs):
				for k in range(opt.game_nagents):
					ws.write_memory(r, b, k + 1, episode.memories[b][k])
			ws.write_json('summary.json', m, r)

			# 4. 复盘（E3 还要总结经验，供下一轮商定协议）
			debrief = self.build_debrief(episode, m)
			ws.write('debrief.md', debrief, r)
			if opt.icl_reflect and r < rounds - 1:
				new = P.clean_markdown(self.ask(backends, 1,
					P.build_lessons_messages(opt, protocol, debrief, lessons), 'lessons'))
				lessons = (lessons + f'\n\n## After round {r}\n' + new).strip()
				ws.write('lessons.md', lessons)

			# 5. 保存状态，被打断后从下一轮接着跑
			np_state = np.random.get_state()
			ws.save_state({
				'round': r, 'protocol': protocol, 'lessons': lessons, 'debrief': debrief,
				'test_log': test_log, 'metrics': metrics_log, 'check': check, 'elapsed': elapsed(),
				'rng': {'numpy': [np_state[0], np_state[1].tolist(), *np_state[2:]], 'random': random.getstate()},
			})

		log_file.close()
		print(f'[{now()}] finished, total time {fmt_duration(elapsed())}')
		rewards = [rew for _, rew in test_log]
		if rewards:
			print('MEAN TEST REWARD over', len(rewards), 'rounds:', sum(rewards) / len(rewards))
		try:
			from icl.plot_icl import plot_icl
			print('curves saved to', plot_icl(ws.run_dir, nagents=opt.game_nagents, nsteps=opt.nsteps))
		except Exception as err:	# 画图失败不影响实验结果，之后可以用 python -m icl.plot_icl 重画
			print('WARNING: failed to plot curves:', err)
		return rewards

	# ---------------- 评价指标 ----------------

	@staticmethod
	def check_metrics(res):
		if res is None:
			return {}
		return {'protocol_value': res['value'], 'protocol_lose': res['lose'], 'protocol_uses_bulb': int(res['uses_bulb']),
			'protocol_disagree': res['disagree'], 'protocol_compile_fail': res['compile_fail']}

	@staticmethod
	def reward_se(episode, n_boot=1000):
		"""归一化 reward 的标准误：对 bs 局有放回重抽样（单独的随机数发生器，不影响游戏的随机数和续跑）"""
		r = episode.r[:, 0].numpy()
		god = episode.game_stats.god_reward.numpy()
		idx = np.random.default_rng(0).integers(0, len(r), (n_boot, len(r)))
		num, den = r[idx].sum(1), god[idx].sum(1)
		return float(np.where(den > 0, num / np.maximum(den, 1), 0).std())

	def write_metrics(self, rows):
		with open(self.ws.path(METRICS_NAME), 'w', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=['round'] + METRIC_FIELDS)
			writer.writeheader()
			writer.writerows(rows)
