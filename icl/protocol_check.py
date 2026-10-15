"""
协议检查：把自然语言协议"编译"成决策表，再穷举所有来访顺序，算出"所有人都完美执行时这份协议能得多少分"

	局面 = (第几天, 看到的灯, 是不是第一次来, 昨天是不是自己)
	每个 agent 都用自己的 LLM 把协议翻译成一张表（局面 → 动作），所以还能看出大家对协议的理解是否一致

这样可以把"协议设计得好不好"（表的分数）和"执行得好不好"（实际比赛的分数）分开。
局限：只覆盖依赖这 4 个特征的协议；依赖"笔记里记的计数"等更复杂状态的协议，会被压缩成这 4 个特征上的近似。
"""
import itertools

import numpy as np

from llm.prompt_builder import OPTION_OFF, OPTION_TELL, bulb_str
from icl.prompts import ACTIONS

ACTION_NAME = {v: k for k, v in ACTIONS.items()}
MAX_ENUM = 200000		# n^T 超过这个数时改用蒙特卡洛抽样


def situations(T):
	"""所有可能出现的局面（去掉不可能的组合，比如第 1 天灯是 ON、第 2 天不是第一次来但昨天不在）"""
	out = [{'day': 1, 'bulb': 0, 'first': True, 'yesterday': False}]
	for day in range(2, T + 1):
		for bulb in (0, 1):
			for first, yesterday in ((True, False), (False, True), (False, False)):
				if day == 2 and not first and not yesterday:	# 第 2 天来过就一定是昨天（第 1 天）来的
					continue
				out.append({'day': day, 'bulb': bulb, 'first': first, 'yesterday': yesterday})
	return out


def key(s):
	return (s['day'], s['bulb'], s['first'], s['yesterday'])


def describe(k):
	day, bulb, first, yesterday = k
	return (f"day {day}, bulb {bulb_str(bulb)}, {'first visit' if first else 'not first visit'}, "
		f"{'was' if yesterday else 'was not'} here yesterday")


def situation_facts(s):
	"""给 rule 后端 / oracle 用：构造一段和局面一致的访问记录 [(day, bit, option), ...]"""
	if s['first']:
		return []
	return [(s['day'] - 1 if s['yesterday'] else s['day'] - 2, 0, OPTION_OFF)]


def oracle_table(T):
	from icl.backend import oracle_option
	return {key(s): oracle_option(s['day'], s['bulb'], situation_facts(s)) for s in situations(T)}


def evaluate(tables, n, T, seed=0):
	"""
	tables: {agent_id: {局面: option}}，每个 agent 按自己的表行动
	返回完美执行时的指标，以及每个"会宣布"的局面里说错的概率
	"""
	if n ** T <= MAX_ENUM:
		seqs = itertools.product(range(1, n + 1), repeat=T)
		total = n ** T
	else:
		rng = np.random.default_rng(seed)
		total = 50000
		seqs = (tuple(rng.integers(1, n + 1, T)) for _ in range(total))
	win = lose = god = 0
	reach = {}			# 局面 -> [宣布次数, 其中说错的次数]
	for seq in seqs:
		god += len(set(seq)) == n
		bulb, seen = OPTION_OFF, set()
		for t, a in enumerate(seq):
			k = (t + 1, bulb, a not in seen, t > 0 and seq[t - 1] == a)
			seen.add(a)
			option = tables[a].get(k, OPTION_OFF)
			if option == OPTION_TELL:
				ok = len(seen) == n
				win += ok
				lose += not ok
				stat = reach.setdefault(k, [0, 0])
				stat[0] += 1
				stat[1] += not ok
				break
			bulb = option
	keys = set().union(*(t.keys() for t in tables.values()))
	# 只比较两种灯泡状态都可能出现的局面（第 1 天灯一定是 OFF）
	uses_bulb = any(t[(d, 1, f, y)] != t[(d, 0, f, y)] for t in tables.values() for d, _, f, y in keys
		if (d, 1, f, y) in t and (d, 0, f, y) in t)
	return {
		'value': (win - lose) / god if god else 0.0,
		'win': win / total, 'lose': lose / total,
		'announces': any(o == OPTION_TELL for t in tables.values() for o in t.values()),
		'uses_bulb': uses_bulb,
		'disagree': disagreement(tables),
		'wrong_situations': sorted(((k, s[1] / s[0], s[0] / total) for k, s in reach.items() if s[1]),
			key=lambda x: -x[1] * x[2]),
	}


def disagreement(tables):
	"""各 agent 的表有多少局面给出不同动作（协议有歧义 / 理解不一致）"""
	ts = list(tables.values())
	keys = set().union(*(t.keys() for t in ts))
	return sum(len({t.get(k) for t in ts}) > 1 for k in keys) / max(1, len(keys))


def report(res, tables, T, min_value):
	"""检查报告（英文，原样给书记员改协议用，也存成 protocol_check.md）"""
	lines = [
		'The protocol was converted into a decision table (situation -> action) by each prisoner, and every '
		'possible order of visits was simulated, assuming everyone follows their table exactly.',
		f"- Normalized reward if executed perfectly: {res['value']:.2f} "
		f"(0 = never announcing, 1.0 = best possible; the protocol needs at least {min_value:.2f} to pass).",
		f"- Games with a wrong announcement: {res['lose']:.0%}; games with a correct announcement: {res['win']:.0%}.",
	]
	if not res['announces']:
		lines.append('- The protocol never tells anyone to ANNOUNCE, so it can never win.')
	if not res['uses_bulb']:
		lines.append('- The action never depends on whether the bulb is ON or OFF: the bulb carries no information.')
	if res['disagree'] > 0:
		lines.append(f"- The prisoners understood the protocol differently in {res['disagree']:.0%} of the situations "
			f"(the wording is ambiguous).")
	if res['wrong_situations']:
		lines.append('Situations where the protocol announces although someone may not have visited yet:')
		for k, p_wrong, p_reach in res['wrong_situations'][:6]:
			lines.append(f'- {describe(k)}: wrong in {p_wrong:.0%} of the cases')
	lines += ['', 'How each prisoner understood the protocol:', '',
		'| situation | ' + ' | '.join(f'prisoner {a}' for a in sorted(tables)) + ' |',
		'|---|' + '---|' * len(tables)]
	for s in situations(T):
		k = key(s)
		lines.append(f'| {describe(k)} | ' + ' | '.join(ACTION_NAME[tables[a].get(k, OPTION_OFF)]
			for a in sorted(tables)) + ' |')
	return '\n'.join(lines)


def table_to_json(tables):
	return {str(a): {describe(k): ACTION_NAME[o] for k, o in t.items()} for a, t in tables.items()}
