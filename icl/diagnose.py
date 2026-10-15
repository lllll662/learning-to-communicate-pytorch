"""
ICL 实验的诊断：按决策统计"执行得对不对"、"宣布时用了什么信息"

	python -m icl.diagnose results/icl_given_0

ICLArena 每轮结束时也会调用 analyze()，结果写进 icl_metrics.csv / summary.json。
只读 decisions.jsonl，不需要模型。

- 按规则的正确率：和 icl/protocols/switch_3_oracle.md 的规则对比（只对 3 人有意义）。
  E1 里就是执行正确率；E2 / E3 里表示和最优协议的接近程度
- 宣布分析：宣布时看到的灯、是不是第一次来、第几天。正确的协议只在灯 ON 时宣布；
  如果大部分宣布都发生在灯 OFF 时，说明赢的局靠的是日期和自己的访问记录，而不是通信
"""
import argparse
import collections
import glob
import json
import os

from icl.backend import oracle_option
from icl.prompts import ACTIONS

RULES = [
	('R1', 'bulb ON + first visit -> ANNOUNCE'),
	('R2', 'bulb ON -> ON'),
	('R3', 'bulb OFF, day > 1, not here yesterday -> ON'),
	('R4', 'otherwise -> OFF'),
]
# 每类错误的后果：说错（-1）比少说（0）严重
CONSEQUENCE = {
	('R1', 'ON'): 'missed win', ('R1', 'OFF'): 'missed win (and signal lost)',
	('R2', 'OFF'): 'signal lost', ('R2', 'ANNOUNCE'): 'WRONG ANNOUNCE',
	('R3', 'OFF'): 'signal lost', ('R3', 'ANNOUNCE'): 'WRONG ANNOUNCE',
	('R4', 'ON'): 'FALSE SIGNAL (next newcomer will announce wrongly)', ('R4', 'ANNOUNCE'): 'WRONG ANNOUNCE',
}


def classify(d, own):
	bulb = d['bulb_seen'] == 'ON'
	first = not own
	yesterday = bool(own) and own[-1]['day'] == d['day'] - 1
	if bulb and first:
		return 'R1'
	if bulb:
		return 'R2'
	if d['day'] > 1 and not yesterday:
		return 'R3'
	return 'R4'


def analyze(recs, n=3):
	"""
	recs: 一轮（或多轮拼在一起，game 需带上轮次区分）的决策记录
	返回 {'rules', 'oracle_acc', 'announce_*', 'answer_first', 'examples', 'by_situation'}
	"""
	games = collections.defaultdict(list)
	for d in recs:
		games[d.get('_round', 0), d['game']].append(d)
	rules = collections.defaultdict(collections.Counter)
	by_situation = collections.defaultdict(lambda: [0, 0])	# (day, 第一次来, 灯) -> [胜, 负]
	examples = {}
	ann = ann_win = ann_on = answer_first = 0
	for ds in games.values():
		ds.sort(key=lambda d: d['day'])
		visited = set()
		for d in ds:
			own = [x for x in ds if x['agent'] == d['agent'] and x['day'] < d['day']]
			visited.add(d['agent'])
			answer_first += d['output'].lstrip().startswith('{')
			if n == 3:
				rule = classify(d, own)
				want = oracle_option(d['day'], d['bulb_seen'] == 'ON', [(x['day'], 0, 0) for x in own])
				got = ACTIONS[d['action']] if d['parsed'] else None
				k = 'ok' if got == want else 'PARSE FAIL' if got is None else d['action']
				rules[rule][k] += 1
				if k != 'ok':
					examples.setdefault((rule, k), d)
			if d['action'] == 'ANNOUNCE':
				ok = len(visited) == n
				ann += 1
				ann_win += ok
				ann_on += d['bulb_seen'] == 'ON'
				by_situation[d['day'], 'first visit' if not own else 'revisit', d['bulb_seen']][0 if ok else 1] += 1
	m = {
		'announce_n': ann,
		'announce_win_rate': ann_win / ann if ann else float('nan'),
		'announce_bulb_on': ann_on / ann if ann else float('nan'),
		'answer_first': answer_first / max(1, len(recs)),
		'rules': {r: dict(c) for r, c in rules.items()},
		'examples': examples,
		'by_situation': dict(by_situation),
	}
	if n == 3:
		for r, _ in RULES:
			c = rules.get(r, collections.Counter())
			m['acc_' + r] = c['ok'] / sum(c.values()) if c else float('nan')
		total = sum(sum(c.values()) for c in rules.values())
		m['oracle_acc'] = sum(c['ok'] for c in rules.values()) / total if total else float('nan')
	return m


def load(run_dir):
	recs = []
	for path in sorted(glob.glob(os.path.join(run_dir, 'round_*', 'decisions.jsonl'))):
		r = int(path.split('round_')[-1][:2])
		recs += [dict(json.loads(line), _round=r) for line in open(path, encoding='utf-8')]
	return recs


def main(run_dir, n=3, show=700):
	recs = load(run_dir)
	if not recs:
		print('no decisions.jsonl found under', run_dir)
		return
	m = analyze(recs, n)
	print(f'{len(recs)} decisions over {len({d["_round"] for d in recs})} rounds; '
		f'outputs starting with the answer (JSON before reasoning): {m["answer_first"]:.0%}\n')
	if n == 3:
		print(f'Agreement with the oracle protocol: {m["oracle_acc"]:.2f}')
		print(f'{"rule":4s} {"situation":44s} {"n":>5s} {"acc":>6s}  wrong choices')
		for rule, desc in RULES:
			c = m['rules'].get(rule, {})
			total = sum(c.values())
			wrong = ', '.join(f'{k} x{v} ({CONSEQUENCE.get((rule, k), "")})' for k, v in c.items() if k != 'ok')
			print(f'{rule:4s} {desc:44s} {total:5d} {c.get("ok", 0) / total if total else 0:6.2f}  {wrong}')
	print(f'\nAnnouncements: {m["announce_n"]}, win rate {m["announce_win_rate"]:.2f}, '
		f'made with the bulb ON: {m["announce_bulb_on"]:.2f}')
	for k in sorted(m['by_situation']):
		w, l = m['by_situation'][k]
		print(f'  day {k[0]}, {k[1]:11s}, bulb {k[2]:3s}: win {w:3d} lose {l:3d}')
	for (rule, key), d in (m['examples'].items() if show > 0 else []):
		print(f'\n===== {rule} chose {key}: round {d["_round"]} game {d["game"]}, day {d["day"]}, '
			f'prisoner {d["agent"]}, bulb {d["bulb_seen"]}')
		out = d['output']
		print(out if len(out) <= show else '...' + out[-show:])


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='per-rule accuracy and announcement analysis of an ICL run')
	parser.add_argument('run_dir')
	parser.add_argument('--nagents', type=int, default=3)
	parser.add_argument('--show', type=int, default=700, help='characters of each example output to print')
	args = parser.parse_args()
	main(args.run_dir, args.nagents, args.show)
