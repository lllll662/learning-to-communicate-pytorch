"""
用协议检查（编译成决策表）快速对比几种决策格式，不用跑完整的比赛

	python -m icl.compare_formats -c config/switch_3_icl_given_local.json
	python -m icl.compare_formats -c config/switch_3_icl_given_local.json --formats rule_check conditions free --agents 1

每种格式把同一份协议（默认是配置里的 icl_protocol_file）编译成 29 个局面的决策表，和最优协议逐项对比：
	一致的局面数、完美执行时的分数、解析失败数，以及按"是否第一次来 × 灯"分组的选择（看出是否只匹配了合取条件的一半）
结果写在 --out 目录（默认 results/format_compare）：每种格式一个子目录（protocol_check / protocol_table /
protocol_compile.jsonl 原始输出），汇总在 compare.md。

--agents 1 只用 1 号编译（29 次生成，而不是 87 次），其余 agent 沿用 1 号的表，适合快速迭代 prompt。
"""
import argparse
import collections
import copy
import json
import os
import re
import time

from utils.dotdic import DotDic


def format_followed(fmt, compile_jsonl):
	"""
	输出是否照格式写：conditions 要真的填了 "= yes" / "= no"（不能是照抄模板的 "= yes/no"），
	rule_check 要有 RULE / CHECK 两行（free 没有格式要求）
	"""
	outs = [json.loads(line)['output'] for line in open(compile_jsonl, encoding='utf-8')]
	if fmt == 'conditions':
		# 照抄模板（"yes/no"、"<...>"）或照抄示例（下雨 / 带伞）都不算
		ok = [bool(re.search(r'=\s*(yes|no)\b(?!\s*/)', o, re.I)) and 'yes/no' not in o and '<' not in o
			and 'raining' not in o.lower() for o in outs]
	elif fmt == 'rule_check':
		ok = ['RULE:' in o and 'CHECK:' in o for o in outs]
	else:
		ok = [True] * len(outs)
	return sum(ok) / max(1, len(ok))


def group_counts(table, T):
	from icl import protocol_check as PC
	g = collections.defaultdict(collections.Counter)
	for s in PC.situations(T):
		g[('first visit' if s['first'] else 'not first', 'bulb ON' if s['bulb'] else 'bulb OFF')][
			PC.ACTION_NAME[table[PC.key(s)]]] += 1
	return g


def main():
	parser = argparse.ArgumentParser(description='compare decision formats by compiling a protocol')
	parser.add_argument('-c', '--config', required=True)
	parser.add_argument('--formats', nargs='+', default=['rule_check', 'conditions'])
	parser.add_argument('--protocol', help='protocol file (default: icl_protocol_file in the config)')
	parser.add_argument('--agents', type=int, nargs='+', help='agents that compile (default: all)')
	parser.add_argument('--out', default='results/format_compare')
	args = parser.parse_args()

	from main import init_opt, create_game
	from icl import protocol_check as PC
	from icl.backend import make_backends
	from icl.icl_arena import ICLArena

	opt = init_opt(DotDic(json.load(open(args.config, encoding='utf-8'))))
	game = create_game(opt)
	T, n = opt.nsteps, opt.game_nagents
	protocol_path = args.protocol or opt.icl_protocol_file or os.path.join(os.path.dirname(PC.__file__),
		'protocols', 'switch_3_oracle.md')
	with open(protocol_path, encoding='utf-8') as f:
		protocol = f.read()
	oracle = PC.oracle_table(T)
	backends = make_backends(opt)		# 各格式共用同一批后端，模型只加载一次

	rows, groups = [], {}
	for fmt in args.formats:
		o = copy.deepcopy(opt)
		o.icl_decision_format = fmt
		o.icl_run_dir = os.path.join(args.out, fmt)
		arena = ICLArena(o, game)
		t0 = time.time()
		print(f'=== format {fmt}')
		res, _ = arena.check_protocol(backends, protocol, 0, agents=args.agents)
		tables = json.load(open(os.path.join(o.icl_run_dir, 'round_00', 'protocol_table.json'), encoding='utf-8'))
		compiled = args.agents or list(range(1, n + 1))
		agree = total = 0
		for a in compiled:
			for s in PC.situations(T):
				total += 1
				agree += tables[str(a)][PC.describe(PC.key(s))] == PC.ACTION_NAME[oracle[PC.key(s)]]
		rows.append((fmt, agree, total, res['value'], res['compile_fail'], res['uses_bulb'],
			format_followed(fmt, os.path.join(o.icl_run_dir, 'round_00', 'protocol_compile.jsonl')), time.time() - t0))
		groups[fmt] = group_counts({PC.key(s): PC.ACTIONS[tables[str(compiled[0])][PC.describe(PC.key(s))]]
			for s in PC.situations(T)}, T)

	want = {('first visit', 'bulb ON'): 'ANNOUNCE', ('first visit', 'bulb OFF'): 'ON (day 1: OFF)',
		('not first', 'bulb ON'): 'ON', ('not first', 'bulb OFF'): 'ON if not here yesterday, else OFF'}
	lines = [f'# Decision format comparison: {os.path.basename(args.config)}', '',
		f'protocol: `{protocol_path}`, compiled by agent(s) {args.agents or "all"}', '',
		'| format | agree with oracle | value if executed perfectly | parse fail | uses bulb | format followed | time (s) |',
		'|---|---|---|---|---|---|---|']
	lines += [f'| {f} | {a}/{t} ({a / t:.0%}) | {v:.2f} | {p} | {u} | {ff:.0%} | {s:.0f} |'
		for f, a, t, v, p, u, ff, s in rows]
	lines += ['', f'Choices of agent {(args.agents or [1])[0]} grouped by (first visit?, bulb):', '',
		'| situation | oracle | ' + ' | '.join(args.formats) + ' |', '|---|---|' + '---|' * len(args.formats)]
	for k in sorted(want):
		lines.append(f'| {k[0]}, {k[1]} | {want[k]} | ' +
			' | '.join(', '.join(f'{act} {c}' for act, c in groups[f][k].most_common()) for f in args.formats) + ' |')
	text = '\n'.join(lines)
	os.makedirs(args.out, exist_ok=True)
	with open(os.path.join(args.out, 'compare.md'), 'w', encoding='utf-8') as f:
		f.write(text + '\n')
	print('\n' + text)


if __name__ == '__main__':
	main()
