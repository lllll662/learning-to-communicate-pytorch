"""
ICL 实验的曲线：读 run_dir/icl_metrics.csv，存成 run_dir/icl_curves.png

	python -m icl.plot_icl results/icl_reflect_0

三个面板：
	上：每轮的归一化 reward（±1.96 标准误的误差带）和"协议完美执行时的分数"（协议检查），加 0.65 / 1.0 参照线。
	    两条线的差距就是执行上的损失；协议分数本身低于 0.65，说明协议设计得不好
	左下：宣布的胜率、宣布时灯是 ON 的比例（正确的协议只在灯 ON 时宣布；比例低说明没有用到通信）
	右下：和最优协议的一致率，按规则分开（3 人时）
"""
import argparse
import csv
import os

import numpy as np
from matplotlib import rc_context
from matplotlib.figure import Figure

from llm.plot_training import (SERIES, STYLE, SURFACE, INK, end_labels, no_comm_baseline,
	plot_line, reference_lines, reward_ylim, set_xlim_with_label_room, style_axes, top_legend)

METRICS_NAME = 'icl_metrics.csv'	# 和 icl_arena.METRICS_NAME 一致
PLOT_NAME = 'icl_curves.png'


def load_metrics(run_dir):
	with open(os.path.join(run_dir, METRICS_NAME), newline='') as f:
		rows = list(csv.DictReader(f))
	cols = rows[0].keys() if rows else []
	return {k: np.array([float(r[k]) if r[k] not in ('', 'nan') else np.nan for r in rows]) for k in cols}


def plot_icl(run_dir, out=None, nagents=3, nsteps=None):
	nsteps = nsteps or 4 * nagents - 6
	m = load_metrics(run_dir)
	x = m['round']
	x_max = int(x.max()) if len(x) else 0
	baseline = no_comm_baseline(nagents, nsteps)
	has_check = 'protocol_value' in m and not np.all(np.isnan(m['protocol_value']))
	has_rules = 'acc_R1' in m and not np.all(np.isnan(m['acc_R1']))

	with rc_context(STYLE):
		fig = Figure(figsize=(12, 7.5), dpi=150, facecolor=SURFACE, layout='constrained')
		gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
		reward = m['avg_reward']
		stats = [f'{len(x)} rounds', f'mean reward {np.nanmean(reward):.2f}',
			f'typical s.e. per round {np.nanmean(m["reward_se"]):.2f}']
		if has_check:
			stats.append(f'mean protocol value {np.nanmean(m["protocol_value"]):.2f}')
		fig.suptitle(f'{os.path.basename(os.path.normpath(run_dir))}\n' + '  ·  '.join(stats),
			x=0.01, ha='left', color=INK, fontsize=11)

		# 上：实际 reward vs 协议完美执行的分数
		ax = fig.add_subplot(gs[0, :])
		style_axes(ax, 'Normalized reward per round', 'normalized reward', xlabel='round')
		set_xlim_with_label_room(ax, x_max, room=0.15)
		band = 1.96 * m['reward_se']
		ax.set_ylim(*reward_ylim(reward - band, m['protocol_value'] if has_check else np.zeros(0)))
		reference_lines(ax, baseline, x_max)
		ax.fill_between(x, reward - band, reward + band, color=SERIES[0], alpha=0.15, linewidth=0)
		plot_line(ax, x, reward, SERIES[0], 'played (±1.96 s.e.)', marker=True)
		labels = [(x[-1], reward[-1], 'played')]
		if has_check:
			plot_line(ax, x, m['protocol_value'], SERIES[1], 'protocol if executed perfectly', marker=True)
			labels.append((x[-1], m['protocol_value'][-1], 'protocol'))
		end_labels(ax, labels)
		top_legend(ax, ncol=2)

		# 左下：宣布分析
		ax = fig.add_subplot(gs[1, 0])
		style_axes(ax, 'Announcements', 'fraction', xlabel='round')
		set_xlim_with_label_room(ax, x_max, room=0.3)
		ax.set_ylim(-0.03, 1.03)
		labels = []
		for i, (k, name) in enumerate((('announce_win_rate', 'win rate'), ('announce_bulb_on', 'made with bulb ON'))):
			plot_line(ax, x, m[k], SERIES[i], name, marker=True)
			last = np.flatnonzero(~np.isnan(m[k]))
			if len(last):
				labels.append((x[last[-1]], m[k][last[-1]], name))
		end_labels(ax, labels)
		top_legend(ax, ncol=2)

		# 右下：和最优协议的一致率
		ax = fig.add_subplot(gs[1, 1])
		style_axes(ax, 'Agreement with the oracle protocol', 'fraction of decisions', xlabel='round')
		set_xlim_with_label_room(ax, x_max, room=0.2)
		ax.set_ylim(-0.03, 1.03)
		if has_rules:
			labels = []
			for i, rule in enumerate(('R1', 'R2', 'R3', 'R4')):
				plot_line(ax, x, m['acc_' + rule], SERIES[i], rule, marker=True)
				last = np.flatnonzero(~np.isnan(m['acc_' + rule]))
				if len(last):
					labels.append((x[last[-1]], m['acc_' + rule][last[-1]], rule))
			end_labels(ax, labels)
			top_legend(ax, ncol=4)
		else:
			ax.text(0.5, 0.5, 'only available for 3 prisoners', transform=ax.transAxes, ha='center', color=INK)

		out = out or os.path.join(run_dir, PLOT_NAME)
		fig.savefig(out, facecolor=SURFACE)
	return out


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='plot ICL metrics from icl_metrics.csv')
	parser.add_argument('run_dir')
	parser.add_argument('-o', '--out', help='output PNG path')
	parser.add_argument('--nagents', type=int, default=3)
	args = parser.parse_args()
	print('saved', plot_icl(args.run_dir, args.out, nagents=args.nagents))
