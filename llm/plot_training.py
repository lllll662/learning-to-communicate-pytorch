"""
训练曲线可视化：读取 llm_save_path/train_log.csv，画成 PNG

	单次训练总览（测试奖励 / 测试时的行为 / loss），默认存为 <run_dir>/training_curves.png：
		python -m llm.plot_training results/lora_switch_3_rl_0
	多次训练对比测试奖励（比如有通信 vs 无通信），默认存为第一个目录旁边的 training_compare.png：
		python -m llm.plot_training results/comm_0 results/nocomm_0 -l comm no-comm -o results/compare.png

LLMArena.train() 训练结束时会自动调用 plot_run。train_log.csv 每个 episode 都会刷新，
所以训练过程中也可以随时运行这个脚本查看进度。
只依赖 matplotlib / numpy，不加载模型；用 Figure 对象直接存图，没有显示器的服务器上也能跑。
"""
import argparse
import csv
import math
import os

import numpy as np
from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

LOG_NAME = 'train_log.csv'		# 和 llm_arena.LOG_NAME 一致
PLOT_NAME = 'training_curves.png'

# 配色：分类色按固定顺序使用；文字只用墨色，颜色只由线条表示
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
STYLE = {'font.family': 'sans-serif', 'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'Arial'], 'font.size': 9}


def no_comm_baseline(nagents, nsteps):
	"""
	不通信时的参照线：策略"只在最后一天宣布"的归一化奖励
	p = T 天内所有人都来过的概率（容斥原理），奖励 = p·(+1) + (1-p)·(-1)，归一化再除以 p
	p < 0.5 时不如永远不宣布（0 分）
	"""
	p = sum((-1) ** k * math.comb(nagents, k) * ((nagents - k) / nagents) ** nsteps for k in range(nagents + 1))
	return max(0.0, (2 * p - 1) / p)


def load_log(run_dir):
	"""返回 {'train': {列名: np.array}, 'test': {...}}，外加最后一行的累计用时（秒）"""
	with open(os.path.join(run_dir, LOG_NAME), newline='') as f:
		rows = list(csv.DictReader(f))
	data = {}
	for phase in ('train', 'test'):
		sel = [r for r in rows if r['phase'] == phase]
		data[phase] = {k: np.array([float(r[k]) if r[k] != '' else np.nan for r in sel])
			for k in ('episode', 'avg_reward', 'tell_rate', 'win', 'lose', 'loss', 'elapsed_s')}
	data['elapsed'] = float(rows[-1]['elapsed_s']) if rows else 0.0
	data['episodes'] = int(max(float(r['episode']) for r in rows)) + 1 if rows else 0
	return data


def ewma(x, alpha):
	out = np.empty(len(x))
	for i, v in enumerate(x):
		out[i] = v if i == 0 else alpha * v + (1 - alpha) * out[i - 1]
	return out


def fmt_duration(seconds):
	s = int(seconds)
	return f'{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}'


# ---------------- 画图小工具 ----------------

def style_axes(ax, title, ylabel, xlabel='episode'):
	ax.set_facecolor(SURFACE)
	for s in ('top', 'right'):
		ax.spines[s].set_visible(False)
	for s in ('left', 'bottom'):
		ax.spines[s].set_color(AXIS)
	ax.tick_params(colors=AXIS, labelcolor=MUTED, length=3)
	ax.grid(axis='y', color=GRID, linewidth=0.6)
	ax.set_axisbelow(True)
	ax.set_title(title, loc='left', color=INK, fontsize=10.5, fontweight='bold')
	ax.set_xlabel(xlabel, color=INK_2)
	ax.set_ylabel(ylabel, color=INK_2)


def plot_line(ax, x, y, color, label=None, lw=2.0, marker=False, alpha=1.0):
	# 点多时不画标记，否则标记连成一串，线看起来像虚线
	kw = dict(marker='o', markersize=4, markeredgecolor=SURFACE, markeredgewidth=1) \
		if marker and len(x) <= 30 else {}
	ax.plot(x, y, color=color, linewidth=lw, label=label, alpha=alpha,
		solid_capstyle='round', solid_joinstyle='round', **kw)


def set_xlim_with_label_room(ax, x_max, room=0.13):
	"""右侧留出空白放线末端的标签；刻度不超过数据范围"""
	x_max = max(x_max, 1)
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))		# episode 是整数，轮数少时不出现 0.5 这样的刻度
	ax.set_xlim(0, x_max * (1 + room))
	ax.set_xticks([t for t in ax.get_xticks() if 0 <= t <= x_max])
	ax.set_xlim(0, x_max * (1 + room))


def end_labels(ax, items):
	"""items: [(x, y, 文字)]，在每条线的末端标注名称；末端靠得太近时依次往上错开，超出上边界就整体往下挪"""
	lo, hi = ax.get_ylim()
	x_lo, x_hi = ax.get_xlim()
	gap = (hi - lo) * 0.07
	placed = []
	for x, y, text in sorted(items, key=lambda t: t[1]):
		y_text = y if not placed else max(y, placed[-1][1] + gap)
		placed.append((x, y_text, text))
	overflow = placed[-1][1] - (hi - gap * 0.5) if placed else 0
	for x, y_text, text in placed:
		ax.text(x + (x_hi - x_lo) * 0.012, y_text - max(0, overflow), text, va='center', ha='left',
			color=INK_2, fontsize=8.5)


def reference_lines(ax, baseline, x_max):
	"""参照线：最优协议 1.0、不通信的参照线。只画到最后一个数据点，右侧留给线末端的标签"""
	x_max = max(x_max, 1)
	for y, text in ((1.0, 'oracle protocol 1.0'), (baseline, f'no-communication baseline {baseline:.2f}')):
		ax.hlines(y, 0, x_max, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
		ax.text(0, y, ' ' + text, va='bottom', ha='left', color=MUTED, fontsize=8)


def top_legend(ax, ncol):
	"""图例放在标题行的右侧，不遮挡数据"""
	ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=ncol, frameon=False,
		labelcolor=INK_2, fontsize=8.5, handlelength=1.6, borderaxespad=0.2)


def reward_ylim(*arrays):
	vals = np.concatenate([a[~np.isnan(a)] for a in arrays if len(a)] or [np.zeros(1)])
	return min(-0.05, vals.min() - 0.05), 1.08


# ---------------- 单次训练总览 ----------------

def plot_run(run_dir, out=None, nagents=3, nsteps=None, smooth=0.05):
	"""
	三个面板：
		上：归一化奖励——测试（贪心）+ 训练采样（含 ε 探索，EWMA 平滑）+ 两条参照线
		左下：测试时的行为——胜率 / 负率 / 宣布率
		右下：训练 loss（原始值 + EWMA）
	"""
	nsteps = nsteps or 4 * nagents - 6		# 和 SwitchGame 一致
	data = load_log(run_dir)
	train, test = data['train'], data['test']
	baseline = no_comm_baseline(nagents, nsteps)
	x_max = data['episodes'] - 1

	with rc_context(STYLE):
		fig = Figure(figsize=(12, 7.5), dpi=150, facecolor=SURFACE, layout='constrained')
		gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])

		# 统计信息写在总标题里
		stats = [f'{data["episodes"]} episodes', f'training time {fmt_duration(data["elapsed"])}']
		if data['episodes']:
			stats.append(f'{data["elapsed"] / data["episodes"]:.1f} s/episode')
		if len(test['episode']):
			best = int(np.nanargmax(test['avg_reward']))
			stats.append(f'best test reward {test["avg_reward"][best]:.2f} @ episode {int(test["episode"][best])}')
			stats.append(f'last {test["avg_reward"][-1]:.2f}')
		fig.suptitle(f'{os.path.basename(os.path.normpath(run_dir))}\n' + '  ·  '.join(stats),
			x=0.01, ha='left', color=INK, fontsize=11)

		# 上：归一化奖励
		ax = fig.add_subplot(gs[0, :])
		style_axes(ax, 'Normalized reward', 'normalized reward')
		set_xlim_with_label_room(ax, x_max)
		train_smooth = ewma(train['avg_reward'], smooth) if len(train['episode']) else np.zeros(0)
		ax.set_ylim(*reward_ylim(test['avg_reward'], train_smooth))	# 按平滑后的曲线定范围，不被单轮噪声拉宽
		reference_lines(ax, baseline, x_max)
		labels = []
		if len(train['episode']):
			plot_line(ax, train['episode'], train_smooth, SERIES[1], f'train, sampled with ε (EWMA α={smooth})', lw=1.5)
			labels.append((train['episode'][-1], train_smooth[-1], 'train'))
		if len(test['episode']):
			plot_line(ax, test['episode'], test['avg_reward'], SERIES[0], 'test, greedy', marker=True)
			labels.append((test['episode'][-1], test['avg_reward'][-1], 'test'))
		end_labels(ax, labels)
		top_legend(ax, ncol=2)

		# 左下：测试时的行为
		ax = fig.add_subplot(gs[1, 0])
		style_axes(ax, 'Test behavior (greedy)', 'fraction of games')
		set_xlim_with_label_room(ax, x_max, room=0.2)
		ax.set_ylim(-0.03, 1.03)
		labels = []
		if len(test['episode']):
			for i, (key, name) in enumerate((('win', 'win'), ('lose', 'lose'), ('tell_rate', 'announce'))):
				plot_line(ax, test['episode'], test[key], SERIES[i], name, marker=True)
				labels.append((test['episode'][-1], test[key][-1], name))
		end_labels(ax, labels)
		top_legend(ax, ncol=3)

		# 右下：训练 loss
		ax = fig.add_subplot(gs[1, 1])
		style_axes(ax, f'Training loss (REINFORCE, EWMA α={smooth})', 'loss')
		set_xlim_with_label_room(ax, x_max, room=0.02)
		ax.axhline(0, color=AXIS, linewidth=1, zorder=1)
		if len(train['episode']):
			plot_line(ax, train['episode'], train['loss'], SERIES[0], lw=1, alpha=0.3)
			plot_line(ax, train['episode'], ewma(train['loss'], smooth), SERIES[0])

		out = out or os.path.join(run_dir, PLOT_NAME)
		fig.savefig(out, facecolor=SURFACE)
	return out


# ---------------- 多次训练对比 ----------------

def plot_compare(run_dirs, labels=None, out=None, nagents=3, nsteps=None):
	"""每次训练一条测试奖励曲线（颜色按固定顺序分配），用于比较不同设定，比如有通信 vs 无通信"""
	assert len(run_dirs) <= len(SERIES), f'at most {len(SERIES)} runs per figure'
	nsteps = nsteps or 4 * nagents - 6
	# 默认用目录名；matplotlib 会忽略下划线开头的图例标签，所以去掉开头的下划线
	labels = labels or [os.path.basename(os.path.normpath(d)).lstrip('_') for d in run_dirs]
	assert len(labels) == len(run_dirs), 'need one label per run'
	runs = [load_log(d) for d in run_dirs]

	with rc_context(STYLE):
		fig = Figure(figsize=(11, 5), dpi=150, facecolor=SURFACE, layout='constrained')
		ax = fig.add_subplot()
		style_axes(ax, 'Normalized test reward (greedy)', 'normalized reward')
		x_max = max(r['episodes'] for r in runs) - 1
		set_xlim_with_label_room(ax, x_max, room=0.16)
		ax.set_ylim(*reward_ylim(*[r['test']['avg_reward'] for r in runs]))
		reference_lines(ax, no_comm_baseline(nagents, nsteps), x_max)
		ends = []
		for i, (run, label) in enumerate(zip(runs, labels)):
			test = run['test']
			if not len(test['episode']):
				continue
			plot_line(ax, test['episode'], test['avg_reward'], SERIES[i], label, marker=True)
			ends.append((test['episode'][-1], test['avg_reward'][-1], label))
		if len(ends) <= 4:		# 线多了直接标注会挤在一起，只保留图例
			end_labels(ax, ends)
		top_legend(ax, ncol=min(len(runs), 4))

		out = out or os.path.join(os.path.dirname(os.path.normpath(run_dirs[0])) or '.', 'training_compare.png')
		fig.savefig(out, facecolor=SURFACE)
	return out


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='plot LLM training curves from train_log.csv')
	parser.add_argument('run_dirs', nargs='+', help='llm_save_path directories (containing train_log.csv)')
	parser.add_argument('-l', '--labels', nargs='+', help='labels for comparing multiple runs')
	parser.add_argument('-o', '--out', help='output PNG path')
	parser.add_argument('--nagents', type=int, default=3, help='number of prisoners (for the no-comm baseline)')
	parser.add_argument('--smooth', type=float, default=0.05, help='EWMA alpha for training curves')
	args = parser.parse_args()

	if len(args.run_dirs) == 1:
		path = plot_run(args.run_dirs[0], args.out, nagents=args.nagents, smooth=args.smooth)
	else:
		path = plot_compare(args.run_dirs, args.labels, args.out, nagents=args.nagents)
	print('saved', path)
