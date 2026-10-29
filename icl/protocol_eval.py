"""
基于比赛记录的协议评估：在一组固定的来访顺序上，用协议真的打一遍比赛

和之前的"协议编译"（把协议翻译成 29 个局面的决策表）不同：
	- 比赛是真的：每一步都走正常的决策 prompt，agent 有自己的访问记录、自己的笔记；
	- 所以计数员这类依赖笔记状态的协议能被正确评估。编译时"没有笔记"，计数永远是 0，会把好协议判成 0 分；
	- 各 agent 本来就可以对自己的角色（1 号是计数员）有不同的行为，不再出现"理解不一致"这种误报。

基准（benchmark）：一组固定种子生成的来访顺序，整次实验都用同一组，所以不同轮次、不同协议的分数可以直接比较。
提前知道来访顺序并不构成作弊：它只是比赛的随机环境，协议本身看不到它。
"""
import numpy as np


def make_benchmark(nagents, nsteps, bs, seed=0):
	"""返回 [bs, nsteps] 的来访顺序（1-indexed），和 SwitchGame 的 active_agent 同形状"""
	rng = np.random.default_rng(seed)
	return rng.integers(1, nagents + 1, (bs, nsteps))


def god_reward(active, nagents):
	"""每一局里所有人是否都来过（归一化奖励的分母）"""
	return np.array([len(set(row.tolist())) == nagents for row in active])


def summarize(episode, m, protocol, nagents):
	"""人类可读的中文评估报告"""
	god = episode.game_stats.god_reward.numpy()
	r = episode.r[:, 0].numpy()
	win = int((r > 0).sum())
	lose = int((r < 0).sum())
	announced = int(episode.tell.sum().item())
	bulb_on = sum(1 for d in episode.decisions if d['action'] == 'ANNOUNCE' and d['bulb_seen'] == 'ON')
	lines = [
		f'# 协议评估（在 {len(r)} 局固定的来访顺序上重放）',
		'',
		f'- **归一化奖励：{m["avg_reward"]:.2f}**（每次实验的机会相同，可以和 0.65 / 1.0 直接比）',
		f'- 正确宣布 {win} 局，错误宣布 {lose} 局，宣布 {announced} 局',
		f'- 宣布时灯是开着的：{bulb_on}/{announced}',
		f'- 解析失败：{m["parse_fail"]:.1%}',
	]
	if lose == 0 and m['avg_reward'] < 0.65 and announced < 10:
		lines.append('- 这份协议说错的次数为 0，但也很少宣布：安全，但赢得少。')
	if announced == 0:
		lines.append('- **从不宣布，所以最好也只能得 0 分。**')
	return '\n'.join(lines)
