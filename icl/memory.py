"""
ICL 实验的显式外部记忆：所有上下文都落成文件，方便事后阅读、分析协议

	run_dir/
		state.json                     续跑用：已完成的轮次、当前协议、累积经验、测试结果、随机数状态
		train_log.csv                  每轮的指标（和 llm_arena 格式一致，可用 python -m llm.plot_training 画图）
		lessons.md                     复盘累积的经验（E3）
		round_00/
			discussion.md              赛前讨论 + 修改意见（E2 / E3）
			protocol.md                这一轮所有人遵守的协议
			debrief.md                 赛后复盘：统计 + 失败对局的完整记录
			decisions.jsonl            每个决策：prompt、原始输出、解析结果、笔记
			summary.json               这一轮的指标
			games/g00/agent_1_memory.md  每局结束时各 agent 的 memory.md
"""
import json
import os

STATE_NAME = 'state.json'


class Workspace:
	def __init__(self, run_dir):
		self.run_dir = run_dir
		os.makedirs(run_dir, exist_ok=True)

	def path(self, *parts):
		p = os.path.join(self.run_dir, *parts)
		os.makedirs(os.path.dirname(p), exist_ok=True)
		return p

	@staticmethod
	def round_name(r):
		return f'round_{r:02d}'

	def write(self, name, text, r=None):
		parts = (self.round_name(r), name) if r is not None else (name,)
		with open(self.path(*parts), 'w', encoding='utf-8') as f:
			f.write(text.rstrip() + '\n')

	def read(self, name, r=None):
		parts = (self.round_name(r), name) if r is not None else (name,)
		p = os.path.join(self.run_dir, *parts)
		if not os.path.exists(p):
			return ''
		with open(p, encoding='utf-8') as f:
			return f.read()

	def write_json(self, name, obj, r=None):
		self.write(name, json.dumps(obj, ensure_ascii=False, indent=1), r)

	def write_jsonl(self, name, records, r):
		with open(self.path(self.round_name(r), name), 'w', encoding='utf-8') as f:
			for rec in records:
				f.write(json.dumps(rec, ensure_ascii=False) + '\n')

	def write_memory(self, r, b, agent_id, text):
		self.write(os.path.join('games', f'g{b:02d}', f'agent_{agent_id}_memory.md'), text, r)

	def load_state(self):
		p = os.path.join(self.run_dir, STATE_NAME)
		if not os.path.exists(p):
			return None
		with open(p, encoding='utf-8') as f:
			return json.load(f)

	def save_state(self, state):
		# 先写临时文件再替换：保存到一半被打断，也不会损坏上一次的状态
		p = os.path.join(self.run_dir, STATE_NAME)
		with open(p + '.tmp', 'w', encoding='utf-8') as f:
			json.dump(state, f, ensure_ascii=False)
		os.replace(p + '.tmp', p)
