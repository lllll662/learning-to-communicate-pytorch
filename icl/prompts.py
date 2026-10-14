"""
上下文学习（ICL）版本的 prompt：规则说明、比赛中的决策、赛前商定协议、赛后复盘

信息隔离：比赛中一个 agent 的 prompt 只由 protocol.md（所有人共享，赛前确定）、
它自己的 memory.md、今天的观测三部分组成，build_decision_messages 的参数里没有其他 agent 的任何信息。
自然语言交流只发生在比赛外（商定协议、复盘）。
"""
import json
import re

from llm.prompt_builder import OPTION_OFF, OPTION_ON, OPTION_TELL, bulb_str, describe_option

# 决策输出里 action 字段的取值 → 游戏选项
ACTIONS = {'OFF': OPTION_OFF, 'ON': OPTION_ON, 'ANNOUNCE': OPTION_TELL}
ACTION_ALIASES = {'TELL': 'ANNOUNCE', 'C': 'ANNOUNCE', 'A': 'OFF', 'B': 'ON'}


def rules(opt):
	"""所有阶段共用的规则说明（和 llm.prompt_builder.system_prompt 相同的游戏，但每个囚犯是独立的个体）"""
	n, T = opt.game_nagents, opt.nsteps
	return (
		f"You are one of {n} prisoners (numbered 1 to {n}) in a cooperative puzzle.\n"
		f"The game lasts {T} days. Each day the warden picks one prisoner uniformly at random and brings "
		f"them into a room with a light bulb. The same prisoner can be picked several times, and some "
		f"prisoners may never be picked.\n"
		f"When you enter the room you see the bulb exactly as the previous day's prisoner left it "
		f"(on day 1 it is OFF). Before leaving you either set the bulb ON or OFF for the next prisoner, "
		f"or you announce: \"Every prisoner has visited the room.\"\n"
		f"If the announcement is correct, everyone wins (reward +1). If it is wrong, everyone loses "
		f"(reward -1). If nobody announces within {T} days, the reward is 0.\n"
		f"During the game prisoners cannot talk to each other: the bulb is the only signal, and it only "
		f"reaches the prisoner who enters the next day. Before the game the prisoners may meet and agree "
		f"on a protocol, written down in protocol.md. Each prisoner also keeps a private memory.md."
	)


# ---------------- 比赛中的决策 ----------------

def render_memory(facts, notes):
	"""
	一个 agent 的 memory.md
	facts: 程序记录的客观访问历史 [(day, 看到的bit, option), ...]
	notes: agent 自己写的笔记（每次决策时覆盖）
	"""
	lines = ['## Visits (recorded automatically)']
	if facts:
		lines += [f"- day {d}: the bulb was {bulb_str(seen)}, you {describe_option(o)}." for d, seen, o in facts]
	else:
		lines.append('- none: this is your first visit.')
	lines += ['', '## Your notes', notes.strip() if notes and notes.strip() else '(empty)']
	return '\n'.join(lines)


def build_decision_messages(opt, agent_id, day, bulb, facts, notes, protocol):
	lines = [
		'# protocol.md (agreed by all prisoners before the game)',
		protocol.strip(),
		'',
		'# memory.md (your private memory; nobody else can read it)',
		render_memory(facts, notes),
		'',
		'# Today',
		f"You are prisoner {agent_id}. Today is day {day} of {opt.nsteps}. "
		f"You are in the room and the bulb is {bulb_str(bulb)}."]
	if day > 1:
		# 从访问记录就能推出来，但直接写明可以减少推理错误（和 llm.prompt_builder 一致）
		was_here = bool(facts) and facts[-1][0] == day - 1
		lines.append(f"Yesterday (day {day - 1}) you were {'' if was_here else 'not '}the prisoner in the room.")
	lines += [
		'',
		'Decide what to do by following the protocol. Think step by step briefly, then end your reply '
		'with a JSON object on its own line:',
		'{"action": "OFF" or "ON" or "ANNOUNCE", "memory": "notes for your future visits"}',
		'"OFF" / "ON" sets the bulb for the next prisoner; "ANNOUNCE" ends the game. '
		'"memory" replaces your previous notes, so keep everything you still need.',
	]
	return [{'role': 'system', 'content': rules(opt)}, {'role': 'user', 'content': '\n'.join(lines)}]


def parse_decision(text):
	"""返回 (option 或 None, 笔记)；优先取最后一个合法 JSON，JSON 写坏时退回正则"""
	for block in reversed(re.findall(r'\{[^{}]*\}', text, re.S)):
		try:
			obj = json.loads(block)
		except json.JSONDecodeError:
			continue
		if isinstance(obj, dict) and 'action' in obj:
			option = _to_option(obj['action'])
			if option is not None:
				return option, str(obj.get('memory', ''))
	m = re.findall(r'"action"\s*:\s*"?([A-Za-z]+)', text)
	if m:
		option = _to_option(m[-1])
		if option is not None:
			notes = re.findall(r'"memory"\s*:\s*"(.*?)"\s*\}', text, re.S)
			return option, notes[-1] if notes else ''
	return None, ''


def _to_option(action):
	a = str(action).strip().upper()
	a = ACTION_ALIASES.get(a, a)
	return ACTIONS.get(a)


# ---------------- 赛前商定协议 / 赛后复盘 ----------------

def _context_block(context):
	"""E3：上一轮的协议、复盘和累积的经验"""
	if not context:
		return []
	lines = []
	if context.get('protocol'):
		lines += ['# Protocol used in the previous round', context['protocol'].strip(), '']
	if context.get('debrief'):
		lines += ['# Debrief of the previous round', context['debrief'].strip(), '']
	if context.get('lessons'):
		lines += ['# Lessons learned so far', context['lessons'].strip(), '']
	return lines


def _protocol_requirements(opt):
	return (f"The protocol must be a short list of precise rules that a prisoner can apply alone, during the "
		f"game, using only: the day number (1 to {opt.nsteps}), the bulb state seen when entering, whether "
		f"they were in the room yesterday, their own visit history, and their own private notes.")


def build_discussion_messages(opt, agent_id, turn, nturns, transcript, context=None):
	lines = _context_block(context) + [
		f"The prisoners are meeting before the game to agree on a protocol. You are prisoner {agent_id}.",
		_protocol_requirements(opt),
		'',
		'# Discussion so far',
		render_transcript(transcript) or '(nobody has spoken yet)',
		'',
		f"It is your turn to speak (round {turn + 1} of {nturns}). Propose a protocol, or point out a concrete "
		f"flaw in a proposal and fix it. Reply in under 200 words.",
	]
	return [{'role': 'system', 'content': rules(opt)}, {'role': 'user', 'content': '\n'.join(lines)}]


def build_draft_messages(opt, transcript, context=None, draft=None, reviews=None):
	"""1 号整理协议；带 draft/reviews 时是按修改意见改稿"""
	lines = _context_block(context) + ['# Discussion', render_transcript(transcript), '']
	if draft is None:
		lines.append('You are prisoner 1 and act as the scribe. Write the final protocol.md that all prisoners '
			'will follow, based on the discussion.')
	else:
		lines += ['# Your draft protocol.md', draft.strip(), '', '# Suggestions from the other prisoners',
			'\n'.join(f'- prisoner {k}: {r.strip()}' for k, r in reviews), '',
			'You are prisoner 1 and act as the scribe. Revise the draft where the suggestions are correct.']
	lines += [_protocol_requirements(opt), 'Output only the markdown content of protocol.md.']
	return [{'role': 'system', 'content': rules(opt)}, {'role': 'user', 'content': '\n'.join(lines)}]


def build_review_messages(opt, agent_id, draft):
	lines = [
		'# Draft protocol.md', draft.strip(), '',
		f"You are prisoner {agent_id}. Check whether this protocol can cause a wrong announcement or can be "
		f"applied ambiguously. If it is fine, reply exactly AGREE. Otherwise reply with one concrete "
		f"suggestion in under 80 words.",
	]
	return [{'role': 'system', 'content': rules(opt)}, {'role': 'user', 'content': '\n'.join(lines)}]


def is_agree(text):
	return text.strip().strip('.').upper().startswith('AGREE')


def build_lessons_messages(opt, protocol, debrief, lessons):
	lines = [
		'# Protocol used in this round', protocol.strip(), '',
		'# Debrief of this round', debrief.strip(), '',
		'# Lessons learned in earlier rounds', lessons.strip() or '(none)', '',
		'You are prisoner 1. Write 1 to 3 new, short lessons (one per line, starting with "- ") that would '
		'help the prisoners do better next round. Do not repeat earlier lessons.',
	]
	return [{'role': 'system', 'content': rules(opt)}, {'role': 'user', 'content': '\n'.join(lines)}]


def render_transcript(transcript):
	return '\n\n'.join(f'**Prisoner {k}:** {text.strip()}' for k, text in transcript)


def clean_markdown(text):
	"""去掉模型可能包在外面的 ``` 代码块和思考过程"""
	text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
	m = re.match(r'^```[a-zA-Z]*\n(.*?)\n```\s*$', text, re.S)
	return (m.group(1) if m else text).strip()
