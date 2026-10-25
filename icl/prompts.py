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

# 决策回复的格式（icl_decision_format）：
#	conditions（默认）：逐条规则、逐个条件写 yes/no，所有条件都是 yes 的第一条规则才适用，最后给 JSON
#	rule_check：FACTS → RULE（引用一条规则）→ CHECK（前面的规则为什么不适用）→ JSON（第二版）
#	free：自由推理后给 JSON，不预填（第一版的做法）
# 第一版 3B 会先给答案再补理由、跳过规则 1；第二版 rule_check 下又只匹配合取条件的一半
# （"第一次来"就宣布，不看灯是不是 ON），CHECK 也不核对所选规则自己的条件。
# 所以 conditions 要求把每条规则的每个条件都对照事实写出来。
# 第一版 conditions 模板里的 "yes/no"、"applies / does not apply" 没放进 <...>，3B 原样照抄、一个 yes/no 都没填，
# 所以占位全部改成 <...>，并附一个和本任务无关的填好的示例（下雨 / 带伞），最后用 DECISION 行把规则对应到动作。
# 预填 DECISION_PREFILL 保证先推理后作答（icl_prefill=false 可关闭；free 格式不预填）
DECISION_FORMATS = ('conditions', 'rule_check', 'free')
DECISION_PREFILL = 'FACTS:'
# 模板里的占位说明，模型照抄时当作空笔记（E1 中 34% 的笔记是照抄的 "notes for your future visits"）
NOTE_PLACEHOLDERS = ('what you want to remember', 'notes for your future visits', 'or empty')


def render_memory(facts, notes, notes_enabled=True):
	"""
	一个 agent 的 memory.md
	facts: 程序记录的客观访问历史 [(day, 看到的bit, option), ...]
	notes: agent 自己写的笔记（每次决策时覆盖）；notes_enabled=false 时不显示（消融：只靠客观记录）
	"""
	lines = ['## Visits (recorded automatically)']
	if facts:
		lines += [f"- day {d}: the bulb was {bulb_str(seen)}, you {describe_option(o)}." for d, seen, o in facts]
	else:
		lines.append('- none: this is your first visit.')
	if notes_enabled:
		lines += ['', '## Your notes', notes.strip() if notes and notes.strip() else '(empty)']
	return '\n'.join(lines)


def fact_lines(day, bulb, first_visit, here_yesterday, prev_days=None):
	"""今天确定知道的事实。都能从访问记录推出来，但直接写明，避免小模型读错（E1 里常把第一次来访说成不是）"""
	lines = [f'- The bulb is {bulb_str(bulb)}.']
	if first_visit:
		lines.append('- This is your first visit to the room.')
	else:
		days = f' (you were here on day {", ".join(map(str, prev_days))})' if prev_days else ''
		lines.append(f'- This is NOT your first visit{days}.')
	if day == 1:
		lines.append('- Today is day 1, so nobody was in the room yesterday.')
	else:
		lines.append(f'- Yesterday (day {day - 1}) you were{"" if here_yesterday else " NOT"} the prisoner in the room.')
	return lines


def decision_format(opt):
	fmt = opt.icl_decision_format or 'conditions'
	assert fmt in DECISION_FORMATS, f'icl_decision_format must be one of {DECISION_FORMATS}'
	return fmt


def _apply_instructions(opt, json_line):
	fmt = decision_format(opt)
	if fmt == 'free':
		return ['Decide what to do by following the protocol. Think step by step briefly, then end your reply '
			'with a JSON object on its own line:', json_line]
	if fmt == 'rule_check':
		return [
			'Apply the protocol: go through its rules in order and use the FIRST rule whose conditions all hold '
			'for these facts.',
			'Reply in exactly this format, with the JSON only at the very end:',
			'FACTS: <the facts above, in one line>',
			'RULE: <quote the rule you apply>',
			'CHECK: <in one line, why each earlier rule does not apply>',
			json_line,
		]
	return [
		'Apply the protocol by checking its rules one at a time, starting from rule 1. Split each rule into its '
		'separate conditions (one fact each) and write yes or no for each condition, using the facts above. A rule '
		'applies only if all of its conditions are yes. Stop at the first rule that applies and do what it says.',
		'',
		# 示例用和本任务无关的规则，只演示格式和"合取条件有一个 no 就不适用"，不暗示任何策略。
		# 放在格式说明前面，这样 prompt 的最后是以 JSON 结尾的格式，模型不会学着示例在 DECISION 行就停下
		'Example of the reasoning, for a different task. Rules: 1. If it is raining and you have an umbrella, go '
		'outside. 2. If it is raining, stay inside. Facts: it is raining; you have no umbrella.',
		'Rule 1: it is raining = yes; you have an umbrella = no -> does not apply',
		'Rule 2: it is raining = yes -> applies',
		'DECISION: rule 2 -> stay inside',
		'',
		'Now reply for your own situation in exactly this format, with the JSON only at the very end. Replace every '
		'<...> with your own words; never copy a <...> placeholder:',
		'FACTS: <the facts above, in one line>',
		'Rule 1: <first condition> = <yes or no>; <second condition, if the rule has one> = <yes or no> '
		'-> <applies or does not apply>',
		'Rule 2: <the same for rule 2; continue until a rule applies>',
		'DECISION: <number of the rule that applies> -> <what that rule says to do>',
		json_line,
	]


def build_decision_messages(opt, agent_id, day, bulb, facts, notes, protocol):
	notes_enabled = opt.icl_notes is not False
	first_visit = not facts
	here_yesterday = bool(facts) and facts[-1][0] == day - 1
	json_line = ('{"action": "<OFF, ON or ANNOUNCE>", "memory": "<what you want to remember on later visits, or empty>"}'
		if notes_enabled else '{"action": "<OFF, ON or ANNOUNCE>"}')
	lines = [
		'# protocol.md (agreed by all prisoners before the game)',
		protocol.strip(),
		'',
		'# memory.md (your private memory; nobody else can read it)',
		render_memory(facts, notes, notes_enabled),
		'',
		'# Today',
		f"You are prisoner {agent_id}. Today is day {day} of {opt.nsteps}. You are in the room.",
		'Facts you know for sure:',
		*fact_lines(day, bulb, first_visit, here_yesterday, [d for d, _, _ in facts]),
		'',
		'"OFF" / "ON" sets the bulb for the next prisoner; "ANNOUNCE" ends the game.'
		+ (' "memory" replaces your previous notes.' if notes_enabled else ''),
		*_apply_instructions(opt, json_line),
	]
	return _with_prefill(opt, [{'role': 'system', 'content': rules(opt)}, {'role': 'user', 'content': '\n'.join(lines)}],
		DECISION_PREFILL)


def _with_prefill(opt, messages, prefill):
	"""在 assistant 回复开头预填固定文字，后端会接着它往下生成（vllm: continue_final_message）"""
	if opt.icl_prefill is False or decision_format(opt) == 'free':
		return messages
	return messages + [{'role': 'assistant', 'content': prefill}]


def prefill_of(messages):
	return messages[-1]['content'] if messages[-1]['role'] == 'assistant' else ''


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
				return option, _clean_notes(obj.get('memory', ''))
	m = re.findall(r'"action"\s*:\s*"?([A-Za-z]+)', text)
	if m:
		option = _to_option(m[-1])
		if option is not None:
			notes = re.findall(r'"memory"\s*:\s*"(.*?)"\s*\}', text, re.S)
			return option, _clean_notes(notes[-1] if notes else '')
	# conditions 格式没写 JSON、停在 DECISION 行时：从 DECISION 行读动作（优先看 "->" 后面的部分）
	lines = re.findall(r'^\s*DECISION:(.*)$', text, re.M)
	if lines:
		line = lines[-1]
		for part in (line.rsplit('->', 1)[-1], line):
			option = _action_in_words(part)
			if option is not None:
				return option, ''
	return None, ''


def _action_in_words(text):
	if re.search(r'announc', text, re.I):
		return OPTION_TELL
	if re.search(r'\bON\b', text, re.I):
		return OPTION_ON
	if re.search(r'\bOFF\b', text, re.I):
		return OPTION_OFF
	return None


def _clean_notes(notes):
	notes = str(notes).strip()
	if (notes.startswith('<') and notes.endswith('>')) or any(p in notes.lower() for p in NOTE_PLACEHOLDERS):
		return ''
	return notes


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


# ---------------- 协议检查（E2 / E3 赛前） ----------------

def build_compile_messages(opt, protocol, situation, agent_id):
	"""
	把协议"编译"成决策表的一项：给定一个抽象局面，问协议规定做什么
	和比赛中的决策用同样的事实描述和回复格式，但没有访问记录和笔记，只考察对协议本身的理解。
	带上编号和比赛中一致；同一个模型、贪心解码时，prompt 完全相同会让"理解不一致"永远是 0
	"""
	s = situation
	lines = [
		'# protocol.md (agreed by all prisoners before the game)',
		protocol.strip(),
		'',
		'# Situation',
		f"You are prisoner {agent_id}. Imagine you enter the room on day {s['day']} of {opt.nsteps}. "
		f"You have no private notes.",
		'Facts you know for sure:',
		*fact_lines(s['day'], s['bulb'], s['first'], s['yesterday']),
		'',
		'What does the protocol tell you to do?',
		*_apply_instructions(opt, '{"action": "<OFF, ON or ANNOUNCE>"}'),
	]
	return _with_prefill(opt, [{'role': 'system', 'content': rules(opt)}, {'role': 'user', 'content': '\n'.join(lines)}],
		DECISION_PREFILL)


def build_fix_messages(opt, protocol, check_report, context=None):
	"""协议没通过检查：书记员（1 号）看检查报告改协议"""
	lines = _context_block(context) + [
		'# Current protocol.md', protocol.strip(), '',
		'# Protocol check', check_report.strip(), '',
		'You are prisoner 1 and act as the scribe. The protocol did not pass the check. Rewrite protocol.md so '
		'that it fixes the problems found by the check.',
		_protocol_requirements(opt),
		'Output only the markdown content of protocol.md.',
	]
	return [{'role': 'system', 'content': rules(opt)}, {'role': 'user', 'content': '\n'.join(lines)}]


def render_transcript(transcript):
	return '\n\n'.join(f'**Prisoner {k}:** {text.strip()}' for k, text in transcript)


def clean_markdown(text):
	"""去掉模型可能包在外面的 ``` 代码块和思考过程"""
	text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
	m = re.match(r'^```[a-zA-Z]*\n(.*?)\n```\s*$', text, re.S)
	return (m.group(1) if m else text).strip()
