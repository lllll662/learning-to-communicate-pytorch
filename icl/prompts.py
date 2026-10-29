"""
上下文学习（ICL）版本的 prompt：规则说明、比赛中的决策、赛前商定协议、赛后复盘

信息隔离：比赛中一个 agent 的 prompt 只由 protocol.md（所有人共享，赛前确定）、
它自己的 memory.md、今天的观测三部分组成，build_decision_messages 的参数里没有其他 agent 的任何信息。
自然语言交流只发生在比赛外（商定协议、复盘）。
"""
import json
import re

from llm.prompt_builder import OPTION_OFF, OPTION_ON, OPTION_TELL
from icl.texts import texts, bulb_word

# 决策输出里 action 字段的取值 → 游戏选项
ACTIONS = {'OFF': OPTION_OFF, 'ON': OPTION_ON, 'ANNOUNCE': OPTION_TELL}
ACTION_ALIASES = {'TELL': 'ANNOUNCE', 'C': 'ANNOUNCE', 'A': 'OFF', 'B': 'ON'}


def rules(opt):
	"""所有阶段共用的规则说明（和 llm.prompt_builder.system_prompt 相同的游戏，但每个囚犯是独立的个体）"""
	return texts(opt)['rules'].format(n=opt.game_nagents, T=opt.nsteps)


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
NOTE_PLACEHOLDERS = ('what you want to remember', 'notes for your future visits', 'or empty', '以后来访时想记住', '可以为空')


def _did(opt, option):
	t = texts(opt)
	return t['did_announce'] if option == OPTION_TELL else t['did_bulb'].format(bulb=bulb_word(opt, option))


def render_memory(facts, notes, notes_enabled=True, opt=None):
	"""
	一个 agent 的 memory.md
	facts: 程序记录的客观访问历史 [(day, 看到的bit, option), ...]
	notes: agent 自己写的笔记（每次决策时覆盖）；notes_enabled=false 时不显示（消融：只靠客观记录）
	"""
	t = texts(opt)
	lines = [t['visits_header']]
	if facts:
		lines += [t['visit_line'].format(d=d, seen=bulb_word(opt, seen), did=_did(opt, o)) for d, seen, o in facts]
	else:
		lines.append(t['no_visits'])
	if notes_enabled:
		lines += ['', t['notes_header'], notes.strip() if notes and notes.strip() else t['notes_empty']]
	return '\n'.join(lines)


def fact_lines(day, bulb, first_visit, here_yesterday, prev_days=None, opt=None):
	"""今天确定知道的事实。都能从访问记录推出来，但直接写明，避免小模型读错（E1 里常把第一次来访说成不是）"""
	t = texts(opt)
	lines = [t['fact_bulb'].format(bulb=bulb_word(opt, bulb))]
	if first_visit:
		lines.append(t['fact_first'])
	else:
		days = t['fact_days'].format(days=t['list_sep'].join(map(str, prev_days))) if prev_days else ''
		lines.append(t['fact_not_first'].format(days=days))
	if day == 1:
		lines.append(t['fact_day1'])
	else:
		lines.append(t['fact_yesterday'].format(y=day - 1, neg='' if here_yesterday else t['neg']))
	return lines


def decision_format(opt):
	fmt = opt.icl_decision_format or 'conditions'
	assert fmt in DECISION_FORMATS, f'icl_decision_format must be one of {DECISION_FORMATS}'
	return fmt


def _apply_instructions(opt, json_line):
	fmt = decision_format(opt)
	t = texts(opt)
	if fmt == 'free':
		return [t['free'], json_line]
	# conditions 的示例用和本任务无关的规则（下雨 / 带伞），只演示格式和"合取条件有一个 no 就不适用"，不暗示任何策略；
	# 示例放在格式说明前面，这样 prompt 的最后是以 JSON 结尾的格式，模型不会学着示例在 DECISION 行就停下
	return list(t[fmt]) + [json_line]


def build_decision_messages(opt, agent_id, day, bulb, facts, notes, protocol):
	notes_enabled = opt.icl_notes is not False
	first_visit = not facts
	here_yesterday = bool(facts) and facts[-1][0] == day - 1
	t = texts(opt)
	json_line = t['json_with_memory'] if notes_enabled else t['json_action']
	lines = [
		t['protocol_header'],
		protocol.strip(),
		'',
		t['memory_header'],
		render_memory(facts, notes, notes_enabled, opt),
		'',
		t['today_header'],
		t['today'].format(a=agent_id, day=day, T=opt.nsteps),
		t['facts_intro'],
		*fact_lines(day, bulb, first_visit, here_yesterday, [d for d, _, _ in facts], opt),
		'',
		t['action_note'] + (t['memory_note'] if notes_enabled else ''),
		*_apply_instructions(opt, json_line),
	]
	return _with_prefill(opt, [{'role': 'system', 'content': rules(opt)}, {'role': 'user', 'content': '\n'.join(lines)}],
		t['prefill'])


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
	# 中文 prompt 下 DECISION 行可能写"宣布 / 开灯 / 关灯"
	if re.search(r'announc|宣布', text, re.I):
		return OPTION_TELL
	if re.search(r'\bON\b|开灯|设为开|保持开|打开', text, re.I):
		return OPTION_ON
	if re.search(r'\bOFF\b|关灯|设为关|保持关|关闭|关掉', text, re.I):
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

def _context_block(context, opt=None):
	"""E3：上一轮的协议、复盘和累积的经验"""
	if not context:
		return []
	t = texts(opt)
	lines = []
	if context.get('protocol'):
		lines += [t['prev_protocol'], context['protocol'].strip(), '']
	if context.get('debrief'):
		lines += [t['prev_debrief'], context['debrief'].strip(), '']
	if context.get('lessons'):
		lines += [t['lessons_so_far'], context['lessons'].strip(), '']
	return lines


def _protocol_requirements(opt):
	return texts(opt)['requirements'].format(T=opt.nsteps)


def build_discussion_messages(opt, agent_id, turn, nturns, transcript, context=None):
	t = texts(opt)
	lines = _context_block(context, opt) + [
		t['meeting'].format(a=agent_id),
		_protocol_requirements(opt),
		'',
		t['discussion_so_far'],
		render_transcript(transcript, opt) or t['nobody_spoke'],
		'',
		t['your_turn'].format(t=turn + 1, nt=nturns),
	]
	return [{'role': 'system', 'content': rules(opt)}, {'role': 'user', 'content': '\n'.join(lines)}]


def build_draft_messages(opt, transcript, context=None, draft=None, reviews=None):
	"""1 号整理协议；带 draft/reviews 时是按修改意见改稿"""
	t = texts(opt)
	lines = _context_block(context, opt) + [t['discussion'], render_transcript(transcript, opt), '']
	if draft is None:
		lines.append(t['scribe'])
	else:
		lines += [t['your_draft'], draft.strip(), '', t['suggestions'],
			'\n'.join(t['suggestion_line'].format(k=k, text=r.strip()) for k, r in reviews), '', t['revise']]
	lines += [_protocol_requirements(opt), t['output_only']]
	return [{'role': 'system', 'content': rules(opt)}, {'role': 'user', 'content': '\n'.join(lines)}]


def build_review_messages(opt, agent_id, draft):
	t = texts(opt)
	lines = [t['draft_header'], draft.strip(), '', t['review'].format(a=agent_id)]
	return [{'role': 'system', 'content': rules(opt)}, {'role': 'user', 'content': '\n'.join(lines)}]


def is_agree(text):
	return text.strip().strip('.').upper().startswith('AGREE')


def build_lessons_messages(opt, protocol, debrief, lessons):
	t = texts(opt)
	lines = [
		t['this_protocol'], protocol.strip(), '',
		t['this_debrief'], debrief.strip(), '',
		t['earlier_lessons'], lessons.strip() or t['none'], '',
		t['write_lessons'],
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
	t = texts(opt)
	lines = [
		t['protocol_header'],
		protocol.strip(),
		'',
		t['situation_header'],
		t['situation'].format(a=agent_id, day=s['day'], T=opt.nsteps),
		t['facts_intro'],
		*fact_lines(s['day'], s['bulb'], s['first'], s['yesterday'], opt=opt),
		'',
		t['what_to_do'],
		*_apply_instructions(opt, t['json_action']),
	]
	return _with_prefill(opt, [{'role': 'system', 'content': rules(opt)}, {'role': 'user', 'content': '\n'.join(lines)}],
		t['prefill'])


def build_fix_messages(opt, protocol, check_report, context=None):
	"""协议没通过检查：书记员（1 号）看检查报告改协议"""
	t = texts(opt)
	lines = _context_block(context, opt) + [
		t['current_protocol'], protocol.strip(), '',
		t['check_header'], check_report.strip(), '',
		t['fix'],
		_protocol_requirements(opt),
		t['output_only'],
	]
	return [{'role': 'system', 'content': rules(opt)}, {'role': 'user', 'content': '\n'.join(lines)}]


def render_transcript(transcript, opt=None):
	return '\n\n'.join(texts(opt)['transcript_line'].format(k=k, text=text.strip()) for k, text in transcript)


def clean_markdown(text):
	"""去掉模型可能包在外面的 ``` 代码块和思考过程"""
	text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
	m = re.match(r'^```[a-zA-Z]*\n(.*?)\n```\s*$', text, re.S)
	return (m.group(1) if m else text).strip()
