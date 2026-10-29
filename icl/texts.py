"""
ICL 的 prompt 文字（中英文）：所有给模型看的、以及复盘里给人看的固定句子都在这里

icl_lang = "zh"：中文；默认 "en"（3B 的实验都是英文 prompt 跑的，保留英文以便复现）。
JSON 里的动作名（OFF / ON / ANNOUNCE）两种语言都保持英文，解析器不用改。
"""

EN = {
	# ---- 规则说明 ----
	'rules': (
		"You are one of {n} prisoners (numbered 1 to {n}) in a cooperative puzzle.\n"
		"The game lasts {T} days. Each day the warden picks one prisoner uniformly at random and brings "
		"them into a room with a light bulb. The same prisoner can be picked several times, and some "
		"prisoners may never be picked.\n"
		"When you enter the room you see the bulb exactly as the previous day's prisoner left it "
		"(on day 1 it is OFF). Before leaving you either set the bulb ON or OFF for the next prisoner, "
		"or you announce: \"Every prisoner has visited the room.\"\n"
		"If the announcement is correct, everyone wins (reward +1). If it is wrong, everyone loses "
		"(reward -1). If nobody announces within {T} days, the reward is 0.\n"
		"During the game prisoners cannot talk to each other: the bulb is the only signal, and it only "
		"reaches the prisoner who enters the next day. Before the game the prisoners may meet and agree "
		"on a protocol, written down in protocol.md. Each prisoner also keeps a private memory.md."),
	# ---- memory.md ----
	'visits_header': '## Visits (recorded automatically)',
	'visit_line': '- day {d}: the bulb was {seen}, you {did}.',
	'no_visits': '- none: this is your first visit.',
	'notes_header': '## Your notes',
	'notes_empty': '(empty)',
	'did_announce': 'announced that everyone has visited',
	'did_bulb': 'left the bulb {bulb}',
	# ---- 今天的事实 ----
	'fact_bulb': '- The bulb is {bulb}.',
	'fact_first': '- This is your first visit to the room.',
	'fact_not_first': '- This is NOT your first visit{days}.',
	'fact_days': ' (you were here on day {days})',
	'list_sep': ', ',
	'fact_day1': '- Today is day 1, so nobody was in the room yesterday.',
	'fact_yesterday': '- Yesterday (day {y}) you were{neg} the prisoner in the room.',
	'neg': ' NOT',
	# ---- 决策 ----
	'protocol_header': '# protocol.md (agreed by all prisoners before the game)',
	'memory_header': '# memory.md (your private memory; nobody else can read it)',
	'today_header': '# Today',
	'today': 'You are prisoner {a}. Today is day {day} of {T}. You are in the room.',
	'facts_intro': 'Facts you know for sure:',
	'action_note': '"OFF" / "ON" sets the bulb for the next prisoner; "ANNOUNCE" ends the game.',
	'memory_note': ' "memory" replaces your previous notes.',
	'json_with_memory': '{"action": "<OFF, ON or ANNOUNCE>", "memory": "<what you want to remember on later visits, or empty>"}',
	'json_action': '{"action": "<OFF, ON or ANNOUNCE>"}',
	'free': ('Decide what to do by following the protocol. Think step by step briefly, then end your reply '
		'with a JSON object on its own line:'),
	'rule_check': [
		'Apply the protocol: go through its rules in order and use the FIRST rule whose conditions all hold '
		'for these facts.',
		'Reply in exactly this format, with the JSON only at the very end:',
		'FACTS: <the facts above, in one line>',
		'RULE: <quote the rule you apply>',
		'CHECK: <in one line, why each earlier rule does not apply>',
	],
	'conditions': [
		'Apply the protocol by checking its rules one at a time, starting from rule 1. Split each rule into its '
		'separate conditions (one fact each) and write yes or no for each condition, using the facts above. A rule '
		'applies only if all of its conditions are yes. Stop at the first rule that applies and do what it says.',
		'',
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
	],
	'prefill': 'FACTS:',
	# ---- 编译（旧的协议检查） ----
	'situation_header': '# Situation',
	'situation': 'You are prisoner {a}. Imagine you enter the room on day {day} of {T}. You have no private notes.',
	'what_to_do': 'What does the protocol tell you to do?',
	# ---- 商定 / 复盘 ----
	'prev_protocol': '# Protocol used in the previous round',
	'prev_debrief': '# Debrief of the previous round',
	'lessons_so_far': '# Lessons learned so far',
	'requirements': ('The protocol must be a short list of precise rules that a prisoner can apply alone, during the '
		'game, using only: the day number (1 to {T}), the bulb state seen when entering, whether '
		'they were in the room yesterday, their own visit history, and their own private notes.'),
	'meeting': 'The prisoners are meeting before the game to agree on a protocol. You are prisoner {a}.',
	'discussion_so_far': '# Discussion so far',
	'nobody_spoke': '(nobody has spoken yet)',
	'your_turn': ('It is your turn to speak (round {t} of {nt}). Propose a protocol, or point out a concrete '
		'flaw in a proposal and fix it. Reply in under 200 words.'),
	'discussion': '# Discussion',
	'scribe': ('You are prisoner 1 and act as the scribe. Write the final protocol.md that all prisoners '
		'will follow, based on the discussion.'),
	'your_draft': '# Your draft protocol.md',
	'suggestions': '# Suggestions from the other prisoners',
	'suggestion_line': '- prisoner {k}: {text}',
	'revise': 'You are prisoner 1 and act as the scribe. Revise the draft where the suggestions are correct.',
	'output_only': 'Output only the markdown content of protocol.md.',
	'draft_header': '# Draft protocol.md',
	'review': ('You are prisoner {a}. Check whether this protocol can cause a wrong announcement or can be '
		'applied ambiguously. If it is fine, reply exactly AGREE. Otherwise reply with one concrete '
		'suggestion in under 80 words.'),
	'this_protocol': '# Protocol used in this round',
	'this_debrief': '# Debrief of this round',
	'earlier_lessons': '# Lessons learned in earlier rounds',
	'none': '(none)',
	'write_lessons': ('You are prisoner 1. Write 1 to 3 new, short lessons (one per line, starting with "- ") that would '
		'help the prisoners do better next round. Do not repeat earlier lessons.'),
	'current_protocol': '# Current protocol.md',
	'check_header': '# Protocol check',
	'fix': ('You are prisoner 1 and act as the scribe. The protocol did not pass the check. Rewrite protocol.md so '
		'that it fixes the problems found by the check.'),
	'transcript_line': '**Prisoner {k}:** {text}',
	# ---- 复盘报告 ----
	'db_results': 'Results over {bs} games: normalized reward {r:.2f}.',
	'db_win': '- correct announcement (win): {n}',
	'db_lose': '- wrong announcement (lose): {n}',
	'db_missed': '- everyone had visited by day {T} but nobody announced: {n}',
	'db_silent': '- not everyone visited and nobody announced (nothing could be done): {n}',
	'db_game': '## Game {b}: {outcome}',
	'db_step': ('- day {day}: prisoner {a} entered (visited so far: {visited}), saw the bulb {bulb}, '
		'chose {action}. Notes: "{notes}"'),
	'db_out_win': 'WIN: prisoner {a} announced correctly on day {day}',
	'db_out_lose': 'LOSE: prisoner {a} announced on day {day}, but prisoner(s) {missing} had not visited yet',
	'db_out_missed': 'MISSED: everyone had visited, but nobody announced',
	'lessons_after': '## After round {r}',
	# ---- discussion.md ----
	'dl_discussion': '# Discussion',
	'dl_draft': '# Draft protocol (prisoner 1)',
	'dl_reviews': '# Reviews',
	'dl_final': '# Final protocol',
	'dl_same': '(same as the draft: everyone agreed)',
}

ZH = {
	# ---- 规则说明 ----
	'rules': (
		"你是一个合作谜题里的 {n} 名囚犯之一（编号 1 到 {n}）。\n"
		"游戏持续 {T} 天。每天典狱长从所有囚犯中等概率随机选一人，带进一间有一盏灯的房间。"
		"同一个囚犯可能被选中多次，也可能有囚犯一次都没被选中。\n"
		"你进门时看到的灯，就是前一天那个囚犯离开时留下的状态（第 1 天灯是关的）。"
		"离开前你要么把灯设为开或关，留给下一个进来的人；要么宣布：\"所有囚犯都已经进过这个房间了。\"\n"
		"宣布正确，所有人获胜（奖励 +1）；宣布错误，所有人失败（奖励 -1）；{T} 天内没人宣布，奖励为 0。\n"
		"比赛期间囚犯之间不能交谈：灯是唯一的信号，而且只能传给第二天进来的人。"
		"比赛开始前，囚犯们可以开会商定一份协议，写在 protocol.md 里。每个囚犯还有一份自己的私人记忆 memory.md。"),
	# ---- memory.md ----
	'visits_header': '## 访问记录（自动记录）',
	'visit_line': '- 第 {d} 天：进门时灯是{seen}，你{did}。',
	'no_visits': '- 无：这是你第一次来。',
	'notes_header': '## 你的笔记',
	'notes_empty': '（空）',
	'did_announce': '宣布了所有人都来过',
	'did_bulb': '把灯设为{bulb}',
	# ---- 今天的事实 ----
	'fact_bulb': '- 灯是{bulb}的。',
	'fact_first': '- 这是你第一次来这个房间。',
	'fact_not_first': '- 这**不是**你第一次来{days}。',
	'fact_days': '（你在第 {days} 天来过）',
	'list_sep': '、',
	'fact_day1': '- 今天是第 1 天，所以昨天没有人在房间里。',
	'fact_yesterday': '- 昨天（第 {y} 天）在房间里的{neg}是你。',
	'neg': '**不**',
	# ---- 决策 ----
	'protocol_header': '# protocol.md（所有囚犯赛前商定的协议）',
	'memory_header': '# memory.md（你的私人记忆，别人看不到）',
	'today_header': '# 今天',
	'today': '你是 {a} 号囚犯。今天是第 {day} 天（共 {T} 天）。你在房间里。',
	'facts_intro': '你确定知道的事实：',
	'action_note': '"OFF" / "ON" 是把灯设为关 / 开，留给下一个人；"ANNOUNCE" 是宣布，游戏结束。',
	'memory_note': ' "memory" 会替换你之前的笔记。',
	'json_with_memory': '{"action": "<OFF、ON 或 ANNOUNCE>", "memory": "<以后来访时想记住的内容，可以为空>"}',
	'json_action': '{"action": "<OFF、ON 或 ANNOUNCE>"}',
	'free': '按照协议决定怎么做。简短地一步步思考，最后单独一行写一个 JSON：',
	'rule_check': [
		'应用协议：按顺序检查各条规则，使用第一条所有条件都对这些事实成立的规则。',
		'严格按下面的格式回答，JSON 放在最后：',
		'FACTS: <上面的事实，写成一行>',
		'RULE: <引用你使用的那条规则>',
		'CHECK: <用一行说明前面每条规则为什么不适用>',
	],
	'conditions': [
		'应用协议：从规则 1 开始逐条检查。把每条规则拆成各自独立的条件（每个条件只说一件事），'
		'根据上面的事实，对每个条件写"是"或"否"。只有所有条件都是"是"，这条规则才适用。'
		'遇到第一条适用的规则就停下，按它说的做。',
		'',
		'推理示例（另一个无关的任务）。规则：1. 如果在下雨并且你带了伞，就出门。2. 如果在下雨，就待在家里。'
		'事实：在下雨；你没带伞。',
		'规则 1：在下雨 = 是；你带了伞 = 否 -> 不适用',
		'规则 2：在下雨 = 是 -> 适用',
		'DECISION: 规则 2 -> 待在家里',
		'',
		'现在针对你自己的情况，严格按下面的格式回答，JSON 放在最后。把每个 <...> 换成你自己的话，不要照抄 <...>：',
		'FACTS: <上面的事实，写成一行>',
		'规则 1：<第一个条件> = <是 或 否>；<第二个条件，如果有的话> = <是 或 否> -> <适用 或 不适用>',
		'规则 2：<规则 2 同样处理；一直检查到某条规则适用为止>',
		'DECISION: <适用的规则编号> -> <这条规则要求做什么>',
	],
	'prefill': 'FACTS:',
	# ---- 编译（旧的协议检查） ----
	'situation_header': '# 局面',
	'situation': '你是 {a} 号囚犯。假设你在第 {day} 天（共 {T} 天）进入房间。你没有任何私人笔记。',
	'what_to_do': '按照协议，你应该怎么做？',
	# ---- 商定 / 复盘 ----
	'prev_protocol': '# 上一轮使用的协议',
	'prev_debrief': '# 上一轮的复盘',
	'lessons_so_far': '# 目前积累的经验',
	'requirements': ('协议必须是一份简短、精确的规则清单，每个囚犯在比赛中能独立执行，只能用到：'
		'第几天（1 到 {T}）、进门时看到的灯、昨天自己在不在房间、自己的访问记录、自己的私人笔记。'),
	'meeting': '囚犯们在赛前开会，商定一份协议。你是 {a} 号囚犯。',
	'discussion_so_far': '# 目前的讨论',
	'nobody_spoke': '（还没有人发言）',
	'your_turn': '现在轮到你发言（第 {t} 轮，共 {nt} 轮）。提出一份协议，或者指出某个提案的具体漏洞并修正它。200 字以内。',
	'discussion': '# 讨论',
	'scribe': '你是 1 号囚犯，担任书记员。根据讨论，写出所有囚犯都要遵守的最终 protocol.md。',
	'your_draft': '# 你起草的 protocol.md',
	'suggestions': '# 其他囚犯的修改意见',
	'suggestion_line': '- {k} 号囚犯：{text}',
	'revise': '你是 1 号囚犯，担任书记员。在修改意见正确的地方修改草稿。',
	'output_only': '只输出 protocol.md 的 markdown 内容。',
	'draft_header': '# protocol.md 草稿',
	'review': ('你是 {a} 号囚犯。检查这份协议会不会导致错误宣布，或者执行时有没有歧义。'
		'如果没问题，只回复 AGREE。否则用 80 字以内给出一条具体的修改意见。'),
	'this_protocol': '# 本轮使用的协议',
	'this_debrief': '# 本轮的复盘',
	'earlier_lessons': '# 之前几轮积累的经验',
	'none': '（无）',
	'write_lessons': '你是 1 号囚犯。写 1 到 3 条新的、简短的经验（每条一行，以 "- " 开头），帮助大家下一轮做得更好。不要重复之前的经验。',
	'current_protocol': '# 当前的 protocol.md',
	'check_header': '# 协议评估',
	'fix': '你是 1 号囚犯，担任书记员。这份协议没有通过评估。重写 protocol.md，解决评估中发现的问题。',
	'transcript_line': '**{k} 号囚犯：** {text}',
	# ---- 复盘报告 ----
	'db_results': '{bs} 局比赛的结果：归一化奖励 {r:.2f}。',
	'db_win': '- 正确宣布（赢）：{n} 局',
	'db_lose': '- 错误宣布（输）：{n} 局',
	'db_missed': '- 第 {T} 天时所有人都来过了，但没人宣布：{n} 局',
	'db_silent': '- 没有所有人都来过，也没人宣布（本来就赢不了）：{n} 局',
	'db_game': '## 第 {b} 局：{outcome}',
	'db_step': '- 第 {day} 天：{a} 号进门（到这天为止来过的人：{visited}），看到灯是{bulb}，选择 {action}。笔记："{notes}"',
	'db_out_win': '赢：{a} 号在第 {day} 天正确宣布',
	'db_out_lose': '输：{a} 号在第 {day} 天宣布了，但 {missing} 号还没来过',
	'db_out_missed': '错失：所有人都来过了，但没人宣布',
	'lessons_after': '## 第 {r} 轮之后',
	# ---- discussion.md ----
	'dl_discussion': '# 讨论记录',
	'dl_draft': '# 协议草稿（1 号整理）',
	'dl_reviews': '# 审阅意见',
	'dl_final': '# 最终协议',
	'dl_same': '（与草稿相同：所有人都同意）',
}

# 中文里灯的状态写成"开 / 关"，JSON 里的动作名仍用 ON / OFF
BULB = {'en': {0: 'OFF', 1: 'ON'}, 'zh': {0: '关', 1: '开'}}


def lang(opt):
	"""opt 为 None（比如 diagnose 等只读工具里调用）时用英文"""
	return 'zh' if opt is not None and opt.icl_lang == 'zh' else 'en'


def texts(opt):
	return ZH if lang(opt) == 'zh' else EN


def bulb_word(opt, bit):
	return BULB[lang(opt)][int(bit)]
