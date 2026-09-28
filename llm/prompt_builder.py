"""
把 Switch 游戏中某个 agent 的观测和历史转成给 LLM 的文本 prompt

LLM 没有 RNN 隐状态，所以 agent 需要记住的信息（今天是第几天、自己以前哪天进过房间、
当时看到的灯和做的选择）都要显式写进 prompt。

通信保持和原任务一样只有 1 bit：进房间的 agent 只能看到前一天那个人留下的灯泡状态。
"""

# 多选题选项，每个选项只对应一个 token，方便直接比较下一个 token 的 logits
OPTIONS = ['A', 'B', 'C']
OPTION_OFF = 0		# 什么都不做，把灯设为 OFF（通信 bit = 0）
OPTION_ON = 1		# 什么都不做，把灯设为 ON（通信 bit = 1）
OPTION_TELL = 2		# 宣布"所有人都进过房间了"（游戏动作 TELL）


def bulb_str(bit):
	return 'ON' if bit else 'OFF'


def describe_option(option):
	if option == OPTION_TELL:
		return 'announced that everyone has visited'
	return 'left the bulb ' + bulb_str(option)


def system_prompt(opt):
	# 所有 agent 共用同一个 system prompt（不含 agent 编号），这样它的 KV cache 只需算一次
	n, T = opt.game_nagents, opt.nsteps
	text = (
		f"You are one of {n} prisoners in a cooperative puzzle.\n"
		f"The game lasts {T} days. Each day the warden picks one prisoner uniformly at random and brings "
		f"them into a room with a light bulb. The same prisoner can be picked several times, and some "
		f"prisoners may never be picked. Prisoners cannot talk to each other.\n"
		f"When you enter the room you see the bulb exactly as the previous day's prisoner left it "
		f"(on day 1 it is OFF). Before leaving you either set the bulb ON or OFF for the next prisoner, "
		f"or you announce: \"Every prisoner has visited the room.\"\n"
		f"If the announcement is correct, everyone wins (reward +1). If it is wrong, everyone loses "
		f"(reward -1). If nobody announces within {T} days, the reward is 0.\n"
		f"All prisoners think the same way you do, so the bulb can be used as a signal between visitors."
	)
	if opt.llm_strategy_hint:
		text += '\n\nStrategy hint: ' + opt.llm_strategy_hint
	return text


def user_prompt(opt, agent_id, day, bulb, history):
	"""
	agent_id: agent 编号（1 开始）
	day: 今天是第几天（1 开始）
	bulb: 进门时看到的灯泡 bit（前一天的人留下的）
	history: 这个 agent 以前的访问记录 [(day, 看到的bit, 选的option), ...]
	"""
	lines = [
		f"You are prisoner {agent_id}.",
		f"Today is day {day} of {opt.nsteps}. You are in the room and the bulb is {bulb_str(bulb)}."]
	if day > 1:
		# 从自己的历史就能推出来，但小模型不一定会推，所以直接写明
		was_here = bool(history) and history[-1][0] == day - 1
		lines.append(f"Yesterday (day {day - 1}) you were {'' if was_here else 'not '}the prisoner in the room.")
	if history:
		lines.append('Your previous visits:')
		for d, seen, option in history:
			lines.append(f"- day {d}: the bulb was {bulb_str(seen)}, you {describe_option(option)}.")
	else:
		lines.append('This is your first visit to the room.')
	lines += [
		'',
		'Choose one option:',
		'A. Leave the bulb OFF.',
		'B. Leave the bulb ON.',
		'C. Announce that every prisoner has visited the room.',
		'Answer with only the letter.',
	]
	return '\n'.join(lines)


def build_messages(opt, agent_id, day, bulb, history):
	return [
		{'role': 'system', 'content': system_prompt(opt)},
		{'role': 'user', 'content': user_prompt(opt, agent_id, day, bulb, history)},
	]
