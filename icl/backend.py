"""
ICL 的 LLM 后端：每个 agent 一个独立实例，互相之间不共享参数状态和 KV cache

	vllm：OpenAI 兼容的 HTTP 接口，每个 agent 连自己的 vllm serve（见 icl/launch_vllm.sh）
	hf：  transformers 在进程内为每个 agent 各加载一份模型，CPU 冒烟测试用
	rule：不用模型，按 icl/protocols/switch_3_oracle.md 的规则直接给出答案，用来验证整条流程（应得 1.0）

配置 icl_agents 是每个 agent 的后端参数列表（vllm: {"url", "model"}；hf: {"model_path"}）。
只写 1 项时所有 agent 共用同一个实例，用作"共用一个 LLM"的消融。

统一接口：backend.chat(requests, temperature, max_tokens) -> [回复文本]
requests 里每项是 {'messages': chat messages, 'meta': 结构化信息}，meta 只有 rule 后端会用。
"""
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from llm.prompt_builder import OPTION_OFF, OPTION_ON, OPTION_TELL
from icl.prompts import ACTIONS

ORACLE_PROTOCOL = os.path.join(os.path.dirname(__file__), 'protocols', 'switch_3_oracle.md')


class VLLMBackend:
	def __init__(self, url, model, opt):
		self.url = url.rstrip('/') + '/chat/completions'
		self.model = model
		self.extra = opt.icl_request_extra or {}	# 比如 {"chat_template_kwargs": {"enable_thinking": false}}
		self.pool = ThreadPoolExecutor(max_workers=opt.icl_concurrency or 32)
		self.timeout = opt.icl_timeout or 300

	def _post(self, messages, temperature, max_tokens):
		payload = dict(self.extra, model=self.model, messages=messages,
			temperature=temperature, max_tokens=max_tokens)
		if messages[-1]['role'] == 'assistant':		# 预填了回复开头：接着它往下生成，而不是另起一条回复
			payload.update(add_generation_prompt=False, continue_final_message=True)
		data = json.dumps(payload).encode()
		for attempt in range(4):
			try:
				req = urllib.request.Request(self.url, data=data, headers={'Content-Type': 'application/json'})
				with urllib.request.urlopen(req, timeout=self.timeout) as resp:
					return json.loads(resp.read())['choices'][0]['message']['content'] or ''
			except urllib.error.HTTPError as err:
				if err.code < 500:		# 请求本身有问题（模型名、上下文过长等），重试没用
					raise RuntimeError(f'{self.url} returned {err.code}: {err.read().decode(errors="replace")}') from err
				if attempt == 3:
					raise RuntimeError(f'request to {self.url} failed: {err}') from err
				time.sleep(2 ** attempt)
			except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
				if attempt == 3:
					raise RuntimeError(f'request to {self.url} failed: {err}') from err
				time.sleep(2 ** attempt)

	def chat(self, requests, temperature, max_tokens):
		futures = [self.pool.submit(self._post, r['messages'], temperature, max_tokens) for r in requests]
		return [f.result() for f in futures]


class HFBackend:
	def __init__(self, model_path, opt):
		import torch
		from transformers import AutoModelForCausalLM, AutoTokenizer
		self.torch = torch
		self.device = 'cuda' if opt.cuda and torch.cuda.is_available() else 'cpu'
		self.tokenizer = AutoTokenizer.from_pretrained(model_path)
		self.tokenizer.padding_side = 'left'		# 生成时左填充，新 token 接在每行末尾
		dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
		self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(self.device).eval()
		self.chunk = opt.icl_hf_batch or 16		# 每次 generate 的条数，显存不够时调小

	def _prompt(self, messages):
		cont = messages[-1]['role'] == 'assistant'		# 预填了回复开头：接着它往下生成
		return self.tokenizer.apply_chat_template(messages, tokenize=False,
			add_generation_prompt=not cont, continue_final_message=cont)

	def chat(self, requests, temperature, max_tokens):
		texts = [self._prompt(r['messages']) for r in requests]
		kw = dict(do_sample=True, temperature=temperature) if temperature > 0 else dict(do_sample=False)
		replies = []
		for i in range(0, len(texts), self.chunk):
			inputs = self.tokenizer(texts[i:i + self.chunk], return_tensors='pt', padding=True,
				add_special_tokens=False).to(self.device)
			with self.torch.inference_mode():
				out = self.model.generate(**inputs, max_new_tokens=max_tokens,
					pad_token_id=self.tokenizer.pad_token_id, **kw)
			replies += self.tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
		return replies


class RuleBackend:
	"""按 oracle 协议作答的假 LLM：输出格式和真模型一样，走同样的解析流程"""
	def __init__(self, opt):
		with open(ORACLE_PROTOCOL, encoding='utf-8') as f:
			self.protocol = f.read()

	def chat(self, requests, temperature, max_tokens):
		return [self._answer(r['meta']) for r in requests]

	def _answer(self, meta):
		phase = meta['phase']
		if phase in ('decide', 'compile'):
			if phase == 'compile':		# 抽象局面：构造一段和它一致的访问记录
				from icl.protocol_check import situation_facts
				meta = dict(meta, facts=situation_facts(meta))
			option = oracle_option(meta['day'], meta['bulb'], meta['facts'])
			action = {v: k for k, v in ACTIONS.items()}[option]
			out = {'action': action}
			if phase == 'decide':
				out['memory'] = f"I have visited {len(meta['facts']) + 1} time(s); last time I chose {action}."
			# 和真模型一样，接在预填的 "FACTS:" 后面
			return f" (as given)\nRULE: oracle rule\nCHECK: earlier rules do not apply\n{json.dumps(out)}"
		if phase in ('draft', 'fix'):
			return self.protocol
		if phase == 'review':
			return 'AGREE'
		if phase == 'lessons':
			return '- Keep following the protocol exactly.'
		return 'I propose the protocol where bulb ON means at least two different prisoners have visited.'


def oracle_option(day, bulb, facts):
	"""附录 B 的协议：灯 ON 表示"至少 2 个不同的人来过" """
	first_visit = not facts
	was_here_yesterday = bool(facts) and facts[-1][0] == day - 1
	if bulb and first_visit:
		return OPTION_TELL
	if bulb or (day > 1 and not was_here_yesterday):
		return OPTION_ON
	return OPTION_OFF


def make_backends(opt):
	"""返回 [None, agent1 的后端, ..., agentN 的后端]（1-indexed，和原代码的 agents 列表一致）"""
	n = opt.game_nagents
	kind = opt.icl_backend
	if kind == 'rule':
		return [None] + [RuleBackend(opt) for _ in range(n)]
	specs = opt.icl_agents or []
	assert len(specs) in (1, n), f'icl_agents needs 1 (shared) or {n} (one per agent) entries'
	if len(specs) == 1:
		print('WARNING: all agents share one backend instance (icl_agents has 1 entry)')
	make = {
		'vllm': lambda s: VLLMBackend(s['url'], s['model'], opt),
		'hf': lambda s: HFBackend(s['model_path'], opt),
	}[kind]
	instances = [make(s) for s in specs]
	return [None] + [instances[0] if len(specs) == 1 else instances[i] for i in range(n)]
