"""
ICL 的 LLM 后端：每个 agent 一个独立实例，互相之间不共享参数状态和 KV cache

	vllm：OpenAI 兼容的 HTTP 接口，每个 agent 连自己的 vllm serve（见 icl/launch_vllm.sh）
	deepseek：DeepSeek 的在线 API（OpenAI 兼容）。key 从环境变量 DEEPSEEK_API_KEY 读，不写进配置和仓库
	hf：  transformers 进程内推理（CPU 冒烟，或单卡本机实验）；同一模型的权重只加载一份，见 HFBackend
	rule：不用模型，按 icl/protocols/switch_3_oracle.md 的规则直接给出答案，用来验证整条流程（应得 1.0）

配置 icl_agents 是每个 agent 的后端参数列表（vllm: {"url", "model"}；deepseek: {"model", "url"?}；hf: {"model_path"}）。
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


def read_api_key(env_name):
	"""
	按顺序找 API key：环境变量 → 仓库根目录的 .deepseek_key（已在 .gitignore 里） → ~/.deepseek_key。
	key 不放进配置文件，避免被提交。文件编码自动识别：PowerShell 的 `echo key > 文件` 默认写 UTF-16（带 BOM），
	Linux / Git Bash 写 UTF-8，两种都能读。
	"""
	key = os.environ.get(env_name)
	if key:
		return key.strip()
	for path in (os.path.join(os.getcwd(), '.deepseek_key'), os.path.expanduser('~/.deepseek_key')):
		if os.path.exists(path):
			key = _read_text_any_encoding(path).strip()
			if key:
				print(f'using API key from {path}')
				return key
	raise RuntimeError(f'{env_name} is not set. Export it (export {env_name}=...) or write the key into '
		f'.deepseek_key in the repository root; do not put the key in the config file')


def _read_text_any_encoding(path):
	with open(path, 'rb') as f:
		raw = f.read()
	for enc in ('utf-8-sig', 'utf-16', 'utf-8'):
		try:
			return raw.decode(enc)
		except UnicodeDecodeError:
			continue
	raise RuntimeError(f'cannot decode {path}: expected a UTF-8 or UTF-16 text file')


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


class DeepSeekBackend:
	"""
	DeepSeek 在线 API。API key 只从环境变量 DEEPSEEK_API_KEY 读，不写进配置文件，避免被提交。
	用 /beta 端点：支持 Chat Prefix Completion（最后一条 assistant 消息加 "prefix": true），
	所以和其他后端一样可以预填回复开头。不支持时（接口返回 400）自动退回"要求回复以某段文字开头"。
	"""
	DEFAULT_URL = 'https://api.deepseek.com/beta/chat/completions'
	KEY_ENV = 'DEEPSEEK_API_KEY'

	def __init__(self, spec, opt):
		self.url = (spec.get('url') or self.DEFAULT_URL).rstrip('/')
		self.model = spec.get('model') or 'deepseek-chat'
		self.api_key = read_api_key(self.KEY_ENV)
		self.extra = opt.icl_request_extra or {}
		self.prefix_supported = True		# 第一次遇到 400 后改成用指令代替预填
		self.pool = ThreadPoolExecutor(max_workers=opt.icl_concurrency or 4)	# 在线 API 并发别太高，容易 429
		self.timeout = opt.icl_timeout or 300

	def _messages(self, messages):
		if messages[-1]['role'] != 'assistant' or not self.prefix_supported:
			return messages
		return list(messages[:-1]) + [dict(messages[-1], prefix=True)]

	def _body(self, messages, temperature, max_tokens):
		return dict(self.extra, model=self.model, messages=self._messages(messages),
			temperature=temperature, max_tokens=max_tokens, stream=False)

	def _post(self, messages, temperature, max_tokens):
		data = json.dumps(self._body(messages, temperature, max_tokens)).encode()
		headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}
		for attempt in range(5):
			try:
				req = urllib.request.Request(self.url, data=data, headers=headers)
				with urllib.request.urlopen(req, timeout=self.timeout) as resp:
					msg = json.loads(resp.read())['choices'][0]['message']
					# deepseek-reasoner 把思考过程放在 reasoning_content，正式回答在 content
					return (msg.get('content') or '') or (msg.get('reasoning_content') or '')
			except urllib.error.HTTPError as err:
				body = err.read().decode(errors='replace')
				if err.code == 400 and messages[-1]['role'] == 'assistant' and self.prefix_supported:
					self.prefix_supported = False		# 端点不支持预填：改成在 user 消息里要求开头
					return self._post(self._instruction_fallback(messages, temperature), temperature, max_tokens)
				if err.code in (429, 500, 502, 503, 504) and attempt < 4:	# 限流 / 服务端错误：退避重试
					time.sleep(min(60, 4 * 2 ** attempt))
					continue
				raise RuntimeError(f'{self.url} returned {err.code}: {body}') from err
			except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
				if attempt == 4:
					raise RuntimeError(f'request to {self.url} failed: {err}') from err
				time.sleep(min(60, 4 * 2 ** attempt))

	@staticmethod
	def _instruction_fallback(messages, temperature):
		prefill = messages[-1]['content']
		messages = [dict(m) for m in messages[:-1]]
		messages[-1]['content'] += f'\nYour reply must start with exactly: {prefill}'
		return messages

	def chat(self, requests, temperature, max_tokens):
		futures = [self.pool.submit(self._post, r['messages'], temperature, max_tokens) for r in requests]
		return [f.result() for f in futures]


class HFBackend:
	"""
	同一个 model_path 的权重在进程里只加载一份（单卡 12 GB 放不下 3 份 3B）。
	这不破坏 agent 之间的独立性：权重只读，每次 generate 都新建 KV cache、调用之间不留状态，
	而且 ICLArena.query 对 hf 按 agent 分别调用，不同 agent 的请求不会出现在同一个 batch 里。
	"""
	_shared = {}		# (model_path, device) -> (tokenizer, model)

	def __init__(self, model_path, opt):
		import torch
		from transformers import AutoModelForCausalLM, AutoTokenizer
		self.torch = torch
		self.device = 'cuda' if opt.cuda and torch.cuda.is_available() else 'cpu'
		key = (os.path.abspath(model_path), self.device)
		if key not in HFBackend._shared:
			tokenizer = AutoTokenizer.from_pretrained(model_path)
			tokenizer.padding_side = 'left'		# 生成时左填充，新 token 接在每行末尾
			# 默认 GPU 用 bfloat16、CPU 用 float32；CPU 内存不够时可设 icl_hf_dtype: "bfloat16"（3B 约 6GB）
			dtype = getattr(torch, opt.icl_hf_dtype) if opt.icl_hf_dtype else (
				torch.bfloat16 if self.device == 'cuda' else torch.float32)
			model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(self.device).eval()
			HFBackend._shared[key] = (tokenizer, model)
		self.tokenizer, self.model = HFBackend._shared[key]
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
		'deepseek': lambda s: DeepSeekBackend(s, opt),
		'hf': lambda s: HFBackend(s['model_path'], opt),
	}[kind]
	instances = [make(s) for s in specs]
	return [None] + [instances[0] if len(specs) == 1 else instances[i] for i in range(n)]
