"""
LLM 策略：所有 agent 共享同一个 LLM（类似 model_know_share=true），用 prompt 里的编号区分 agent

不做自由生成，而是把决策当成多选题：只看 assistant 回复第一个 token 在 A/B/C 上的 logits。
这样不会出现解析失败，而且得到的是一个概率分布，之后做 RL 微调时可以直接拿来算 log-prob。
"""
import copy

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm.prompt_builder import OPTIONS


class LLMPolicy:
	def __init__(self, opt):
		self.opt = opt
		self.device = 'cuda' if opt.cuda and torch.cuda.is_available() else 'cpu'
		dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32

		self.tokenizer = AutoTokenizer.from_pretrained(opt.llm_model_path)
		# 右填充：后缀接在共享前缀的 KV cache 之后，右填充保证真实 token 的位置编码连续
		self.tokenizer.padding_side = 'right'
		self.model = AutoModelForCausalLM.from_pretrained(opt.llm_model_path, dtype=dtype).to(self.device)
		self.model.eval()

		# 每个选项字母对应的 token id
		self.option_ids = []
		for option in OPTIONS:
			ids = self.tokenizer.encode(option, add_special_tokens=False)
			assert len(ids) == 1, f'option {option} is not a single token'
			self.option_ids.append(ids[0])

		# system prompt 的 KV cache：所有 prompt 共用同一个 system prompt，只算一次
		self.prefix_text = None
		self.prefix_cache = None

	def _split(self, messages):
		"""把一条 chat prompt 切成 共享前缀(system) + 各自的后缀(user + assistant 开头)"""
		text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
		prefix = self.tokenizer.apply_chat_template(messages[:1], tokenize=False)
		assert text.startswith(prefix)
		return prefix, text[len(prefix):]

	def _get_prefix_cache(self, prefix):
		if prefix != self.prefix_text:
			ids = self.tokenizer(prefix, return_tensors='pt', add_special_tokens=False).input_ids.to(self.device)
			self.prefix_cache = self.model(input_ids=ids, use_cache=True).past_key_values
			self.prefix_text = prefix
			self.prefix_len = ids.shape[1]
		return self.prefix_cache

	@torch.inference_mode()
	def option_logits(self, messages_list):
		"""
		messages_list: 每局游戏一份 chat messages
		返回: [len(messages_list), len(OPTIONS)] 每个选项的 logits
		"""
		parts = [self._split(m) for m in messages_list]
		prefix = parts[0][0]
		assert all(p == prefix for p, _ in parts), 'all prompts must share the same system prompt'
		prefix_cache = self._get_prefix_cache(prefix)
		# 选项字母对应的 lm_head 行：只算这几个 token 的 logits，不算整个词表
		option_head = self.model.lm_head.weight[self.option_ids]

		suffixes = [s for _, s in parts]
		chunk = self.opt.llm_forward_bs or len(suffixes)
		logits = []
		for i in range(0, len(suffixes), chunk):
			# 后缀右填充：前缀 cache 后面紧接真实 token，位置编码不受 padding 影响
			inputs = self.tokenizer(suffixes[i:i + chunk], return_tensors='pt', padding=True,
				add_special_tokens=False).to(self.device)
			n = inputs.input_ids.shape[0]
			cache = copy.deepcopy(prefix_cache)		# forward 会原地往 cache 里追加，所以每次复制
			cache.batch_repeat_interleave(n)
			mask = torch.cat([torch.ones(n, self.prefix_len, dtype=inputs.attention_mask.dtype,
				device=self.device), inputs.attention_mask], dim=1)
			hidden = self.model.model(input_ids=inputs.input_ids, attention_mask=mask,
				past_key_values=cache, use_cache=True).last_hidden_state
			# 取每条后缀最后一个真实 token 的隐状态来预测回答的第一个 token
			last = inputs.attention_mask.sum(1) - 1
			h = hidden[torch.arange(n, device=self.device), last]
			logits.append((h @ option_head.T).float().cpu())
		return torch.cat(logits)

	def select(self, logits, sample=False):
		"""
		sample=True：按 softmax(logits / temperature) 采样（探索，训练用）
		sample=False：取概率最大的选项（贪心，测试用）
		"""
		if sample:
			probs = F.softmax(logits / (self.opt.llm_temperature or 1.0), dim=-1)
			return torch.multinomial(probs, 1).squeeze(1)
		return logits.argmax(dim=-1)
