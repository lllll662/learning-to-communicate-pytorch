"""
LLM 策略：所有 agent 共享同一个 LLM（类似 model_know_share=true），用 prompt 里的编号区分 agent

不做自由生成，而是把决策当成多选题：只看 assistant 回复第一个 token 在 A/B/C 上的 logits。
这样不会出现解析失败，而且得到的是一个概率分布，可以直接用来算策略梯度的 log π。

训练（llm_train=true）：LoRA + REINFORCE（组内基线）
	- 只训练 LoRA 参数，原模型冻结
	- 探索：π_mix = (1-ε)·softmax(logits/T) + ε/3，采样和算 log π 都用 π_mix
	- 回报：每局最终团队奖励 R_b（γ=1、奖励只在结尾），优势 A_b = R_b - mean(R)
	- loss = -Σ A_b · log π_mix(a|prompt) / 决策数
"""
import copy

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model

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

		# LoRA：peft 会把 LoRA 层原地注入 self.model，所以下面 self.model.model(...) 的前向自动包含 LoRA
		self.peft_model = None
		if opt.llm_lora_path:
			self.peft_model = PeftModel.from_pretrained(self.model, opt.llm_lora_path, is_trainable=bool(opt.llm_train))
		elif opt.llm_train:
			self.peft_model = get_peft_model(self.model, LoraConfig(
				r=opt.llm_lora_r or 8, lora_alpha=2 * (opt.llm_lora_r or 8), lora_dropout=0.0,
				target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']))
		if opt.llm_train:
			self.params = [p for p in self.model.parameters() if p.requires_grad]
			self.optimizer = torch.optim.AdamW(self.params, lr=opt.learningrate)

		# 每个选项字母对应的 token id
		self.option_ids = []
		for option in OPTIONS:
			ids = self.tokenizer.encode(option, add_special_tokens=False)
			assert len(ids) == 1, f'option {option} is not a single token'
			self.option_ids.append(ids[0])

		# system prompt 的 KV cache：所有 prompt 共用同一个 system prompt，只算一次
		# 注意：LoRA 参数更新后 KV 会变，learn_from_episode 里会把它清掉重算
		self.prefix_text = None
		self.prefix_cache = None

	def _split(self, messages_list):
		"""把每条 chat prompt 切成 共享前缀(system) + 各自的后缀(user + assistant 开头)"""
		prefix = None
		suffixes = []
		for messages in messages_list:
			text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
			p = self.tokenizer.apply_chat_template(messages[:1], tokenize=False)
			assert text.startswith(p)
			assert prefix is None or p == prefix, 'all prompts must share the same system prompt'
			prefix = p
			suffixes.append(text[len(p):])
		return prefix, suffixes

	def _prefix_forward(self, prefix):
		ids = self.tokenizer(prefix, return_tensors='pt', add_special_tokens=False).input_ids.to(self.device)
		return self.model(input_ids=ids, use_cache=True).past_key_values, ids.shape[1]

	def _suffix_logits(self, suffixes, prefix_cache, prefix_len):
		"""
		在 batch=1 的前缀 cache 后面接一批后缀，返回 [len(suffixes), len(OPTIONS)] 的 logits
		prefix_cache 会被原地修改（扩成 batch、追加后缀的 KV）
		"""
		# 后缀右填充：前缀 cache 后面紧接真实 token，位置编码不受 padding 影响
		inputs = self.tokenizer(suffixes, return_tensors='pt', padding=True,
			add_special_tokens=False).to(self.device)
		n = inputs.input_ids.shape[0]
		prefix_cache.batch_repeat_interleave(n)
		mask = torch.cat([torch.ones(n, prefix_len, dtype=inputs.attention_mask.dtype,
			device=self.device), inputs.attention_mask], dim=1)
		hidden = self.model.model(input_ids=inputs.input_ids, attention_mask=mask,
			past_key_values=prefix_cache, use_cache=True).last_hidden_state
		# 取每条后缀最后一个真实 token 的隐状态来预测回答的第一个 token
		last = inputs.attention_mask.sum(1) - 1
		h = hidden[torch.arange(n, device=self.device), last]
		# 选项字母对应的 lm_head 行：只算这几个 token 的 logits，不算整个词表
		option_head = self.model.lm_head.weight[self.option_ids]
		return (h @ option_head.T).float()

	@torch.inference_mode()
	def option_logits(self, messages_list):
		"""
		推理用（采样 / 评估），不带梯度
		messages_list: 每局游戏一份 chat messages
		返回: [len(messages_list), len(OPTIONS)] 每个选项的 logits
		"""
		prefix, suffixes = self._split(messages_list)
		if prefix != self.prefix_text:
			self.prefix_cache, self.prefix_len = self._prefix_forward(prefix)
			self.prefix_text = prefix

		chunk = self.opt.llm_forward_bs or len(suffixes)
		logits = []
		for i in range(0, len(suffixes), chunk):
			cache = copy.deepcopy(self.prefix_cache)		# forward 会原地往 cache 里追加，所以每次复制
			logits.append(self._suffix_logits(suffixes[i:i + chunk], cache, self.prefix_len).cpu())
		return torch.cat(logits)

	def action_probs(self, logits):
		"""行为策略 π_mix = (1-ε)·softmax(logits/T) + ε/3：ε 部分保证 C（宣布）也能被探索到"""
		probs = F.softmax(logits / (self.opt.llm_temperature or 1.0), dim=-1)
		eps = self.opt.eps or 0
		return (1 - eps) * probs + eps / len(OPTIONS)

	def select(self, logits, sample=False):
		"""
		sample=True：按 π_mix 采样（探索，训练用）
		sample=False：取概率最大的选项（贪心，测试用）
		"""
		if sample:
			return torch.multinomial(self.action_probs(logits), 1).squeeze(1)
		return logits.argmax(dim=-1)

	def learn_from_episode(self, episode):
		"""
		REINFORCE：对 episode.decisions 里每个决策 (b, messages, option) 做一次策略梯度更新
		返回 loss（所有局奖励相同、优势全为 0 时不更新，返回 0）
		"""
		opt = self.opt
		R = episode.r[:, 0]				# 团队奖励，所有 agent 相同
		adv = R - R.mean()
		# 优势为 0 的决策对梯度没有贡献，直接跳过（省掉一大半前向/反向）
		decisions = [(m, o, adv[b].item()) for b, m, o in episode.decisions if adv[b] != 0]
		if not decisions:
			return 0.0

		self.optimizer.zero_grad()
		total_loss = 0.0
		chunk = opt.llm_forward_bs or len(decisions)
		for i in range(0, len(decisions), chunk):
			batch = decisions[i:i + chunk]
			prefix, suffixes = self._split([m for m, _, _ in batch])
			# 带梯度的前向：前缀每个 chunk 重算一次（LoRA 也作用在前缀上，梯度要能传回去）
			prefix_cache, prefix_len = self._prefix_forward(prefix)
			logits = self._suffix_logits(suffixes, prefix_cache, prefix_len)
			options = torch.tensor([o for _, o, _ in batch], device=self.device)
			a = torch.tensor([x for _, _, x in batch], device=self.device)
			logp = torch.log(self.action_probs(logits)[torch.arange(len(batch), device=self.device), options])
			# 除以全部决策数（包括跳过的），和"对整个 batch 求平均"一致
			loss = -(a * logp).sum() / len(episode.decisions)
			loss.backward()				# 逐 chunk 反向、梯度累加，控制显存
			total_loss += loss.item()

		clip_grad_norm_(self.params, max_norm=1.0)
		self.optimizer.step()
		# 参数变了，system prompt 的 KV cache 失效
		self.prefix_text = None
		return total_loss

	def state_dict(self):
		"""断点续训用：LoRA 参数 + 优化器状态（原模型冻结，不需要保存）"""
		return {
			'lora': {n: p.detach().cpu() for n, p in self.model.named_parameters() if p.requires_grad},
			'optimizer': self.optimizer.state_dict(),
		}

	def load_state_dict(self, state):
		params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
		assert params.keys() == state['lora'].keys(), 'checkpoint LoRA does not match the current model'
		with torch.no_grad():
			for n, p in params.items():
				p.copy_(state['lora'][n])
		self.optimizer.load_state_dict(state['optimizer'])	# 会自动把状态搬到参数所在的设备
		self.prefix_text = None		# 参数变了，system prompt 的 KV cache 失效

	def save(self, path):
		self.peft_model.save_pretrained(path)
