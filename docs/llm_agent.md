# LLM agent 改动背景（2026-09-28）

这份文档记录把 Switch 游戏的 DQN agent 换成开源 LLM（Qwen）这次改动的背景、设计决策和已知坑，
方便以后出问题时理解"为什么这么写"，再决定怎么修。

## 1. 目标和范围

- **目标**：先把 LLM 版本跑通做演示（只推理，不训练），之后迁到 GPU 上再做 LoRA + RL 微调。
- **范围**：只新增了一条 LLM 分支，原来的 DQN（RIAL / DIAL）代码路径不变。
  `SwitchGame`、`Arena`、`CNetAgent` 都没改；`main.py` 只加了一个 `agent_type` 分支。

## 2. 文件一览

| 文件 | 作用 |
|---|---|
| `llm/prompt_builder.py` | 把 agent 的观测和历史写成 chat prompt，定义 A/B/C 三个选项 |
| `llm/llm_agent.py` | `LLMPolicy`：加载 Qwen，对 A/B/C 打分（logits），负责采样或贪心选择 |
| `llm/llm_arena.py` | `LLMArena(Arena)`：重写 `run_episode` / `train`，驱动游戏 |
| `main.py` | `opt.agent_type == 'llm'` 时走 LLM 分支；延迟 import，旧环境不装 transformers 也能跑 DQN |
| `config/switch_3_llm.json` | 3 人零样本 |
| `config/switch_3_llm_hint.json` | 3 人，prompt 里附带一个可行协议（`llm_strategy_hint`） |
| `requirements_llm.txt` | `llm_cpu` 环境的依赖和搭建命令（清华源 + ModelScope） |

新增的配置项：`agent_type`、`llm_model_path`、`llm_temperature`、`llm_forward_bs`、`llm_strategy_hint`。

## 3. 关键设计决策

### 3.1 通信保持 1 bit（灯泡 ON/OFF）
原任务的难点在于只有 1 bit 通信，而且只能传给下一天进来的人（`game_comm_limited=true`）。
如果让 LLM 发自然语言，它可以直接说"2 号、3 号都来过了"，任务就失去意义。
所以 LLM 只能选 A（灯 OFF）/ B（灯 ON）/ C（宣布）。
`LLMArena` 里用 `assert opt.game_comm_limited` 保证这一语义：进门看到的灯就是前一天的人留下的。

### 3.2 记忆写进 prompt，代替 RNN 隐状态
DQN 靠 GRU 记住自己的历史，LLM 没有这个机制，所以每次调用都要把以下信息写进 user prompt：
- 自己的编号、今天第几天
- 进门看到的灯
- 昨天自己在不在房间：本来能从历史推出来，但 0.5B 模型推不出来，所以直接写明
- 自己以前每次来访时看到的灯和当时的选择

### 3.3 多选打分，不做自由生成
只看 assistant 回复第一个 token 在 "A"/"B"/"C" 上的 logits，得到 3 类分布：
- 不会出现输出解析失败
- `sample=True` 时按分布采样（相当于 ε-greedy 的探索），`sample=False` 时取 argmax
- 这组 logits 就是将来做策略梯度时要用的 log π(a|s)

### 3.4 每步只调用 active agent
不在房间的 agent 只能选 NOTHING，所以每一步只为当前 active 的 agent 构造 prompt。
`bs` 局游戏的 prompt 拼成一个 batch，一次前向完成。

### 3.5 共享 system prompt 的 KV cache（性能优化）
CPU 上的前向是算力瓶颈（约 290 token/s），system prompt 约占每条 prompt token 的 60%。
做法：
- system prompt **不含 agent 编号**（编号放到 user prompt 里），所以所有调用的 system prompt 完全相同
- 第一次调用时算出 system 部分的 KV cache 并缓存（`_get_prefix_cache`），之后每个 chunk
  `deepcopy` 一份、再 `batch_repeat_interleave(n)` 复制成 batch
- 后缀（user 部分 + assistant 开头）**右填充**，取每行最后一个真实 token 的 hidden state，
  只和 `lm_head.weight[option_ids]` 相乘，不计算整个词表

提速约 3 倍（batch 16 时每次调用从约 18s 降到约 5–6s）。

## 4. 已踩过的坑（修改时注意）

1. **padding 方向必须是 right**。
   前缀 cache 后面接后缀，如果左填充，pad 会夹在前缀和真实 token 中间，读取位置也会错。
   第一版就因为 `padding_side='left'` 没改，logits 和正确值差了 27。
   另外，不带 cache 的左填充写法如果不传 `position_ids`，padding 行的位置编码也会偏，
   第一次零样本跑的结果受了这个影响（后来已重跑）。
   **改 `option_logits` 之后，务必用"每条 prompt 单独、不 padding 跑一遍"作为基准对比**
   （之前的误差是 2.5e-5，argmax 全部一致）。
2. **system prompt 必须对所有调用相同**。`option_logits` 里有 assert 检查。
   如果以后要在 system prompt 里放 agent 相关内容，要么挪到 user prompt，要么按不同前缀分别缓存。
3. **transformers 5.x 的 API**：`from_pretrained(..., dtype=...)`（不是 `torch_dtype`）；
   `DynamicCache.batch_repeat_interleave` 是原地修改；forward 会原地往 cache 里追加，所以必须 deepcopy。
4. **旧环境 `test1`（Python 3.8 / torch 1.12）装不了新版 transformers**，所以另建了 `llm_cpu`。
   `main.py` 里 LLM 相关的 import 是延迟的，别挪到文件顶部，否则 DQN 在旧环境里会跑不起来。
5. `LLMArena.run_episode` 没有调用 `Arena.create_step_record`：它会建 RNN hidden 张量，
   依赖 `model_rnn_size` 等 DQN 配置项，LLM 配置里没有这些字段。

## 5. 验证情况和结果

- 前缀 cache 路径和"逐条不 padding"的基准比对：最大误差 2.5e-5，argmax 一致。
- **规则 oracle**（严格执行 `llm_strategy_hint` 里的协议，bs=256）：归一化 reward 1.0，
  从未说错（lose=0）。说明游戏、奖励、prompt 里给的信息都是正确且足够的。
- 改完 `main.py` 后，DQN 的 `quick_test.json` 在 `test1` 环境里仍然正常（reward 1.0）。
- Qwen2.5-0.5B-Instruct，CPU，5 个 episode × 16 局：

| 配置 | 归一化 reward | 表现 | 耗时 |
|---|---|---|---|
| `switch_3_llm` | 0.0 | 每步都选 A（灯 OFF），从不宣布 | 约 207s |
| `switch_3_llm_hint` | 0.0 | 会把灯打开，但从不宣布（P(C) 约 0.1%） | 约 245s |

结论：分数是 0 的原因是 0.5B 模型能力不够，不是流程 bug。
如果以后换大模型后仍然是 0，先用 oracle 检查流程，再查 prompt。

提示里的协议（3 人时从不说错）：灯 ON 表示"至少两个不同的人来过"。
灯 ON 且这是我第一次来 → 宣布；
灯已经是 ON，或者今天不是第 1 天且昨天在房间的不是我 → 开灯；否则关灯。

## 6. 后续计划

1. GPU 上换 Qwen2.5-1.5B / 3B，设 `"cuda": 1`，重新评估零样本和带提示两种配置。
2. LoRA + REINFORCE（组内基线）：γ=1、奖励只在结尾，每局的最终奖励就是每一步的回报。
   `LLMArena.train` 现在只做评估，需要实现 `train_mode=True` 时的采样、记录 prompt，
   以及 `learn_from_episode`。注意 `option_logits` 目前带 `@torch.inference_mode()`，
   训练时要另写一个有梯度的版本。
3. 可选：先用训练好的 DIAL 策略生成轨迹，对 LLM 做 SFT 热启动。
