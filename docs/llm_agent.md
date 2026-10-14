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
6. **Linux GPU 上报 `RuntimeError: Failed to find C compiler`**（2026-09-30，在 GPU 服务器上第一次正式训练时遇到）：
   torch 2.14 在 GPU 上的部分算子（比如 RoPE 里的矩阵乘）走 Triton，Triton 第一次启动时要用 gcc
   编译一个辅助模块，精简容器里通常没有 gcc。解决办法：把 gcc 装到 conda 环境里（命令见 `requirements_llm.txt`），
   再 `export CC=$(which gcc)`。如果之后又报 `cannot find -lcuda`，就把 `TRITON_LIBCUDA_PATH` 设成 libcuda.so 所在的目录。
   装好后，1001 个 episode 大约 35–40 分钟（每个 episode 约 2 秒）。

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

## 6. LoRA + REINFORCE 微调（第二次改动）

### 6.1 算法
- **只训练 LoRA**：r=8，作用在 q/k/v/o 投影上，约 108 万参数（占 0.22%），原模型冻结。
  3 个囚犯共用同一个 LoRA（参数共享），靠 prompt 里的编号区分身份。
- **回报**：γ=1、奖励只在结尾，所以每局的团队奖励 R_b 就是这局每个决策的回报。
- **优势**：A_b = R_b − mean(R)，组内基线，bs 就是组的大小。优势为 0 的决策直接跳过，不做前向。
- **loss** = −Σ A_b · log π_mix(a|prompt) / 这个 episode 的决策总数；梯度裁剪 1.0，优化器 AdamW。
- 训练循环：每个 episode 先采样 → 更新一次；每 `step_test` 轮用贪心策略评估一次并写入 CSV。
  结束时把 LoRA adapter 保存到 `llm_save_path`。

### 6.2 冷启动与探索（为什么有 ε）
零样本时 P(C) 约 0.1%，几乎从不宣布 → 所有局 R=0 → 优势全是 0 → 没有梯度。
所以采样用混合策略 **π_mix = (1−ε)·softmax(logits/T) + ε/3**：
有 ε 的概率在 A/B/C 中均匀选，保证宣布会发生，从而产生输赢信号。
**采样和算 log π 都用 π_mix**，这样策略梯度对 π_mix 是无偏的，不需要做重要性采样修正。
评估时用 logits 的 argmax（贪心），不含 ε。
没有采用 SFT 热启动，是为了不把人为设计的协议灌进模型（研究的是通信能否自己涌现）。
如果 RL 学不动，SFT 热启动是备选方案。

### 6.3 带梯度前向与 KV cache
- **推理**（`option_logits`）：system prompt 的 KV 只算一次并缓存，每个 chunk deepcopy 后使用。
- **训练**（`learn_from_episode`）：每个 chunk 用 batch=1 **带梯度地重算一次**前缀，再
  `batch_repeat_interleave` 扩成 batch。LoRA 也作用在 system prompt 上，梯度要能传回前缀。
  带梯度的张量不能 deepcopy，所以训练时不复用推理的缓存。
- **每次 `optimizer.step()` 之后必须把 `prefix_text` 置为 None**，否则推理还在用旧参数算出的 KV。
- 前向和反向逐 chunk 进行（`llm_forward_bs`）、梯度累加，以控制显存和内存。

### 6.4 验证
- 训练路径（共享前缀 + 右填充 + 分 chunk 反向）的 LoRA 梯度，和"逐条完整 prompt、不 padding、
  不用 cache"的基准对比：相对误差 7.8e-5；loss 前 5 位有效数字一致。
  验证前先把 lora_B 设成随机值，否则初始 B=0 时 A 的梯度恒为 0，对比没有意义。
- 一次 `learn_from_episode` 之后：前缀缓存已失效，推理结果和基准的误差为 1.1e-5。
- **以后改 `_suffix_logits` / `learn_from_episode` 时，重做这两项对比。**

### 6.5 配置
| 配置 | 用途 |
|---|---|
| `switch_3_llm_rl.json` | 正式训练，bs=32、1001 个 episode、`cuda: 1`（没有 GPU 时自动回落到 CPU） |
| `switch_3_llm_rl_quick.json` | CPU 上验证流程，bs=8、3 个 episode |

新增配置项：`llm_train`、`llm_lora_r`、`llm_save_path`、`llm_lora_path`（加载已训练的 adapter，
可以配合 `llm_train: false` 做纯评估）、`llm_ckpt_every`、`llm_resume`（见 6.7）。另外复用了原有的 `eps`、`learningrate`、`step_test`。

### 6.6 预期与监控
- 参照线："只在第 6 天宣布"不需要通信，归一化 reward 约为 0.65；oracle 协议为 1.0。
  如果曲线卡在 0.65 附近，说明模型学会了看日期，但没学会用灯。
- 训练日志（`-v`）里：lose 高 → 乱宣布；tell rate 趋近 0 → 缩回"永远不说"的状态。

### 6.7 断点续训与时间记录（第三次改动，2026-09-30）
GPU 正式训练要跑上千个 episode，为了防止中途被打断后从头再来，加了自动保存和续训。

**保存目录**（`main.py` 会给 `llm_save_path` 加上 trial 编号，和结果 CSV 一致，例如 `results/lora_switch_3_rl_0/`）：

| 文件 | 内容 | 何时写 |
|---|---|---|
| `checkpoint.pt` | LoRA 参数、AdamW 状态、已完成的 episode、测试结果列表、累计用时、torch / CUDA / numpy / random 的随机数状态、配置 | 每 `llm_ckpt_every` 个 episode（默认 = `step_test`），以及最后一个 episode |
| `train_log.csv` | 每个 episode 的 train / test 指标（reward、tell rate、win、lose、loss）、累计用时 `elapsed_s`、时间戳 | 每个 episode |
| `adapter_*.safetensors` 等 | 最终的 LoRA adapter，可用 `llm_lora_path` 加载 | 训练结束时 |

**续训**：用同样的命令重新运行即可。启动时发现 `checkpoint.pt` 就从下一个 episode 接着训，并且：
- 结果 CSV（`-r` 指定的）会被 `main.py` 以 `'w'` 打开，所以 `train()` 会把 checkpoint 里的测试结果先补写回去；
- `train_log.csv` 里 checkpoint 之后的行会被丢掉（这些 episode 会重新训练）；
- 随机数状态也恢复了，所以"中断 + 续训"和"一口气跑完"的结果完全一样（CPU 上验证过，见下）；
- 累计用时不包含中断的那段时间；ETA 按本次运行的平均速度估算。

**注意事项**：
- 想重新开始一次训练：换一个 `llm_save_path`、删掉 `checkpoint.pt`，或者设 `"llm_resume": false`。
  否则会**自动续训**，可能接上上一次的实验。续训时如果配置和 checkpoint 不一致（`nepisodes`、`step_test`、
  `llm_ckpt_every`、`llm_resume` 除外），会打印 `WARNING: config differs from checkpoint`。
- 想在训完后多训一段：把 `nepisodes` 改大，再运行。
- 最多损失 `llm_ckpt_every` 个 episode 的进度。checkpoint 约 13MB（LoRA 加上 Adam 的两份状态），
  保存时先写 `.tmp` 再替换，保存到一半被打断也不会损坏上一个 checkpoint。
- 没有捕获 Ctrl+C 做额外保存：中断时参数可能正在更新到一半，状态不一致，所以只依赖定期保存。

**验证**：用快速配置和固定 seed 做了对比：
- A：一口气跑 3 个 episode；
- B：先跑 2 个，再换一个 seed 续训第 3 个；续训前还往 `train_log.csv` 里写了一行假数据。

结果：两者最终的 LoRA 参数、测试结果和 CSV 完全一致，假数据行也被正确去掉。
**以后改训练循环或 checkpoint 内容时，重做这个对比。**

### 6.8 训练曲线可视化（2026-09-30）
`llm/plot_training.py` 读取 `train_log.csv` 画 PNG，只依赖 matplotlib / numpy，不加载模型。
- **自动**：`train()` 结束、保存 adapter 之后调用 `plot_run`，生成 `llm_save_path/training_curves.png`。
  画图出错只打印 WARNING，不影响训练结果。
- **手动**（训练中途也可以，`train_log.csv` 每个 episode 都会刷新）：
  ```
  python -m llm.plot_training results/lora_switch_3_rl_0
  ```
- **多次训练对比**（比如通信消融），每次训练一条测试奖励曲线：
  ```
  python -m llm.plot_training results/comm_0 results/nocomm_0 -l comm no-comm -o results/compare.png
  ```

`training_curves.png` 的内容：
| 面板 | 内容 |
|---|---|
| 标题 | episode 数、累计训练用时、每个 episode 的平均用时、最好 / 最后一次测试奖励 |
| 上 | 归一化奖励：测试（贪心）和训练采样（含 ε，EWMA 平滑）；虚线参照 1.0（oracle）和不通信基线（3 人时为 0.65，按 `nagents` / `nsteps` 计算） |
| 左下 | 测试时的 win / lose / announce 比例，用于判断是乱宣布还是不敢宣布（见 6.6） |
| 右下 | REINFORCE loss（原始值 + EWMA）。组内基线下 loss 的绝对值没有意义，只用来看是否发散 |

注意：
- 训练曲线含 ε 探索，比测试低很多是正常的，**以测试曲线为准**。
- 用 `Figure` 对象直接存图，不依赖显示器，服务器上也能跑；图里的文字用英文，避免服务器缺中文字体时显示方块。

## 7. 第一次 GPU 正式训练的结果（2026-09-30 训练，2026-10-09 分析）

`switch_3_llm_rl.json`，1 个 trial，1001 个 episode，用时 31 分钟：
- **测试 reward 始终为 0**。训练后的模型 P(B) ≈ 1、P(C) ≈ 4e-8，也就是"永远开灯、永远不宣布"。
- **参数确实更新了**，问题出在训练方法本身：
  - ε 探索产生的宣布大多是错的，所以 C 被抑制；
  - 策略梯度对低概率动作的推动与它的概率成正比。P(C) 约为 0.001 时，一次正确宣布只能推动约 1%，加了重要性修正也一样；
  - 训练时的宣布都是 ε 随机触发的，不看灯，所以灯和奖励无关，A/B 的选择变成噪声，最后坍缩到 B。
- 训练时的 tell rate 稳定在 0.34，正好等于 ε 造成的下限 1 − (1 − ε/3)^6。**以后只要训练 tell rate 停在这个值，就说明模型自己完全不宣布。**

详细分析和改进方向见 `docs/llm_switch_report.md` 第 4.4–4.5 节和第 6 节。
**如果下次改进后测试 reward 仍为 0，先看 P(C) 有没有上升，不要只看 reward。**

## 8. 后续计划

> 2026-10-12：根据导师意见，Switch 任务不再做权重训练，转向上下文学习，见 `docs/llm_icl.md`。下面的计划保留作为记录。

1. GPU 上正式跑 `switch_3_llm_rl.json`，并和 RIAL/DIAL 的曲线对比。
2. 通信消融：无通信（灯恒定、prompt 中不提灯）、信道噪声、更大的模型。
3. 如果 RL 冷启动过不去：用 oracle 或 DIAL 的轨迹做 SFT 热启动。
