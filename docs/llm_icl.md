# 上下文学习（ICL）通信框架：改动背景（2026-10-12）

这份文档记录 `icl/` 的背景、设计决策和注意事项，方便以后出问题时理解"为什么这么写"。
第一版 LLM agent 和 LoRA 训练的背景见 `docs/llm_agent.md`。

## 1. 为什么从权重训练转向上下文学习

- LoRA + REINFORCE 在 1001 轮后收敛到"永远开灯、永远不宣布"，测试 reward 为 0（分析见 `docs/llm_switch_report.md` 4.5 节）。
- 导师的意见：
  - 这个任务的信息密度太低（一局只有一个 ±1 的团队奖励），不适合做权重微调；
  - 应该先做**上下文学习**：用 prompt 和显式外部记忆（`protocol.md`、`memory.md`）搭建通信框架，这更能体现 LLM 的通用能力；
  - **每个 agent 用独立的 LLM 实例**，避免共享 KV cache 带来意料之外的关联。
- `llm/` 下的权重训练代码保留不动，`SwitchGame` / `Arena` 也没有改。

## 2. 实验设计

**不变量**：比赛中唯一的信道仍然是 1 bit 灯泡（`game_comm_limited=true`），奖励和归一化指标不变，0.65 / 1.0 两条参照线照旧。

**自然语言只出现在比赛外**：
- 赛前商定协议：对应经典囚徒谜题里"开局前可以开会"的设定；
- 赛后复盘：所有人看到上帝视角的对局记录。

这对应 Foerster 论文里的"集中学习、分散执行"：开会 / 复盘是集中学习，学习的载体是文字；比赛是分散执行，只能靠灯泡。

**比赛中一个 agent 能看到的上下文**（`icl/prompts.py: build_decision_messages`，参数里没有其他 agent 的信息）：

| 部分 | 内容 | 谁写 |
|---|---|---|
| `protocol.md` | 所有人共享的协议 | 人（E1）或 agent 赛前商定（E2 / E3） |
| `memory.md` → Visits | 自己每次来访：日期、进门时看到的灯、做了什么 | 程序自动记录（保证正确） |
| `memory.md` → Your notes | 自己的笔记，每次决策时覆盖写入 | agent 自己 |
| Today | 第几天、看到的灯、昨天是不是自己在房间 | 程序 |

**决策输出**：先推理，最后输出一行 JSON：`{"action": "OFF|ON|ANNOUNCE", "memory": "..."}`。
- 解析失败时用温度 ≥ 0.7 重试一次（贪心解码重试会得到同样的输出）；
- 还是失败就按 OFF 处理，笔记保持不变，并计入 `parse fail`。

**三组实验**：

| 配置 | 实验 | 协议从哪来 | 测的是什么 |
|---|---|---|---|
| `switch_3_icl_given.json` | E1 | `icl/protocols/switch_3_oracle.md`（报告附录 B 的最优协议） | 能否执行协议、管理记忆，是 E2 / E3 的上限 |
| `switch_3_icl_plan.json` | E2 | 赛前商定 1 次，之后 4 轮比赛都用这份协议 | 能否自己设计可用的协议。建议换 `icl_seed` 重复 3 次 |
| `switch_3_icl_reflect.json` | E3 | 每轮赛后复盘，下一轮重新商定；`lessons.md` 不断积累 | 上下文学习曲线，横轴是轮次 |

**商定协议的流程**（`ICLArena.plan`）：
1. 每人按顺序发言 `icl_plan_turns` 轮，每个人都能看到前面所有人说的话；
2. 1 号作为书记员整理出 `protocol.md`；
3. 其他人审阅，回复 AGREE 或一条修改意见；
4. 有意见时 1 号改一次。

## 3. 代码结构

| 文件 | 作用 |
|---|---|
| `icl/backend.py` | 后端：`vllm`（每个 agent 连自己的 HTTP 服务，只用标准库）、`hf`（进程内推理，CPU 冒烟或单卡本机；同一模型只加载一份权重）、`rule`（按 oracle 协议作答的假 LLM，验证流程用） |
| `icl/prompts.py` | 规则说明、决策 / 讨论 / 整理协议 / 审阅 / 总结经验的 prompt，JSON 解析 |
| `icl/memory.py` | `Workspace`：所有上下文落成文件（目录结构见文件开头的注释） |
| `icl/icl_arena.py` | `ICLArena(LLMArena)`：比赛、商定、复盘、主循环、续跑 |
| `icl/launch_vllm.sh` | 每个 agent 启动一个 `vllm serve`，端口 8001.. |
| `main.py` | `agent_type: "icl"` 分支；`icl_run_dir` 和 `llm_save_path` 一样会加上 trial 编号 |

新增配置项：
- `icl_backend`、`icl_agents`（每个 agent 一项：vllm 填 `url` / `model`，hf 填 `model_path`；只写 1 项表示所有 agent 共用一个实例，用作消融）；
- `icl_protocol_file`、`icl_plan_turns`、`icl_rounds`、`icl_reflect`、`icl_debrief_games`；
- `icl_temperature`（比赛中，默认 0）、`icl_max_tokens`、`icl_plan_temperature`（比赛外，默认 0.7）、`icl_plan_max_tokens`；
- `icl_request_extra`（原样并入 vLLM 请求，比如 Qwen3 关闭思考模式：`{"chat_template_kwargs": {"enable_thinking": false}}`）、`icl_concurrency`、`icl_timeout`；
- `icl_run_dir`、`icl_seed`、`icl_resume`。
- 第二次改动新增的配置项（见第 7 节）：`icl_prefill`、`icl_notes`、`icl_check_protocol`、`icl_check_min`、`icl_check_retries`、`icl_hf_batch`。

`bs` 是每轮并行的局数。`nepisodes` / `step_test` 在 ICL 里不用。

## 4. 运行（GPU 服务器）

```bash
conda activate llm_vllm
modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir models/Qwen2.5-3B-Instruct
bash icl/launch_vllm.sh models/Qwen2.5-3B-Instruct      # 3 个服务，端口 8001-8003

conda activate llm_gpu        # 客户端在哪个环境都行，只用标准库
python -u main.py -c config/switch_3_icl_given.json -r results -v    # E1
python -u main.py -c config/switch_3_icl_plan.json -r results        # E2
python -u main.py -c config/switch_3_icl_reflect.json -r results     # E3
python -m icl.plot_icl results/icl_reflect_0                        # 重画曲线（训练结束时会自动画）
python -m icl.diagnose results/icl_given_0                          # 按规则的正确率 + 宣布分析
```

被打断后用同样的命令重新运行，会从下一轮接着跑。想从头开始：换 `icl_run_dir`、删掉 `state.json`，或者设 `"icl_resume": false`。

换模型只需要用新模型重新运行 `launch_vllm.sh`，不用改配置，因为模型名固定是 agent1..3。比较不同模型时，记得换 `icl_run_dir`，否则会接着上一次的实验继续跑。

### 4.1 单卡本机（Windows，RTX 4080 Laptop 12 GB，没有 vLLM）

`config/switch_3_icl_{given,plan,reflect}_local.json` 的实验参数和服务器配置相同，只是改用 hf 后端（`cuda: 1`），运行目录是 `results/icl_*_local_v2`。

- 3 个 agent 填同一个 `model_path` 时，`HFBackend` 只加载一份权重（3B 约 5.9 GB），因为 12 GB 放不下 3 份。这仍然算独立实例：权重只读；每次 `generate` 都新建 KV cache；hf 分支按 agent 依次调用，不同 agent 的请求不会进同一个 batch。
- `icl_hf_batch: 16` 是实测选的。batch 为 16 / 32 / 48 时，显存峰值分别是 7.1 / 8.4 / 9.6 GB。但 batch 加大一倍，每个 token 的耗时也差不多翻倍，因为瓶颈在 HF `generate` 的逐 token 开销，GPU 利用率只有 10–60%。48 已经接近显存上限，反而更慢。
- 笔记本在长时间负载下会到 85–89°C 并降频（2400 → 1300 MHz 以下），速度可能再慢一倍。建议接电源、垫高机身或用散热垫，并打开厂商的性能模式。
- 无人值守运行：`.venv\Scripts\python -u -m icl.run_batch <配置...> -r results`。

## 5. 验证（CPU）

| 检查 | 结果 |
|---|---|
| `rule` 后端，3 轮 × 32 局，含商定、复盘、经验 | 归一化 reward 1.0，解析失败 0，所有文件都正确生成 |
| 信息隔离：每个决策 prompt 里的访问记录和这个 agent 自己之前的决策完全一致，不出现其他囚犯的编号 | 450 个决策，0 处违规 |
| 续跑：先跑 2 轮，再改成 3 轮重新运行 | 第 3 轮的 `decisions.jsonl` 和一口气跑完的逐字节相同（随机数状态恢复正确）；结果 CSV 补写正确 |
| `hf` 后端，3 份独立的 Qwen2.5-0.5B，bs=2，2 轮 E3 | 流程全部走通（讨论 → 整理协议 → 审阅 → 比赛 → 复盘 → 经验 → 下一轮重新商定），解析失败 0，CPU 上约 4 分钟。但 0.5B 商定出的"协议"是一张没有意义的表格，比赛中总是选 OFF，reward 为 0，说明需要更大的模型 |

**以后改 `run_episode` 或 prompt 时，重新跑 `rule` 配置，确认仍然是 1.0。** 信息隔离的检查脚本思路：对 `decisions.jsonl` 里的每条记录，从同一局、同一个 agent 之前的决策重建访问记录，和 prompt 里的对比。

## 6. 注意事项

- 自由生成比"首 token 打分"慢得多，CPU 上只适合冒烟测试。
- 1 号兼任书记员，所以协议的措辞会偏向 1 号。如果之后要研究"谁主导协议"，可以轮换书记员。
- 复盘给的是上帝视角（谁哪天来过），这是比赛外的信息，不会进入比赛中的 prompt。
- `train_log.csv` 仍然沿用 `llm_arena` 的格式，可以用 `llm.plot_training` 画；但 ICL 的指标以 `icl_metrics.csv` 和 `icl_curves.png` 为准。

## 7. Qwen2.5-3B 第一次结果和第二次改动（2026-10-15）

**结果**（`results_icl/`，hf 后端，每轮 32 局；分析方法见 `icl/diagnose.py`）：
- **E1 = 0**：从不宣布。R1（灯 ON 且第一次来 → 宣布）0/122，推理里一次都没提到 announce。
  另外有这些问题：
  - R3 正确率 0.46；
  - R4 有 35% 误开灯；
  - 30% 的输出先给 JSON 再补推理；
  - 34% 的笔记照抄了模板里的占位文字。
- **E2**：商定的协议没定义宣布条件，用的是"之前每天都在房间"这类不可能成立的条件。第 2 天宣布了 8 次，全输。
- **E3**：协议同样没有逻辑，宣布条件依赖从未定义的 "prediction"。分数偶尔上升，原因是：
  - 宣布主要发生在"第一次来、第 3 天以后、灯 OFF"的局面，靠的是自己的访问记录，不是灯泡；
  - 理论上"第一次来且第 3 天以后就宣布"的分数是 0.28，和最高的一轮（0.29）一致；
  - 剩下的起伏都在每轮 32 局的噪声范围内（标准差约 0.2）。

**改动**：

| 问题 | 改动 |
|---|---|
| 先答后想、跳过规则 1、读错"是不是第一次来" | 决策 prompt 把事实逐条写明（灯、是否第一次来及来过哪几天、昨天在不在）。回复格式固定为 `FACTS → RULE → CHECK（前面的规则为什么不适用）→ JSON`，并在 assistant 回复开头**预填** `FACTS:`（vLLM 用 `continue_final_message`，hf 用 `apply_chat_template(continue_final_message=True)`），保证先推理后作答。`icl_prefill: false` 可关闭 |
| 照抄模板占位文字 | 占位文字改成 `<...>` 的形式；解析时识别出照抄就当作空笔记 |
| 自写笔记可能误导 | `icl_notes: false`：消融，只保留程序记录的客观访问历史 |
| 每轮 32 局噪声太大 | E1/E2/E3 配置改为 `bs: 128`；每轮记录 reward 的 bootstrap 标准误 |
| 只看 reward 看不出原因 | 每轮写 `icl_metrics.csv`，并画 `icl_curves.png`：和最优协议的一致率（按 R1–R4）、宣布次数 / 胜率 / 宣布时灯是 ON 的比例、协议检查的结果 |
| 商定出的协议没有逻辑 | **协议检查**（`icl/protocol_check.py`，见下） |

**协议检查**：
1. 每个 agent 用自己的模型，把协议"编译"成一张决策表：29 种局面 →（第几天, 看到的灯, 是否第一次来, 昨天是不是自己）→ 动作。prompt 用 `build_compile_messages`，格式和比赛中的决策一样，只是没有访问记录和笔记。
2. 穷举 3⁶ = 729 种来访顺序，算出"所有人都按自己的表完美执行时"的归一化 reward，同时记录：
   - 会在哪些局面说错；
   - 动作是否依赖灯泡（不依赖就说明灯泡不携带信息）；
   - 各 agent 的表有多少局面不一致（协议有歧义）。
3. `icl_check_protocol`：
   - `"report"`：只记录（E1 用。人写的协议不修改，这时它衡量的是模型对协议的理解）；
   - `"revise"`：完美执行分数低于 `icl_check_min`（默认 0.65，即不通信也能拿到的分数）时，把检查报告交给 1 号修改，最多 `icl_check_retries`（默认 2）次，用分数最高的一版（E2 / E3 用）。
4. 输出 `round_xx/protocol_check*.md`、`protocol_table*.json`、`protocol_v*.md`。

解读：
- "协议完美执行的分数"和"实际分数"之差，就是执行上的损失；
- 协议分数本身低，说明设计有问题。

**注意**：
- `revise` 等于给了 agent 一个"能推演协议的工具"。如果要研究纯粹的自主设计，就用 `report` 跑一组对照。
- 检查报告里只说分数和出错的局面，不提示具体策略。
- 编译只覆盖依赖这 4 个特征的协议；依赖笔记里的计数等更复杂状态的协议，只能近似。

**验证**（CPU）：
- `rule` 配置仍为 1.0，协议检查 1.0，R1–R4 全部 1.0；
- 用模拟后端先给一份坏协议（"第一次来且第 3 天以后宣布"）：检查得 0.28（理论值 0.278），报告指出灯泡没用到、第 3 天说错率 50%；改写后的协议得 1.0 并被采用；
- `evaluate` 算出的分数和理论值一致："只在第 6 天宣布" 0.65、oracle 1.0。

## 8. 第三次改动：逐条核对条件的决策格式（2026-10-15）

**问题**（`results/icl_given_local_v2_0`，3B，hf，单卡本机）：第二版格式（`FACTS → RULE → CHECK`）下，E1 每局都在第 1 天宣布，全输，reward 约 −1.35。
- FACTS 一行写对了（灯 OFF、第一次来），RULE 却引用规则 1（灯 ON 且第一次来 → 宣布）；CHECK 写的是"规则 2、3、4 不适用"，还把事实说反了。
- 编译出的决策表只有 38/87 和最优协议一致。规律是**只要是第一次来就宣布，不看灯**：模型只核对了合取条件里和事实字面重合的那一半。
- 第一版正好相反（灯 ON 时套用"灯 ON → 开灯"，从不宣布）。两次都是"哪条规则和事实字面重合最明显就套哪条"，prompt 格式只改变了哪一半更显眼。
- 第二版格式本身的缺陷：没有任何一步核对所选规则自己的条件；选的是规则 1 时，CHECK（解释前面的规则）没有内容可写，只能编造。

**改动**：
- `icl_decision_format`：
  - `conditions`（新的默认）：逐条规则、逐个条件写 `= yes/no`，所有条件都是 yes 的第一条规则才适用；
  - `rule_check`：第二版；
  - `free`：第一版的自由推理，不预填。
  比赛和编译协议用同一种格式。
- 编译协议的原始输出存成 `round_xx/protocol_compile*.jsonl`（之前只存了解析结果，查不到 24% 解析失败的原因）。
- `icl/compare_formats.py`：只做编译检查（29 个局面 × 每种格式），对比各格式和最优协议的一致数、完美执行分数、解析失败，以及按"是否第一次来 × 灯"分组的选择。改 prompt 时先用它迭代，再跑完整对局：
  ```
  python -m icl.compare_formats -c config/switch_3_icl_given_local.json --formats rule_check conditions --agents 1
  ```
- `icl_hf_dtype`：hf 后端的精度（比如 CPU 内存不够时用 `"bfloat16"`）。

**验证**（CPU）：
- `rule` 配置仍为 1.0；`compare_formats` 在 rule 后端下三种格式都是 87/87。
- 0.5B 上两种格式都能跑通，但 `conditions` 只有 31% 的输出照格式写了 `= yes/no`（`rule_check` 是 100%）。0.5B 在哪种格式下都做不了这个任务，分数没有参考意义。为了区分"没照格式写"和"照写了但判断错"，`compare_formats` 加了 "format followed" 一列。
- 本机 CPU（16GB 内存）跑 3B 不现实：bf16 + 换页，70 分钟连一种格式的 29 次生成都没跑完。**3B 的格式对比要在 GPU 电脑上跑。**

**3B 第一次格式对比**（`results/format_compare`）：新格式一致数 20/29（旧格式 14/29），但这**不是逐条核对带来的**。
- 22/29 条输出原样照抄了模板里的 `= yes/no`、`-> applies / does not apply`，没有一条真正填写了 yes 或 no；条件栏里塞的是整条规则原文。
- 原因：旧格式的占位放在 `<...>` 里，模型会替换；新格式的 `yes/no` 没放进 `<...>`，3B 当成了要原样输出的文字。
- 照抄之后它凭直觉直接给动作：灯 OFF 的局面几乎全对，但一次都不宣布。旧格式则是会推理，但只看一半条件。
- 对比工具里"照格式写"的检查原来会把照抄的 `= yes/no` 算成已填写，已修正（这次数据重算：新格式 0%）。

**模板修正**：
- 所有占位都放进 `<...>`，并明确要求不要照抄；
- 加一个和本任务无关的、已填好的示例（下雨 / 带伞，演示"合取条件里有一个 no，这条规则就不适用"），放在格式说明之前，不暗示任何策略；
- 新增 `DECISION: <规则编号> -> <动作>` 一行，把规则对应到动作；
- 解析器兜底：没写 JSON、停在 DECISION 行时，从 DECISION 行读出动作；
- "照格式写"现在要求真的填了 yes/no，而且没有照抄 `<...>` 或下雨的示例。

0.5B 上新模板仍然不会照做（把答案写在 `<...>` 里、照抄示例），这在预期之内。3B 的效果要在 GPU 上重跑 `compare_formats` 确认。

## 9. DeepSeek 在线 API 后端（2026-10-25）

为了在跑本地模型之前先确认"换个更强的模型，框架是否可行"，加了一个在线 API 后端。

- **配置**：`config/switch_3_icl_deepseek.json`，`icl_backend: "deepseek"`，每条请求用 `{"model": "deepseek-chat"}`。
  端点默认 `https://api.deepseek.com/beta/chat/completions`（`/beta` 支持预填回复开头）。
- **key 只从本地读取，不写进配置和仓库**：环境变量 `DEEPSEEK_API_KEY`，或者仓库根目录的 `.deepseek_key`、
  `~/.deepseek_key`（后两者已在 `.gitignore` 里）。配置文件里只有模型名。
- **预填**：请求里最后一条 assistant 消息带 `"prefix": true`，模型接着它往下写。如果端点返回 400（不支持预填），
  自动退回"在 user 消息里要求回复以某段文字开头"。
- **解析**：`content` 为空时读 `reasoning_content`（思考类模型会把思考过程放这里）。
- **限流**：429 / 5xx 按 4、8、16、32 秒退避重试；`icl_concurrency` 默认 4，别开太高。
- **可用模型**（2026-10 实测）：`deepseek-chat`、`deepseek-flash`、`deepseek-v4-pro`。

**验证**：用本地 mock 服务（模拟 prefix 预填和响应格式）跑通了单次调用和 `main.py` 的完整一轮（比赛 + 协议检查 + 指标 + 画图），
没有消耗真实额度。真实 API 的连通性、模型列表和 prefix 续写也已单独确认。

用法：
```bash
echo 'sk-...' > .deepseek_key          # 一次性，文件不会被提交
python -u main.py -c config/switch_3_icl_deepseek.json -r results
python -m icl.compare_formats -c config/switch_3_icl_deepseek.json --formats conditions rule_check --agents 1
```

**结果**（`deepseek-chat`，3 个 agent 共用一个 API，每轮 32 局，2026-10-25）：

| 实验 | 结果 |
|---|---|
| 协议编译（最优协议） | 两种格式都是 87/87，3 个 agent 的表完全一致（3B：10–14/29） |
| E1 给定协议（`results/deepseek_given_0`） | 4 轮全部 1.0；R1–R4 正确率 100%；所有宣布都在灯 ON 时，全部正确 |
| E2 赛前商定（`results/deepseek_plan_0`） | 设计出"1 号计数、2/3 号各开一次灯"的经典计数员协议，从不说错，但 6 天里很难数满。按协议写成程序穷举，理论值 0.185；实际 4 轮均值 0.15 |
| E3 复盘迭代（`results/deepseek_reflect_0`） | 第 0 轮协议是"第 6 天宣布（加一个几乎不触发的过滤条件）"，检查 0.65、实际 0.61，相当于不通信的基线。之后每轮复盘都加条件，协议从 1.9KB 涨到 4.4KB，检查分数 0.49 → 0；第 4 轮起基本不再宣布。10 轮均值 0.11 |

结论：
- 换成强模型后，**执行给定协议没有问题**，说明 3B 的失败是能力问题，框架本身可行。
- 但 DeepSeek **自己设计的协议都没超过不通信的 0.65**：E2 选了安全但赢得少的计数员协议，E3 找到第 6 天宣布之后，复盘让协议越改越复杂、越来越保守，最后不再宣布。
- **协议检查的局限**：计数员协议依赖笔记里的计数 c，超出了 4 个特征；编译时又写了"没有笔记"，所以 c 总是 0，检查判它"从不宣布"（0.00–0.11），但实际比赛中能正确宣布。另外有角色分工的协议，"各 agent 理解不一致"（69%）是设计使然，不代表协议有歧义。所以 E2/E3 的 `revise` 是在被低估的分数上打回协议。改进方向：编译时把"笔记里的计数"也作为局面特征，或者直接用比赛记录评估协议。

## 10. 第五次改动：中文 prompt、基于比赛记录的协议评估、E3 保留更好的协议（2026-10-27）

**中文 prompt**（`icl_lang: "zh"`）：
- 所有给模型看的文字（规则、决策、商定、复盘）都放在 `icl/texts.py` 的中英文表里；`prompts.py` 只负责拼装。
- JSON 里的动作名（OFF / ON / ANNOUNCE）保持英文，解析器不用改；灯在中文里写成"开 / 关"。
- 中文版最优协议：`icl/protocols/switch_3_oracle_zh.md`。
- 默认仍是英文；改动前后英文 prompt 逐字相同（17 种 prompt 对照过），3B 的实验可以原样复现。
- 控制台输出、复盘报告（debrief.md）、协议评估报告都改成中文。

**基于比赛记录的协议评估**（`icl_check_method: "replay"`，新的默认）：
- 旧的"编译"把协议翻译成 29 个局面的决策表，只有 4 个特征，而且编译时声明"没有笔记"。依赖笔记状态的协议（比如计数员协议的计数 c）会被严重低估：理论值 0.185 的计数员协议，被判成 0.00–0.11。
- 新方法：用这份协议在一组**固定的来访顺序**上真的打一遍比赛（`icl_eval_bs` 局，默认等于 `bs`；种子 `icl_eval_seed`）。agent 有完整的访问记录和自己的笔记，所以能正确评估带状态的协议；有角色分工的协议也不会再被误报"理解不一致"。
- 整次实验用同一组来访顺序，不同轮次、不同协议的分数可以直接比较。评估不消耗正式比赛的随机数，续跑仍然可复现。
- 输出：`round_xx/protocol_eval*.md`（中文报告）、`protocol_eval*.jsonl`（评估比赛的完整记录）。
- 旧方法仍可用：`icl_check_method: "compile"`；`icl/compare_formats.py` 也仍然用编译做快速格式对比。

**E3 保留更好的协议**（`icl_keep_better`，默认开启）：
- 复盘后商定的新协议，如果评估分数比上一轮的协议低，就保留上一轮的协议，新协议存为 `round_xx/protocol_rejected.md`。`icl_metrics.csv` 的 `protocol_kept` = 1 表示这一轮沿用了上一版。
- 目的是防止 E3 越改越差（上一次 DeepSeek 的 E3 在第 4 轮之后不再宣布）。

**验证**（本机，不调用 API）：
- rule 后端：中英文都是 1.0；替换为重放评估后协议评估也是 1.0。
- 计数员协议（假后端严格按协议执行，计数存在笔记里）：重放 4000 局得 **0.19**，和穷举的理论值 0.185 一致；旧的编译式检查只给 0.00–0.11。
- 保留更好的协议：第 0 轮好协议（1.0），之后每轮复盘都给出坏协议（0.0）→ 两次都保留了好协议，`protocol_kept` 正确记录。
- 续跑：恢复后第 2 轮的决策和评估记录与一口气跑完的逐字节相同。

## 11. 改用中转商 / 换 key（2026-10-27）

配置里的 `icl_agents` 每一项都可以单独指定：

```json
"icl_agents": [
   {"url": "https://中转商域名/v1", "model": "deepseek-chat", "prefix": false, "key_env": "DEEPSEEK_API_KEY"}
]
```

- `url`：写 `.../chat/completions`、`.../v1` 或站点根地址都行，缺少的 `/chat/completions` 会自动补上。
- `model`：中转商的模型名可能和官方不同，先用 `python -m icl.check_api -c <配置>` 查出来再填。
- `prefix`：中转商一般不支持官方的"预填回复开头"。不填时后端会先试，遇到 400 自动退回"要求回复以某段文字开头"；填 `false` 可以直接跳过这次失败。
- `key_env`：key 所在的环境变量名，默认 `DEEPSEEK_API_KEY`。换 key 只需要改环境变量，或者覆盖仓库根目录的 `.deepseek_key`（已忽略提交）。
- 官方端点是默认值，不用写。

**检查配置**（不消耗多少额度，几秒钟）：
```bash
python -m icl.check_api -c config/switch_3_icl_deepseek.json
```
依次检查：key 能否读到、地址能否连上、模型列表、一次最简单的对话、是否支持预填。每一步失败都会给出原因和修法。

**验证**：用本地模拟的中转服务（不支持预填 / 支持预填 / 模型名写错）测试了 `check_api` 和后端：地址自动补全、400 后退化、`prefix: false` 直接不发送、模型名错误都能正确报告。
