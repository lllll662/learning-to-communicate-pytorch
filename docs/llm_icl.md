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
| `icl/backend.py` | 后端：`vllm`（每个 agent 连自己的 HTTP 服务，只用标准库）、`hf`（进程内每个 agent 各加载一份模型，CPU 冒烟用）、`rule`（按 oracle 协议作答的假 LLM，验证流程用） |
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
python -m llm.plot_training results/icl_reflect_0                   # 画曲线（横轴 episode 就是轮次）
```

被打断后用同样的命令重新运行，会从下一轮接着跑。想从头开始：换 `icl_run_dir`、删掉 `state.json`，或者设 `"icl_resume": false`。

换模型只需要用新模型重新运行 `launch_vllm.sh`，不用改配置，因为模型名固定是 agent1..3。比较不同模型时，记得换 `icl_run_dir`，否则会接着上一次的实验继续跑。

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
- `plot_training` 的横轴标签写的是 episode，ICL 里表示的是轮次；loss 面板是空的。
