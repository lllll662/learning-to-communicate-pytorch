#!/usr/bin/env bash
# 为每个 agent 启动一个独立的 vLLM 服务（不同端口、各自的显存和 KV cache），供 icl_backend=vllm 使用
#
# 用法：bash icl/launch_vllm.sh <模型本地路径或 ModelScope ID> [agent 数，默认 3] [起始端口，默认 8001]
#   例：bash icl/launch_vllm.sh models/Qwen2.5-3B-Instruct
#   agent k 的地址是 http://localhost:<起始端口 + k - 1>/v1，模型名是 agentk（和 config/switch_3_icl_*.json 一致）
#
# 可选环境变量：
#   GPUS=0,1,2        第 i 个服务用 GPUS 里的第 (i mod 个数) 块卡；不设则都在当前可见的卡上
#   GPU_UTIL=0.28     每个服务占用的显存比例；不设则按"0.9 / 同一块卡上的服务数"分配
#   MAX_LEN=8192      最大上下文长度
#   EXTRA_ARGS="..."  其他 vllm serve 参数，比如显存紧张时加 --enforce-eager
# 停止所有服务：pkill -f "vllm serve"
set -e
MODEL=${1:?usage: bash icl/launch_vllm.sh <model> [nagents] [port0]}
N=${2:-3}
PORT0=${3:-8001}
IFS=',' read -r -a GPU_LIST <<< "${GPUS:-}"
NGPU=${#GPU_LIST[@]}
PER_GPU=$(( NGPU > 0 ? (N + NGPU - 1) / NGPU : N ))
UTIL=${GPU_UTIL:-$(python -c "print(round(0.9 / $PER_GPU, 2))")}
export VLLM_USE_MODELSCOPE=True		# 模型 ID 走 ModelScope 下载（本地路径不受影响）
mkdir -p logs

for i in $(seq 0 $((N - 1))); do
	port=$((PORT0 + i))
	name=agent$((i + 1))
	if [ "$NGPU" -gt 0 ]; then export CUDA_VISIBLE_DEVICES=${GPU_LIST[$((i % NGPU))]}; fi
	# 关闭 prefix caching：每个 agent 只用自己的上下文，不跨请求复用 KV
	nohup vllm serve "$MODEL" --port "$port" --served-model-name "$name" \
		--gpu-memory-utilization "$UTIL" --max-model-len "${MAX_LEN:-8192}" \
		--no-enable-prefix-caching --seed $((i + 1)) ${EXTRA_ARGS:-} > "logs/vllm_$name.log" 2>&1 &
	echo "$name: http://localhost:$port/v1  (GPU ${CUDA_VISIBLE_DEVICES:-default}, util $UTIL, log logs/vllm_$name.log)"
	# 等这个服务分配完显存再启动下一个，否则同时启动会互相误判剩余显存
	until curl -s "http://localhost:$port/v1/models" > /dev/null; do
		if ! kill -0 $! 2> /dev/null; then echo "$name failed to start, see logs/vllm_$name.log"; exit 1; fi
		sleep 5
	done
	echo "$name ready"
done
