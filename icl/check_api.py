"""
检查 API 配置是否可用：地址、模型名、key，以及是否支持"预填回复开头"

	python -m icl.check_api                               # 用 config/switch_3_icl_deepseek.json 里第 1 个 agent
	python -m icl.check_api -c config/我的中转商配置.json
	python -m icl.check_api -c config/xxx.json --agent 2  # 检查第 2 个 agent

做三件事，任何一步失败都会打印具体原因和修法：
	1. GET <地址>/models          列出这个 key 能用的模型名（中转商可能不支持这个接口，失败不算错）
	2. POST chat/completions      发一条最简单的请求，确认 key、模型名和地址都对
	3. 带 prefix 再发一次          看是否支持预填（只有官方 /beta 端点支持；不支持时后端会自动退化）
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

from icl.backend import _endpoint, read_api_key


def _request(url, key, body=None, timeout=60):
	req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
		headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
	with urllib.request.urlopen(req, timeout=timeout) as resp:
		return json.loads(resp.read())


def main():
	if hasattr(sys.stdout, 'reconfigure'):		# Windows 控制台默认 GBK，中文会乱码
		sys.stdout.reconfigure(encoding='utf-8', errors='replace')
	parser = argparse.ArgumentParser(description='check an OpenAI-compatible API endpoint')
	parser.add_argument('-c', '--config', default='config/switch_3_icl_deepseek.json')
	parser.add_argument('--agent', type=int, default=1, help='which entry of icl_agents to check (1-based)')
	args = parser.parse_args()

	opt = json.load(open(args.config, encoding='utf-8'))
	spec = opt['icl_agents'][min(args.agent, len(opt['icl_agents'])) - 1]
	url = _endpoint(spec.get('url') or 'https://api.deepseek.com/beta/chat/completions')
	model = spec.get('model') or 'deepseek-chat'
	key_env = spec.get('key_env') or 'DEEPSEEK_API_KEY'
	print(f'配置：{args.config}')
	print(f'地址：{url}')
	print(f'模型：{model}')
	print(f'key ：环境变量 {key_env} 或 .deepseek_key')
	try:
		key = read_api_key(key_env)
	except RuntimeError as err:
		print('[FAIL]', err)
		return 1
	print(f'     读到 {key[:6]}…{key[-4:]}（{len(key)} 个字符）')

	# 1. 模型列表（可选）
	try:
		data = _request(url.replace('/chat/completions', '/models'), key)
		print('[OK] 模型列表：', ', '.join(m.get('id', '?') for m in data.get('data', []))[:300])
	except Exception as err:		# 中转商常不支持 /models，不算失败
		print('[--] 模型列表不可用（不影响使用）：', err)

	# 2. 最基本的对话
	try:
		r = _request(url, key, {'model': model, 'max_tokens': 20, 'temperature': 0,
			'messages': [{'role': 'user', 'content': 'Reply with the single word: hello'}]})
		print('[OK] 对话正常：', repr(r['choices'][0]['message'].get('content'))[:80])
	except urllib.error.HTTPError as err:
		body = err.read().decode(errors='replace')[:300]
		print(f'[FAIL] 对话失败 {err.code}：{body}')
		print('  404/401 → 检查地址和 key；400 + "model" → 模型名不对，用上面的模型列表换一个')
		return 1
	except Exception as err:
		print('[FAIL] 对话失败：', err)
		print('  检查 url 是否正确（多数中转商的地址形如 https://xxx.com/v1）')
		return 1

	# 3. 预填（可选）
	if spec.get('prefix', True) is False:
		print('[--] 配置里 prefix=false，不测试预填（中转商一般不支持，这是正常的）')
		return 0
	try:
		r = _request(url, key, {'model': model, 'max_tokens': 20, 'temperature': 0, 'messages': [
			{'role': 'user', 'content': 'List the colours: "1. red, 2. orange, 3."'},
			{'role': 'assistant', 'content': '1. red, 2. orange, 3.', 'prefix': True}]})
		print('[OK] 支持预填回复开头：', repr(r['choices'][0]['message'].get('content'))[:80])
	except urllib.error.HTTPError as err:
		print(f'[--] 不支持预填（{err.code}），后端会自动改用"要求回复以某段文字开头"，可以继续用；'
			f'也可以在 icl_agents 里加 "prefix": false 省掉这次 400')
	except Exception as err:
		print('[--] 预填测试失败：', err)
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
