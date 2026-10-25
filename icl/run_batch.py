"""
无人值守地依次跑多个 ICL 实验
	- 运行期间阻止 Windows 自动睡眠（不影响合盖、手动睡眠；脚本退出后自动恢复）
	- 某个实验异常退出时等一会儿重跑：ICLArena 会从 state.json 续跑，已完成的轮次不会重做
	- 一个实验重试几次仍然失败，就跳过，接着跑下一个
	- 输出同时打印到屏幕，并写到 logs/<配置名>.log（追加写，重跑的输出接在后面）

用法：python -u -m icl.run_batch config/a.json config/b.json [-r results] [--retries 3] [--wait 60]
"""
import argparse
import ctypes
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def now():
	return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def keep_awake():
	if os.name == 'nt':		# 对调用它的线程生效，直到线程退出；这里是一直在等子进程的主线程
		ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def run_once(config, results_path, log):
	cmd = [sys.executable, '-u', 'main.py', '-c', config, '-r', results_path]
	proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
		text=True, encoding='utf-8', errors='replace', bufsize=1)
	for line in proc.stdout:
		print(line, end='')
		log.write(line)
		log.flush()
	return proc.wait()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('configs', nargs='+', help='按顺序运行的配置文件')
	parser.add_argument('-r', '--results_path', default='results')
	parser.add_argument('--retries', type=int, default=3, help='每个实验失败后最多重跑几次')
	parser.add_argument('--wait', type=int, default=60, help='重跑前等待的秒数')
	args = parser.parse_args()

	keep_awake()
	os.makedirs('logs', exist_ok=True)
	summary = []
	for config in args.configs:
		log_path = os.path.join('logs', Path(config).stem + '.log')
		code = None
		with open(log_path, 'a', encoding='utf-8') as log:
			for attempt in range(args.retries + 1):
				header = f'\n===== [{now()}] {config}, attempt {attempt + 1}/{args.retries + 1} =====\n'
				print(header, end='')
				log.write(header)
				code = run_once(config, args.results_path, log)
				if code == 0:
					break
				msg = f'===== [{now()}] {config} exited with code {code}'
				msg += f', retrying in {args.wait}s =====\n' if attempt < args.retries else ', giving up =====\n'
				print(msg, end='')
				log.write(msg)
				if attempt < args.retries:
					time.sleep(args.wait)
		summary.append((config, code, log_path))

	print(f'\n===== [{now()}] all done =====')
	for config, code, log_path in summary:
		print(f'{"OK    " if code == 0 else "FAILED"} {config}  (log: {log_path})')
	sys.exit(0 if all(code == 0 for _, code, _ in summary) else 1)


if __name__ == '__main__':
	main()
