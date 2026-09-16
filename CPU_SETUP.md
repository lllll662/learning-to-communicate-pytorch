# CPU 运行指南

## 环境配置

这个项目已适配为在CPU上运行。使用 `test1` conda环境：

```bash
# 激活环境
conda activate test1

# 或直接指定Python路径
"C:\Users\86133\anaconda3\envs\test1\python.exe" main.py -c config/quick_test.json
```

## 快速测试

```bash
# 快速测试配置（2个agent, 50个epoch，~30秒完成）
"C:\Users\86133\anaconda3\envs\test1\python.exe" main.py -c config/quick_test.json -v

# 标准配置（3个agent, 5001个epoch，需要较长时间）
"C:\Users\86133\anaconda3\envs\test1\python.exe" main.py -c config/switch_3_dial.json -v

# 保存结果到CSV
"C:\Users\86133\anaconda3\envs\test1\python.exe" main.py -c config/switch_3_dial.json -r results/ -n 3 -v
```

## 主要改动

1. **requirements.txt**: 更新PyTorch到2.0.1（兼容CPU）
2. **switch_cnet.py**: 移除过时的 `Variable()` 包装
3. **arena.py**: 移除过时的 `Variable()` 包装
4. **config/quick_test.json**: 添加快速测试配置

## 已验证

- ✓ PyTorch 1.12.1+cpu 在test1环境中可用
- ✓ 代码在CPU上成功运行
- ✓ 快速测试配置完成训练（50个epoch）
- ✓ Git提交完成

## 下一步

你现在可以：
1. 修改配置或代码进行实验
2. 在 `local_test` 分支上继续开发
3. 完成后push到你的GitHub fork
