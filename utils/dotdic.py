import numpy as np
import copy

# 继承原生 `dict`，是字典的子类，拥有普通字典全部功能。
class DotDic(dict):
	# 三个魔法方法，实现点号 `.` 读写

	# `__getattr__`：当写 `d.key` 读取属性时触发，等价 `dict.get("key")`
	__getattr__ = dict.get
	# 执行 `d.xxx = value` 的时候，自动转为字典赋值 `d["xxx"] = value`
	__setattr__ = dict.__setitem__
	# 执行 `del d.xxx`，等价 `del d["xxx"]`
	__delattr__ = dict.__delitem__

	# 重写深拷贝方法
	# 如果不写这个函数：直接`copy.deepcopy(dot_dic_obj)`，深拷贝出来会变成普通`dict`，丢失点号访问能力。
	def __deepcopy__(self, memo=None):
		return DotDic(copy.deepcopy(dict(self), memo=memo))