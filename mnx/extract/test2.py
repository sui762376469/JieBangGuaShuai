import psutil
import os

# 获取当前进程
p = psutil.Process(os.getpid())
print(f"当前进程优先级: {p.nice()}")  # Windows 下可能是 32(正常) 或 64(低)
print(f"CPU 亲和性: {p.cpu_affinity()}")