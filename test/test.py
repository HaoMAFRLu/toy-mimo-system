import numpy as np
import scipy.signal as signal

# 1️⃣ 定义传递函数 H(s) = 1 / (s^2 + 3s + 2)
num = [1]         
den = [1, 3, 2]   
H = signal.TransferFunction(num, den)

# 2️⃣ 生成输入信号 u(t) (正弦波)
T = 10  # 总时间（秒）
fs = 100  # 采样率（Hz）
t = np.linspace(0, T, T * fs)  # 时间向量
u = np.sin(2 * np.pi * 1 * t)  # 1Hz 正弦波

# 3️⃣ 计算系统响应 y(t)
t_out, y_out, _ = signal.lsim(H, U=u, T=t)

# 4️⃣ 检查 y_out 是否包含复数
print("y_out 是否包含虚部:", np.any(np.iscomplex(y_out)))
print("y_out 数据类型:", y_out.dtype)
