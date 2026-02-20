#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import ctypes

# ====================== 终极系统级日志屏蔽 ======================
# 1. 屏蔽Python层面的stderr
class DevNull:
    def write(self, msg):
        pass
sys.stderr = DevNull()

# 2. 屏蔽CUDA底层日志（针对Linux）
try:
    libc = ctypes.CDLL('libc.so.6')
    devnull = os.open('/dev/null', os.O_WRONLY)
    libc.dup2(devnull, 2)
    os.close(devnull)
except:
    pass

# 3. TF环境变量强化
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['ABSL_LOG_LEVEL'] = '1'
# =================================================================

print("=" * 60)
print("Framework Comprehensive Test")
print("=" * 60)

# 以下保留你原有所有代码，一字不改
# ------------------------------
# 1. Python & numpy
# ------------------------------
print(f"\n1. Python version: {sys.version}")
print(f"   numpy version: {np.__version__}")

# ------------------------------
# 2. PyTorch + CUDA
# ------------------------------
try:
    import torch
    print(f"\n2. PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        v = torch.version.cuda
        name = torch.cuda.get_device_name(0)
        print(f"   CUDA version: {v}")
        print(f"   GPU: {name}")

    # 简单张量运算
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    print(f"   Torch tensor test: a + b = {a + b}")

except Exception as e:
    print("\n2. PyTorch import failed:", e)

# ------------------------------
# 3. TensorFlow + GPU
# ------------------------------
try:
    import tensorflow as tf
    print(f"\n3. TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"   TF GPUs found: {len(gpus)}")
    for gpu in gpus:
        print(f"      - {gpu}")

    # 简单运算
    x = tf.constant([1, 2, 3])
    print(f"   TF tensor test: x = {x}")

except Exception as e:
    print("\n3. TensorFlow import failed:", e)

# ------------------------------
# 4. TensorRT
# ------------------------------
try:
    import tensorrt as trt
    print(f"\n4. TensorRT version: {trt.__version__}")
except Exception as e:
    print("\n4. TensorRT import failed:", e)

print("\n" + "=" * 60)
print("All framework tests finished.")
print("=" * 60)

