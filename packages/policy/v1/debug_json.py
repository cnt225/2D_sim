#!/usr/bin/env python3

import torch
import numpy as np
import json
import sys, os

sys.path.append('.')
from inference import MotionRFMInference, InferenceConfigs
from convert_trajectory import convert_inference_result

def find_tensors(obj, path=""):
    """딕셔너리/리스트에서 Tensor 찾기"""
    if isinstance(obj, torch.Tensor):
        print(f"🔍 Tensor 발견: {path} = {type(obj)} {obj.shape if hasattr(obj, 'shape') else ''}")
        return True
    elif isinstance(obj, dict):
        found = False
        for key, value in obj.items():
            if find_tensors(value, f"{path}.{key}"):
                found = True
        return found
    elif isinstance(obj, (list, tuple)):
        found = False
        for i, value in enumerate(obj):
            if find_tensors(value, f"{path}[{i}]"):
                found = True
        return found
    return False

# 간단 추론
start = torch.eye(4, dtype=torch.float32)
target = torch.eye(4, dtype=torch.float32)
target[:3, 3] = torch.tensor([1.0, 1.0, 0.0])
pc = np.random.randn(200, 3)

print("🚀 추론 실행...")
engine = MotionRFMInference('checkpoints/motion_rcfm_final_epoch10.pth', 'configs/motion_rcfm.yml')
result = engine.generate_trajectory(start, target, pc, InferenceConfigs.fast())

print("🔍 추론 결과에서 Tensor 찾기:")
find_tensors(result, "result")

print("\n🔄 변환 후:")
converted = convert_inference_result(result)
find_tensors(converted, "converted")

print("\n💾 JSON 저장 시도...")
try:
    with open("test.json", 'w') as f:
        json.dump(converted, f, indent=2)
    print("✅ 성공!")
except Exception as e:
    print(f"❌ 실패: {e}")



