#!/usr/bin/env python3
"""
Generate poses for all environments in circle_envs_10k
"""

import os
import sys
import glob
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

def generate_poses_for_env(env_path: str) -> Tuple[str, bool]:
    """단일 환경에 대해 pose 생성"""
    env_name = Path(env_path).stem
    cmd = f"python batch_pose_generator.py circle_envs_10k/{env_name} 3 --num_poses 10"
    
    print(f"🎯 Generating poses for {env_name}...")
    result = os.system(cmd)
    
    return env_name, result == 0

def main():
    # 환경 파일 목록 가져오기
    env_pattern = "../../../data/pointcloud/circle_envs_10k/circle_env_*.ply"
    env_files = glob.glob(env_pattern)
    
    if not env_files:
        print(f"❌ No environment files found matching pattern: {env_pattern}")
        return
    
    total_envs = len(env_files)
    print(f"🚀 Found {total_envs} environments")
    
    # 출력 디렉토리 생성
    output_dir = Path("../../../data/pose/circle_envs_10k")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 진행 상황 추적
    start_time = time.time()
    success_count = 0
    
    # 순차적으로 처리 (나중에 병렬화 가능)
    for i, env_file in enumerate(env_files, 1):
        env_name, success = generate_poses_for_env(env_file)
        
        if success:
            success_count += 1
        
        # 진행 상황 출력
        elapsed = time.time() - start_time
        avg_time = elapsed / i
        remaining = avg_time * (total_envs - i)
        
        print(f"Progress: {i}/{total_envs} ({i/total_envs*100:.1f}%)")
        print(f"Success rate: {success_count}/{i} ({success_count/i*100:.1f}%)")
        print(f"Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s")
        print("")
    
    # 최종 결과
    total_time = time.time() - start_time
    print(f"✅ Completed pose generation for {total_envs} environments")
    print(f"   Success: {success_count}/{total_envs} ({success_count/total_envs*100:.1f}%)")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Average time per env: {total_time/total_envs:.1f}s")

if __name__ == "__main__":
    main() 