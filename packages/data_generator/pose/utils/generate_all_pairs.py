#!/usr/bin/env python3
"""
Generate pose pairs for all environments in circle_envs_10k
"""

import os
import sys
import glob
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

def generate_pairs_for_env(pose_file: str) -> Tuple[str, bool]:
    """단일 환경의 pose 파일에 대해 pairs 생성"""
    env_name = Path(pose_file).stem
    cmd = f"python pose_pair_generator.py --input {pose_file}"
    
    print(f"🎯 Generating pairs for {env_name}...")
    result = os.system(cmd)
    
    return env_name, result == 0

def main():
    # Pose 파일 목록 가져오기
    pose_pattern = "../../../data/pose/circle_envs_10k/circle_envs/*_rb_3_poses.json"
    pose_files = glob.glob(pose_pattern)
    
    if not pose_files:
        print(f"❌ No pose files found matching pattern: {pose_pattern}")
        return
    
    total_envs = len(pose_files)
    print(f"🚀 Found {total_envs} pose files")
    
    # 출력 디렉토리 생성
    output_dir = Path("../../../data/pose_pairs/circle_envs_10k")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 진행 상황 추적
    start_time = time.time()
    success_count = 0
    
    # 순차적으로 처리
    for i, pose_file in enumerate(pose_files, 1):
        env_name, success = generate_pairs_for_env(pose_file)
        
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
    print(f"✅ Completed pair generation for {total_envs} environments")
    print(f"   Success: {success_count}/{total_envs} ({success_count/total_envs*100:.1f}%)")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Average time per env: {total_time/total_envs:.1f}s")

if __name__ == "__main__":
    main() 