#!/usr/bin/env python3
"""
Generate trajectories for all pose pairs in environments 1-1000
"""

import os
import sys
import glob
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

from rrt_connect import create_se3_planner

def load_pose_pairs(pair_file: str) -> List[Dict[str, List[float]]]:
    """Pose pair 파일 로드"""
    with open(pair_file, 'r') as f:
        data = json.load(f)
        return data['pose_pairs']['data']

def generate_trajectories_for_env(env_id: int) -> Tuple[str, Dict[str, Any]]:
    """단일 환경의 모든 pose pairs에 대해 trajectory 생성"""
    env_name = f"circle_env_{env_id:06d}"
    
    # 파일 경로 (root/data 기준)
    pair_file = f"../../../data/pose_pairs/circle_envs_10k/{env_name}_rb_3_pairs.json"
    env_file = f"../../../data/pointcloud/circle_envs_10k/{env_name}.ply"
    
    if not os.path.exists(pair_file) or not os.path.exists(env_file):
        print(f"❌ Files not found: {pair_file} or {env_file}")
        return env_name, {
            "success": False,
            "error": "Files not found",
            "total_pairs": 0,
            "success_pairs": 0
        }
    
    # Pose pairs 로드
    pose_pairs = load_pose_pairs(pair_file)
    total_pairs = len(pose_pairs)
    
    print(f"✅ Loaded {total_pairs} pose pairs from {pair_file}")
    print(f"✅ Using environment: {env_file}")
    
    # RRT 플래너 생성
    planner = create_se3_planner(rigid_body_id=3, pointcloud_file=env_file)
    
    # 결과 저장 디렉토리 (root/data/trajectories 기준)
    output_dir = Path(f"../../../data/trajectories/circle_envs_10k/{env_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    trajectories = []
    
    # 각 pair에 대해 trajectory 생성
    for i, pair in enumerate(pose_pairs):
        start_pose = pair['init']
        goal_pose = pair['target']
        
        print(f"🎯 Generating trajectory for pair {i+1}/{total_pairs}...")
        result = planner.plan_trajectory(start_pose, goal_pose)
        
        # 결과 저장
        output_file = output_dir / f"trajectory_{i:02d}.json"
        planner.save_trajectory(result, str(output_file), start_pose, goal_pose)
        
        if result.success:
            success_count += 1
            print(f"✅ Generated trajectory with {result.num_waypoints} waypoints")
        else:
            print(f"❌ Failed to generate trajectory")
        
        trajectories.append({
            "pair_id": i,
            "success": result.success,
            "num_waypoints": result.num_waypoints if result.success else 0,
            "planning_time": result.planning_time,
            "path_length": result.path_length if result.success else 0
        })
    
    return env_name, {
        "success": True,
        "total_pairs": total_pairs,
        "success_pairs": success_count,
        "trajectories": trajectories
    }

def main():
    # 결과 저장 디렉토리
    results_dir = Path("../../../data/trajectories/circle_envs_10k")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 진행 상황 추적
    start_time = time.time()
    success_count = 0
    total_pairs = 0
    success_pairs = 0
    
    # 결과 요약
    summary = {
        "environments": {},
        "total_envs": 1,  # 테스트: 1개만
        "success_envs": 0,
        "total_pairs": 0,
        "success_pairs": 0,
        "total_time": 0,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 테스트: 0번 환경만 처리
    for env_id in range(1):
        env_start_time = time.time()
        
        # Trajectory 생성
        env_name, result = generate_trajectories_for_env(env_id)
        
        # 결과 업데이트
        if result["success"]:
            success_count += 1
            total_pairs += result["total_pairs"]
            success_pairs += result["success_pairs"]
        
        # 진행 상황 출력
        elapsed = time.time() - start_time
        avg_time = elapsed / (env_id + 1)
        remaining = avg_time * (1 - (env_id + 1))  # 테스트: 1개만
        
        print(f"\nProgress: {env_id+1}/1 ({(env_id+1)/1*100:.1f}%)")
        print(f"Success rate: {success_count}/{env_id+1} ({success_count/(env_id+1)*100:.1f}%)")
        if total_pairs > 0:
            print(f"Pair success rate: {success_pairs}/{total_pairs} ({success_pairs/total_pairs*100:.1f}%)")
        else:
            print("Pair success rate: N/A (no pairs processed yet)")
        print(f"Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s")
        print(f"Average time per env: {avg_time:.1f}s\n")
        
        # 결과 저장
        summary["environments"][env_name] = result
        summary["success_envs"] = success_count
        summary["total_pairs"] = total_pairs
        summary["success_pairs"] = success_pairs
        summary["total_time"] = elapsed
        
        # 중간 결과 저장
        with open(results_dir / "generation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    # 최종 결과
    summary["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(results_dir / "generation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Completed trajectory generation for 1 environment")  # 테스트: 1개만
    print(f"   Success environments: {success_count}/1 ({success_count/1*100:.1f}%)")
    if total_pairs > 0:
        print(f"   Success pairs: {success_pairs}/{total_pairs} ({success_pairs/total_pairs*100:.1f}%)")
    else:
        print("   Success pairs: N/A")
    print(f"   Total time: {elapsed:.1f}s")
    print(f"   Average time per env: {elapsed/1:.1f}s")

if __name__ == "__main__":
    main() 