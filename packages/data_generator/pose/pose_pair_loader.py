#!/usr/bin/env python3
"""
SE(3) Pose Pair Loader
Init-target SE(3) pose 쌍 데이터를 로드하고 인덱싱하는 모듈

사용법:
    from pose_pair_loader import SE3PosePairLoader
    
    loader = SE3PosePairLoader()
    init_pose, target_pose = loader.get_pose_pair("circles_only_rb_0", 0)
    available_files = loader.list_available_pairs()
    total_pairs = loader.get_pair_count("circles_only_rb_0")
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional


class SE3PosePairLoader:
    """SE(3) Pose 쌍 로더"""
    
    def __init__(self, data_dir: str = "../../data/init_target"):
        """
        Args:
            data_dir: SE(3) pose pair 파일들이 있는 디렉토리
        """
        # 상대 경로를 절대 경로로 변환
        import os
        if not os.path.isabs(data_dir):
            # pose 폴더에서 실행될 때와 루트에서 실행될 때 모두 고려
            if os.path.exists(data_dir):
                self.data_dir = data_dir
            elif os.path.exists(os.path.join("..", data_dir)):
                self.data_dir = os.path.join("..", data_dir)
            else:
                self.data_dir = data_dir
        else:
            self.data_dir = data_dir
        self._cache = {}  # 파일 캐시
    
    def _load_pair_file(self, filename: str) -> Dict[str, Any]:
        """SE(3) Pose pair 파일 로드 (캐시 사용)"""
        if filename not in self._cache:
            file_path = os.path.join(self.data_dir, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"SE(3) pose pair file not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                self._cache[filename] = json.load(f)
        
        return self._cache[filename]
    
    def list_available_pairs(self) -> List[str]:
        """
        사용 가능한 SE(3) pose pair 파일들의 환경명 리스트 반환
        
        Returns:
            환경명 리스트 (예: ["circles_only_rb_0", "circles_only_rb_3"])
        """
        if not os.path.exists(self.data_dir):
            return []
        
        pair_files = []
        for file in os.listdir(self.data_dir):
            if file.endswith("_pairs.json"):
                # "circles_only_rb_0_pairs.json" -> "circles_only_rb_0"
                env_name = file.replace("_pairs.json", "")
                pair_files.append(env_name)
        
        return sorted(pair_files)
    
    def get_pair_count(self, env_name: str) -> int:
        """
        특정 환경의 SE(3) pose 쌍 개수 반환
        
        Args:
            env_name: 환경명 (예: "circles_only_rb_0")
            
        Returns:
            pose 쌍 개수
        """
        filename = f"{env_name}_pairs.json"
        try:
            data = self._load_pair_file(filename)
            return data['pose_pairs']['count']
        except FileNotFoundError:
            return 0
    
    def get_pose_pair(self, env_name: str, pair_index: int) -> Tuple[List[float], List[float]]:
        """
        특정 환경의 특정 인덱스 SE(3) pose 쌍 반환
        
        Args:
            env_name: 환경명 (예: "circles_only_rb_0")
            pair_index: 쌍 인덱스 (0부터 시작)
            
        Returns:
            (init_pose, target_pose) SE(3) 포즈 쌍
            각 포즈는 [x, y, z, roll, pitch, yaw] 형태
        """
        filename = f"{env_name}_pairs.json"
        data = self._load_pair_file(filename)
        
        pairs = data['pose_pairs']['data']
        if pair_index < 0 or pair_index >= len(pairs):
            raise IndexError(f"Pair index {pair_index} out of range [0, {len(pairs)-1}]")
        
        pair = pairs[pair_index]
        return pair['init'], pair['target']
    
    def get_all_pairs(self, env_name: str) -> List[Dict[str, List[float]]]:
        """
        특정 환경의 모든 SE(3) pose 쌍 반환
        
        Args:
            env_name: 환경명
            
        Returns:
            pose 쌍 리스트 [{"init": [x,y,z,roll,pitch,yaw], "target": [x,y,z,roll,pitch,yaw]}, ...]
        """
        filename = f"{env_name}_pairs.json"
        data = self._load_pair_file(filename)
        return data['pose_pairs']['data']
    
    def get_metadata(self, env_name: str) -> Dict[str, Any]:
        """
        특정 환경의 메타데이터 반환
        
        Args:
            env_name: 환경명
            
        Returns:
            메타데이터 딕셔너리
        """
        filename = f"{env_name}_pairs.json"
        data = self._load_pair_file(filename)
        
        return {
            'environment': data.get('environment', {}),
            'rigid_body': data.get('rigid_body', {}),
            'generation_info': data.get('generation_info', {}),
            'source_file': data.get('source_file', ''),
            'pose_format': data['pose_pairs'].get('format', 'se3_pose_pairs')
        }
    
    def sample_random_pairs(self, env_name: str, num_pairs: int, seed: int = None) -> List[Tuple[List[float], List[float]]]:
        """
        특정 환경에서 랜덤하게 SE(3) pose 쌍들 샘플링
        
        Args:
            env_name: 환경명
            num_pairs: 샘플링할 쌍 개수
            seed: 랜덤 시드
            
        Returns:
            랜덤 SE(3) pose 쌍들의 리스트
        """
        import random
        if seed is not None:
            random.seed(seed)
        
        all_pairs = self.get_all_pairs(env_name)
        if num_pairs > len(all_pairs):
            print(f"Warning: Requested {num_pairs} pairs, but only {len(all_pairs)} available")
            num_pairs = len(all_pairs)
        
        sampled_pairs = random.sample(all_pairs, num_pairs)
        return [(pair['init'], pair['target']) for pair in sampled_pairs]
    
    def get_pairs_by_distance(self, env_name: str, min_distance: float = 0.0, 
                             max_distance: float = float('inf')) -> List[Dict[str, Any]]:
        """
        거리 조건에 맞는 SE(3) pose 쌍들 필터링
        
        Args:
            env_name: 환경명
            min_distance: 최소 거리
            max_distance: 최대 거리
            
        Returns:
            거리 조건에 맞는 pose 쌍들
        """
        all_pairs = self.get_all_pairs(env_name)
        filtered_pairs = []
        
        for pair in all_pairs:
            # distance 정보가 있는 경우 사용
            if 'distance' in pair:
                distance = pair['distance']
                if min_distance <= distance <= max_distance:
                    filtered_pairs.append(pair)
            else:
                # distance 정보가 없는 경우 계산
                distance = self._calculate_pose_distance(pair['init'], pair['target'])
                if min_distance <= distance <= max_distance:
                    pair_with_distance = pair.copy()
                    pair_with_distance['distance'] = distance
                    filtered_pairs.append(pair_with_distance)
        
        return filtered_pairs
    
    def _calculate_pose_distance(self, pose1: List[float], pose2: List[float]) -> float:
        """
        두 SE(3) 포즈 간의 거리 계산
        
        Args:
            pose1, pose2: SE(3) poses [x, y, z, roll, pitch, yaw]
            
        Returns:
            거리 값
        """
        import numpy as np
        
        # 위치 차이 (Euclidean distance)
        pos_diff = np.array(pose1[:3]) - np.array(pose2[:3])
        pos_distance = np.linalg.norm(pos_diff)
        
        # 방향 차이 (yaw만 고려, 2D 시뮬레이션이므로)
        yaw_diff = abs(pose1[5] - pose2[5])
        # yaw 차이를 [-π, π] 범위로 정규화
        yaw_diff = min(yaw_diff, 2*np.pi - yaw_diff)
        
        # 위치와 방향을 결합한 거리 (가중치 적용)
        total_distance = pos_distance + 0.5 * yaw_diff  # yaw에 0.5 가중치
        
        return total_distance
    
    def print_summary(self, env_name: str) -> None:
        """
        특정 환경의 SE(3) pose 쌍 정보 요약 출력
        
        Args:
            env_name: 환경명
        """
        try:
            metadata = self.get_metadata(env_name)
            pair_count = self.get_pair_count(env_name)
            
            print(f"📊 SE(3) Pose Pair Summary: {env_name}")
            print(f"   Environment: {metadata['environment'].get('name', 'Unknown')}")
            print(f"   Rigid Body: {metadata['rigid_body'].get('name', 'Unknown')}")
            print(f"   Total Pairs: {pair_count}")
            print(f"   Format: {metadata['pose_format']}")
            print(f"   Source File: {metadata['source_file']}")
            
            if pair_count > 0:
                # 첫 번째 쌍 예시 출력
                init_pose, target_pose = self.get_pose_pair(env_name, 0)
                print(f"   Sample Pair 0:")
                print(f"     Init:   [x={init_pose[0]:.2f}, y={init_pose[1]:.2f}, yaw={init_pose[5]:.2f}]")
                print(f"     Target: [x={target_pose[0]:.2f}, y={target_pose[1]:.2f}, yaw={target_pose[5]:.2f}]")
                
        except FileNotFoundError:
            print(f"❌ SE(3) pose pair file not found for: {env_name}")
        except Exception as e:
            print(f"❌ Error loading {env_name}: {e}")
    
    def clear_cache(self) -> None:
        """캐시 클리어"""
        self._cache.clear()


# 호환성을 위한 별칭
PosePairLoader = SE3PosePairLoader


if __name__ == "__main__":
    # 간단한 테스트
    print("🚀 SE(3) Pose Pair Loader Test...")
    
    loader = SE3PosePairLoader()
    
    # 사용 가능한 pair 파일들 확인
    available_pairs = loader.list_available_pairs()
    print(f"Available SE(3) pose pair files: {available_pairs}")
    
    if available_pairs:
        # 첫 번째 파일에 대한 정보 출력
        env_name = available_pairs[0]
        loader.print_summary(env_name)
        
        # 랜덤 쌍 샘플링 테스트
        if loader.get_pair_count(env_name) > 0:
            print(f"\n📝 Testing random sampling:")
            random_pairs = loader.sample_random_pairs(env_name, min(3, loader.get_pair_count(env_name)))
            for i, (init, target) in enumerate(random_pairs):
                print(f"   Pair {i}: [{init[0]:.2f},{init[1]:.2f},{init[5]:.2f}] -> [{target[0]:.2f},{target[1]:.2f},{target[5]:.2f}]")
    
    print("\n🎉 Test completed!")
