#!/usr/bin/env python3
"""
SE(3) Pose Pair Generator
기존 SE(3) pose 데이터에서 init-target pose 쌍을 생성하는 모듈

사용법:
    python pose_pair_generator.py --input ../../data/pose/circles_only_rb_3_poses.json
    python pose_pair_generator.py --batch  # 모든 pose 파일에 대해 batch 처리
"""

import json
import os
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np


class SE3PosePairGenerator:
    """SE(3) Pose 쌍 생성기"""
    
    def __init__(self, seed: int = None):
        """
        Args:
            seed: 랜덤 시드
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def load_pose_data(self, pose_file: str) -> Dict[str, Any]:
        """SE(3) Pose JSON 파일 로드"""
        with open(pose_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_pose_pairs(self, poses: List[List[float]], num_pairs: int = None) -> List[Dict[str, List[float]]]:
        """
        SE(3) Pose 리스트에서 init-target 쌍 생성
        
        Args:
            poses: SE(3) pose 리스트 (각각 [x, y, z, roll, pitch, yaw] 형태)
            num_pairs: 생성할 쌍의 개수. None이면 poses 개수와 같게 설정
            
        Returns:
            SE(3) pose 쌍 리스트 [{"init": [x,y,z,roll,pitch,yaw], "target": [x,y,z,roll,pitch,yaw]}, ...]
        """
        if len(poses) < 2:
            raise ValueError("At least 2 poses are required to generate pairs")
        
        if num_pairs is None:
            num_pairs = len(poses)
        
        # 가능한 모든 쌍 생성 (자기 자신 제외)
        all_pairs = []
        for i in range(len(poses)):
            for j in range(len(poses)):
                if i != j:  # 자기 자신 제외
                    all_pairs.append({
                        "init": poses[i],
                        "target": poses[j],
                        "init_index": i,
                        "target_index": j
                    })
        
        # 랜덤하게 num_pairs 개 선택
        if len(all_pairs) < num_pairs:
            print(f"Warning: Requested {num_pairs} pairs, but only {len(all_pairs)} unique pairs possible")
            return all_pairs
        
        selected_pairs = random.sample(all_pairs, num_pairs)
        return selected_pairs
    
    def create_pair_file(self, input_file: str, output_dir: str = "../../data/pose_pairs") -> str:
        """
        단일 SE(3) pose 파일에서 pair 파일 생성
        
        Args:
            input_file: 입력 SE(3) pose JSON 파일
            output_dir: 출력 디렉토리
            
        Returns:
            생성된 pair 파일 경로
        """
        
        # 출력 디렉토리 생성
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 입력 파일 로드
        pose_data = self.load_pose_data(input_file)
        poses = pose_data['poses']['data']
        
        print(f"Loading SE(3) poses from: {input_file}")
        print(f"Available poses: {len(poses)}")
        
        # Pose 쌍 생성
        pose_pairs = self.generate_pose_pairs(poses)
        
        # 출력 파일명 생성
        input_filename = Path(input_file).stem
        # "circles_only_rb_0_poses" -> "circles_only_rb_0"
        base_name = input_filename.replace("_poses", "")
        output_filename = f"{base_name}_pairs.json"
        output_path = Path(output_dir) / output_filename
        
        # Pair 데이터 구성
        pair_data = {
            "source_file": input_file,
            "environment": pose_data['environment'],
            "rigid_body": pose_data['rigid_body'],
            "pose_pairs": {
                "data": pose_pairs,
                "count": len(pose_pairs),
                "format": "se3_pose_pairs",
                "description": "Init-target SE(3) pose pairs for path planning"
            },
            "generation_info": {
                "source_poses": len(poses),
                "generated_pairs": len(pose_pairs),
                "generation_method": "random_sampling_without_replacement"
            }
        }
        
        # 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pair_data, f, indent=2, ensure_ascii=False)
        
        print(f"Generated {len(pose_pairs)} SE(3) pose pairs")
        print(f"Saved to: {output_path}")
        
        return str(output_path)
    
    def batch_create_pairs(self, input_dir: str = "../../data/pose", 
                          output_dir: str = "../../data/pose_pairs") -> List[str]:
        """
        지정된 디렉토리의 모든 SE(3) pose 파일에서 pair 파일들 생성
        
        Args:
            input_dir: 입력 SE(3) pose 파일들이 있는 디렉토리
            output_dir: 출력 디렉토리
            
        Returns:
            생성된 pair 파일 경로들의 리스트
        """
        
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # _rb_X_poses.json 패턴의 파일들 찾기
        pose_files = list(input_path.glob("*_rb_*_poses.json"))
        
        if not pose_files:
            print("No SE(3) pose files found (pattern: *_rb_*_poses.json)")
            return []
        
        print(f"Found {len(pose_files)} SE(3) pose files")
        
        created_files = []
        for pose_file in pose_files:
            try:
                pair_file = self.create_pair_file(str(pose_file), output_dir)
                created_files.append(pair_file)
                print(f"✅ Created: {Path(pair_file).name}")
            except Exception as e:
                print(f"❌ Failed to process {pose_file.name}: {e}")
        
        print(f"\n📊 Summary: {len(created_files)}/{len(pose_files)} pair files created")
        return created_files
    
    def calculate_pose_distance(self, pose1: List[float], pose2: List[float]) -> float:
        """
        두 SE(3) 포즈 간의 거리 계산 (Euclidean distance on position + angular difference)
        
        Args:
            pose1, pose2: SE(3) poses [x, y, z, roll, pitch, yaw]
            
        Returns:
            거리 값
        """
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
    
    def generate_diverse_pairs(self, poses: List[List[float]], num_pairs: int, 
                              min_distance: float = 1.0) -> List[Dict[str, Any]]:
        """
        다양성을 고려한 pose 쌍 생성 (너무 가까운 포즈들 제외)
        
        Args:
            poses: SE(3) pose 리스트
            num_pairs: 생성할 쌍의 개수
            min_distance: 최소 거리 제한
            
        Returns:
            다양한 SE(3) pose 쌍 리스트
        """
        diverse_pairs = []
        attempts = 0
        max_attempts = num_pairs * 10
        
        while len(diverse_pairs) < num_pairs and attempts < max_attempts:
            # 랜덤하게 두 포즈 선택
            i, j = random.sample(range(len(poses)), 2)
            
            # 거리 계산
            distance = self.calculate_pose_distance(poses[i], poses[j])
            
            # 최소 거리 조건 만족 시 추가
            if distance >= min_distance:
                pair = {
                    "init": poses[i],
                    "target": poses[j],
                    "init_index": i,
                    "target_index": j,
                    "distance": distance
                }
                diverse_pairs.append(pair)
            
            attempts += 1
        
        if len(diverse_pairs) < num_pairs:
            print(f"Warning: Only generated {len(diverse_pairs)} diverse pairs out of {num_pairs} requested")
        
        return diverse_pairs


# 호환성을 위한 별칭
PosePairGenerator = SE3PosePairGenerator


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='Generate SE(3) pose pairs for path planning')
    
    parser.add_argument('--input', type=str, default=None,
                       help='Input SE(3) pose JSON file')
    parser.add_argument('--batch', action='store_true',
                       help='Process all SE(3) pose files in batch')
    parser.add_argument('--output_dir', type=str, default="../../data/pose_pairs",
                       help='Output directory for pair files')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible results')
    parser.add_argument('--diverse', action='store_true',
                       help='Generate diverse pairs with minimum distance constraint')
    parser.add_argument('--min_distance', type=float, default=1.0,
                       help='Minimum distance for diverse pair generation')
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_arguments()
    
    try:
        # 생성기 초기화
        generator = SE3PosePairGenerator(seed=args.seed)
        
        if args.batch:
            # 배치 처리
            print("🚀 Starting batch SE(3) pose pair generation...")
            created_files = generator.batch_create_pairs(output_dir=args.output_dir)
            print(f"🎉 Batch processing completed! Created {len(created_files)} pair files.")
            
        elif args.input:
            # 단일 파일 처리
            print(f"🚀 Generating SE(3) pose pairs from: {args.input}")
            output_file = generator.create_pair_file(args.input, args.output_dir)
            print(f"🎉 SE(3) pose pairs saved to: {output_file}")
            
        else:
            print("Error: Please specify --input file or --batch mode")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
