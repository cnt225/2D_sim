#!/usr/bin/env python3
"""
통합 Pose 생성기
환경별로 pose와 pose_pair를 한번에 생성하여 HDF5에 저장하는 파이프라인

사용법:
    generator = UnifiedPoseGenerator(config_file, h5_path)
    result = generator.generate_complete_dataset(
        env_path="circle_env_000000.ply", 
        rb_ids=[0, 1, 2], 
        num_poses=100,
        num_pairs=50
    )
"""

import os
import time
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from .unified_pose_manager import UnifiedPoseManager
    from .random_pose_generator import SE3RandomPoseGenerator
except ImportError:
    from unified_pose_manager import UnifiedPoseManager
    from random_pose_generator import SE3RandomPoseGenerator


class UnifiedPoseGenerator:
    """통합 pose 및 pose_pair 생성기"""
    
    def __init__(self, config_file: str = "config/rigid_body_configs.yaml", 
                 h5_path: str = "/home/dhkang225/2D_sim/data/pose/unified_poses.h5",
                 seed: Optional[int] = None):
        """
        Args:
            config_file: rigid body 설정 파일 경로
            h5_path: 출력 HDF5 파일 경로 (기본: root/data/pose/unified_poses.h5)
            seed: 랜덤 시드
        """
        self.config_file = config_file
        self.h5_path = h5_path
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 초기화
        self.pose_generator = SE3RandomPoseGenerator(config_file, seed)
        self.pose_manager = UnifiedPoseManager(h5_path)
        
        print(f"🚀 UnifiedPoseGenerator initialized")
        print(f"   Config: {config_file}")
        print(f"   Output: {h5_path}")
        print(f"   Seed: {seed}")
    
    def _resolve_environment_path(self, env_path: str) -> str:
        """
        환경 경로 해석 (root/data/pointcloud 기준)
        
        Args:
            env_path: 환경 파일 경로 (상대경로 또는 절대경로)
            
        Returns:
            해석된 절대 경로
        """
        # 기본 pointcloud 디렉토리
        pointcloud_root = Path("/home/dhkang225/2D_sim/data/pointcloud")
        
        # 절대 경로인 경우
        if os.path.isabs(env_path):
            if os.path.exists(env_path):
                return env_path
            else:
                raise FileNotFoundError(f"Environment file not found: {env_path}")
        
        # 상대 경로인 경우 - pointcloud_root 기준으로 검색
        possible_paths = [
            pointcloud_root / env_path,                              # data/pointcloud/circle_env_000000.ply
            pointcloud_root / env_path / f"{env_path}.ply",         # data/pointcloud/circle_env_000000/circle_env_000000.ply
            pointcloud_root / "circles_only" / env_path,            # data/pointcloud/circles_only/circles_only.ply
        ]
        
        # .ply 확장자 자동 추가
        if not env_path.endswith('.ply'):
            possible_paths.extend([
                pointcloud_root / f"{env_path}.ply",
                pointcloud_root / env_path / f"{env_path}.ply"
            ])
        
        for path in possible_paths:
            if path.exists():
                print(f"   Found environment: {path}")
                return str(path)
        
        raise FileNotFoundError(f"Environment file not found. Tried paths: {[str(p) for p in possible_paths]}")
    
    def _extract_env_name(self, env_path: str) -> str:
        """환경 파일 경로에서 환경 이름 추출"""
        env_file = Path(env_path)
        env_name = env_file.stem
        
        # circles_only 등의 특수 케이스 처리
        if env_name == "circles_only":
            return "circles_only"
        
        # circle_env_000000 형태
        return env_name
    
    def _generate_poses(self, env_path: str, rb_id: int, num_poses: int, 
                       safety_margin: float = 0.05, max_attempts: int = 1000) -> List[List[float]]:
        """
        특정 환경-RB에 대해 collision-free pose 생성
        
        Args:
            env_path: 환경 PLY 파일 경로
            rb_id: Rigid body ID
            num_poses: 목표 pose 개수
            safety_margin: 안전 여유거리
            max_attempts: pose당 최대 시도 횟수
            
        Returns:
            생성된 SE(3) pose 리스트
        """
        print(f"   Generating {num_poses} poses for rb_{rb_id}...")
        
        poses = self.pose_generator.generate_multiple_poses(
            rigid_body_id=rb_id,
            ply_file=env_path,
            num_poses=num_poses,
            safety_margin=safety_margin,
            max_attempts=max_attempts
        )
        
        print(f"   ✅ Generated {len(poses)}/{num_poses} collision-free poses")
        return poses
    
    def _calculate_min_pair_distance(self, environment_bounds: tuple) -> float:
        """환경 크기를 기준으로 최소 pair 거리 계산"""
        min_x, max_x, min_y, max_y = environment_bounds
        
        width = max_x - min_x
        height = max_y - min_y
        
        # 가로/세로 중 짧은 것의 0.5배
        min_dimension = min(width, height)
        min_distance = min_dimension * 0.5
        
        return min_distance
    
    def _calculate_pose_distance(self, pose1: List[float], pose2: List[float]) -> float:
        """두 pose 간 유클리드 거리 계산 (중심점 기준)"""
        x1, y1 = pose1[0], pose1[1]
        x2, y2 = pose2[0], pose2[1]
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    def _generate_pose_pairs(self, poses: List[List[float]], num_pairs: int) -> np.ndarray:
        """
        환경 크기 기반 최소 거리 필터링을 적용한 pose_pair 생성
        
        Args:
            poses: SE(3) pose 리스트
            num_pairs: 목표 pair 개수
            
        Returns:
            pose_pair 배열 (M, 12) [init_pose + target_pose]
        """
        if len(poses) < 2:
            print(f"   ⚠️ Need at least 2 poses to generate pairs, got {len(poses)}")
            return np.array([]).reshape(0, 12)
        
        print(f"   Generating {num_pairs} pose pairs from {len(poses)} poses...")
        
        # collision_detector에서 환경 bounds 가져오기
        environment_bounds = self.pose_generator.collision_detector.environment_bounds
        if not environment_bounds:
            print("   ⚠️ Environment bounds not available, using random selection")
            return self._generate_pose_pairs_fallback(poses, num_pairs)
        
        # 최소 거리 계산 (환경 크기의 50%)
        min_distance = self._calculate_min_pair_distance(environment_bounds)
        min_x, max_x, min_y, max_y = environment_bounds
        env_width = max_x - min_x
        env_height = max_y - min_y
        print(f"   Environment size: {env_width:.1f}×{env_height:.1f}m")
        print(f"   Min pair distance: {min_distance:.2f}m (50% of min dimension)")
        
        # 거리 조건을 만족하는 pairs 필터링
        valid_pairs = []
        for i in range(len(poses)):
            for j in range(len(poses)):
                if i != j:  # 자기 자신 제외
                    pose1, pose2 = poses[i], poses[j]
                    distance = self._calculate_pose_distance(pose1, pose2)
                    
                    if distance >= min_distance:
                        # 12차원 배열로 concat: [x,y,z,roll,pitch,yaw] + [x,y,z,roll,pitch,yaw]
                        pair = pose1 + pose2
                        valid_pairs.append(pair)
        
        # 충분한 valid pairs가 있는지 확인
        total_possible = len(poses) * (len(poses) - 1)  # 자기 자신 제외한 모든 조합
        print(f"   Valid pairs found: {len(valid_pairs)}/{total_possible} (distance >= {min_distance:.2f}m)")
        
        if len(valid_pairs) < num_pairs:
            print(f"   ⚠️ Requested {num_pairs} pairs, but only {len(valid_pairs)} valid pairs available")
            print(f"   ⚠️ Using all {len(valid_pairs)} available valid pairs")
            selected_pairs = valid_pairs
        else:
            # 랜덤하게 선택 (중복 없이)
            selected_pairs = random.sample(valid_pairs, num_pairs)
        
        print(f"   ✅ Generated {len(selected_pairs)} pose pairs with min distance {min_distance:.2f}m")
        return np.array(selected_pairs)
    
    def _generate_pose_pairs_fallback(self, poses: List[List[float]], num_pairs: int) -> np.ndarray:
        """환경 bounds 정보가 없을 때 사용하는 기존 랜덤 방식"""
        # 가능한 모든 쌍 생성 (자기 자신 제외)
        all_pairs = []
        for i in range(len(poses)):
            for j in range(len(poses)):
                if i != j:  # 자기 자신 제외
                    init_pose = poses[i]
                    target_pose = poses[j]
                    # 12차원 배열로 concat: [x,y,z,roll,pitch,yaw] + [x,y,z,roll,pitch,yaw]
                    pair = init_pose + target_pose
                    all_pairs.append(pair)
        
        # 요청된 개수만큼 랜덤 선택 (중복 없이)
        if len(all_pairs) < num_pairs:
            print(f"   ⚠️ Requested {num_pairs} pairs, but only {len(all_pairs)} unique pairs possible")
            selected_pairs = all_pairs
        else:
            selected_pairs = random.sample(all_pairs, num_pairs)
        
        print(f"   ✅ Generated {len(selected_pairs)} pose pairs (fallback mode)")
        return np.array(selected_pairs)
    
    def generate_complete_dataset(self, env_path: str, rb_ids: List[int], 
                                 num_poses: int = 100, num_pairs: int = 50,
                                 safety_margin: float = 0.05, max_attempts: int = 1000) -> Dict[str, Any]:
        """
        환경-RB별 complete pose dataset 생성 (pose + pose_pair)
        
        Args:
            env_path: 환경 파일 경로 (상대경로, data/pointcloud 기준)
            rb_ids: Rigid body ID 리스트
            num_poses: 각 RB별 목표 pose 개수
            num_pairs: 각 RB별 목표 pose_pair 개수
            safety_margin: 안전 여유거리
            max_attempts: pose당 최대 시도 횟수
            
        Returns:
            결과 리포트 딕셔너리
        """
        start_time = time.time()
        
        # 환경 경로 해석
        resolved_env_path = self._resolve_environment_path(env_path)
        env_name = self._extract_env_name(resolved_env_path)
        
        print(f"🚀 Generating complete dataset for {env_name}")
        print(f"   Environment: {resolved_env_path}")
        print(f"   Rigid bodies: {rb_ids}")
        print(f"   Target poses per RB: {num_poses}")
        print(f"   Target pairs per RB: {num_pairs}")
        
        results = {
            'env_name': env_name,
            'env_path': resolved_env_path,
            'rb_ids': rb_ids,
            'target_poses': num_poses,
            'target_pairs': num_pairs,
            'rb_results': {},
            'total_time': 0,
            'success': True
        }
        
        # RB별로 pose + pose_pair 생성
        for rb_id in rb_ids:
            print(f"\n📍 Processing rb_{rb_id}...")
            rb_start_time = time.time()
            
            try:
                # 1. collision-free poses 생성
                poses = self._generate_poses(
                    resolved_env_path, rb_id, num_poses, safety_margin, max_attempts
                )
                
                if len(poses) == 0:
                    print(f"   ❌ No valid poses generated for rb_{rb_id}")
                    results['rb_results'][rb_id] = {
                        'poses_generated': 0,
                        'pairs_generated': 0,
                        'success': False,
                        'time': time.time() - rb_start_time
                    }
                    results['success'] = False
                    continue
                
                # 2. pose_pairs 생성
                pose_pairs = self._generate_pose_pairs(poses, num_pairs)
                
                # 3. HDF5에 저장
                pose_metadata = {
                    'safety_margin': safety_margin,
                    'max_attempts': max_attempts,
                    'success_rate': len(poses) / num_poses * 100 if num_poses > 0 else 0,
                    'rb_config': self.pose_generator.get_rigid_body_config(rb_id).__dict__ if self.pose_generator.get_rigid_body_config(rb_id) else {}
                }
                
                pair_metadata = {
                    'generation_method': 'random_sampling_without_replacement',
                    'source_poses': len(poses),
                    'generation_success_rate': len(pose_pairs) / num_pairs * 100 if num_pairs > 0 else 0
                }
                
                # poses를 numpy 배열로 변환
                poses_array = np.array(poses)
                
                # HDF5에 저장
                pose_success = self.pose_manager.add_poses(env_name, rb_id, poses_array, pose_metadata)
                pair_success = self.pose_manager.add_pose_pairs(env_name, rb_id, pose_pairs, pair_metadata)
                
                rb_time = time.time() - rb_start_time
                results['rb_results'][rb_id] = {
                    'poses_generated': len(poses),
                    'pairs_generated': len(pose_pairs),
                    'pose_save_success': pose_success,
                    'pair_save_success': pair_success,
                    'success': pose_success and pair_success,
                    'time': rb_time
                }
                
                print(f"   ✅ rb_{rb_id} completed in {rb_time:.1f}s")
                
            except Exception as e:
                print(f"   ❌ Error processing rb_{rb_id}: {e}")
                results['rb_results'][rb_id] = {
                    'poses_generated': 0,
                    'pairs_generated': 0,
                    'success': False,
                    'error': str(e),
                    'time': time.time() - rb_start_time
                }
                results['success'] = False
        
        total_time = time.time() - start_time
        results['total_time'] = total_time
        
        # 요약 출력
        self._print_results_summary(results)
        
        return results
    
    def _print_results_summary(self, results: Dict[str, Any]):
        """결과 요약 출력"""
        print(f"\n📊 Generation Summary for {results['env_name']}")
        print(f"   Total time: {results['total_time']:.1f}s")
        print(f"   Overall success: {'✅' if results['success'] else '❌'}")
        
        total_poses = sum(rb['poses_generated'] for rb in results['rb_results'].values())
        total_pairs = sum(rb['pairs_generated'] for rb in results['rb_results'].values())
        
        print(f"   Total poses generated: {total_poses}")
        print(f"   Total pairs generated: {total_pairs}")
        
        print(f"\n   Per rigid body:")
        for rb_id, rb_result in results['rb_results'].items():
            status = "✅" if rb_result['success'] else "❌"
            print(f"     rb_{rb_id}: {status} {rb_result['poses_generated']} poses, {rb_result['pairs_generated']} pairs ({rb_result['time']:.1f}s)")


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 Testing UnifiedPoseGenerator...")
    
    # 테스트용 HDF5 파일
    test_h5_path = "/tmp/test_unified_poses.h5"
    if os.path.exists(test_h5_path):
        os.remove(test_h5_path)
    
    try:
        # 생성기 초기화
        generator = UnifiedPoseGenerator(
            config_file="config/rigid_body_configs.yaml",
            h5_path=test_h5_path,
            seed=42
        )
        
        # 테스트 데이터셋 생성
        result = generator.generate_complete_dataset(
            env_path="circles_only.ply",  # data/pointcloud에서 찾을 상대경로
            rb_ids=[0],  # 테스트용으로 RB 0만
            num_poses=5,  # 테스트용 적은 개수
            num_pairs=3
        )
        
        print(f"\nTest result: {result}")
        
        # 저장된 데이터 확인
        summary = generator.pose_manager.get_summary()
        print(f"HDF5 Summary: {summary}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        # 정리
        if os.path.exists(test_h5_path):
            os.remove(test_h5_path)
    
    print("🎉 Test completed")
