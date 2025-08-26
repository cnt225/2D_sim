#!/usr/bin/env python3
"""
SE(3) Trajectory GT Generator
기존 SE(3) pose pairs를 사용하여 RRT-Connect로 trajectory GT 생성

주요 기능:
- SE(3) pose pairs 로드
- RRT-Connect로 trajectory 계획
- collision_detector와 연동된 isStateValid
- Trajectory GT JSON 저장 (기존)
- HDF5 직접 저장 (신규)
"""

import json
import time
import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import sys

# SE(3) RRT planner import
from rrt_connect import create_se3_planner, SE3TrajectoryResult

# HDF5 tools import
sys.path.append(str(Path(__file__).parent.parent / 'hdf5_tools'))
from hdf5_schema_creator import create_hdf5_schema, add_environment_metadata, add_rigid_body_metadata, add_generation_settings, create_pose_pair_group

# SE3 functions import
sys.path.append(str(Path(__file__).parent.parent.parent / 'utils'))
from SE3_functions import trajectory_euler_to_quaternion, euler_6d_to_quaternion_7d


class SE3TrajectoryGTGenerator:
    """SE(3) Trajectory GT 생성기"""
    
    def __init__(self, rigid_body_id: int, pointcloud_file: str = None):
        """
        Args:
            rigid_body_id: Rigid body ID (0-3)
            pointcloud_file: 환경 PLY 파일 경로 (optional)
        """
        self.rigid_body_id = rigid_body_id
        self.pointcloud_file = pointcloud_file
        
        # SE(3) RRT planner 생성
        self.planner = create_se3_planner(rigid_body_id, pointcloud_file)
        
        # 통계
        self.stats = {
            "total_pairs": 0,
            "successful_plans": 0,
            "failed_plans": 0,
            "total_planning_time": 0.0,
            "avg_planning_time": 0.0,
            "avg_path_length": 0.0,
            "avg_waypoints": 0.0
        }
        
        print(f"✅ SE(3) Trajectory GT Generator initialized")
        print(f"   - Rigid body ID: {rigid_body_id}")
        print(f"   - Environment: {pointcloud_file or 'empty'}")
    
    def load_pose_pairs(self, pose_pairs_file: str) -> List[Dict[str, Any]]:
        """SE(3) pose pairs JSON 파일 로드"""
        try:
            with open(pose_pairs_file, 'r') as f:
                data = json.load(f)
            
            # pose_pairs.data에서 실제 pairs 추출
            if 'pose_pairs' in data and 'data' in data['pose_pairs']:
                pairs = data['pose_pairs']['data']
            else:
                # 호환성을 위해 직접 리스트인 경우도 처리
                pairs = data if isinstance(data, list) else []
            
            print(f"✅ Loaded pose pairs from: {pose_pairs_file}")
            print(f"   - Total pairs: {len(pairs)}")
            
            return pairs
            
        except Exception as e:
            print(f"❌ Failed to load pose pairs: {e}")
            return []
    
    def generate_trajectory(self, start_pose: List[float], goal_pose: List[float], 
                          max_planning_time: float = 5.0) -> Optional[SE3TrajectoryResult]:
        """단일 trajectory 생성"""
        try:
            result = self.planner.plan_trajectory(start_pose, goal_pose, max_planning_time)
            
            # 통계 업데이트
            self.stats["total_planning_time"] += result.planning_time
            
            if result.success:
                self.stats["successful_plans"] += 1
            else:
                self.stats["failed_plans"] += 1
            
            return result
            
        except Exception as e:
            print(f"❌ Trajectory generation error: {e}")
            self.stats["failed_plans"] += 1
            return None
    
    def generate_trajectories_from_poses(self, pose_pairs: List[Dict[str, Any]], 
                                       max_planning_time: float = 5.0,
                                       output_dir: str = "trajectories") -> List[Dict[str, Any]]:
        """여러 pose pairs에서 trajectories 생성"""
        
        self.stats["total_pairs"] = len(pose_pairs)
        trajectories = []
        
        print(f"🚀 Generating {len(pose_pairs)} trajectories...")
        
        # Output directory 생성
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, pose_pair in enumerate(pose_pairs):
            print(f"\n--- Trajectory {i+1}/{len(pose_pairs)} ---")
            
            try:
                # Pose pair에서 start, goal 추출
                start_pose = pose_pair.get("init")
                goal_pose = pose_pair.get("target")
                
                if not start_pose or not goal_pose:
                    print(f"⚠️ Invalid pose pair format, skipping...")
                    continue
                
                print(f"Start: [{start_pose[0]:.2f}, {start_pose[1]:.2f}, {start_pose[5]:.2f}]")
                print(f"Goal:  [{goal_pose[0]:.2f}, {goal_pose[1]:.2f}, {goal_pose[5]:.2f}]")
                
                # Trajectory 생성
                result = self.generate_trajectory(start_pose, goal_pose, max_planning_time)
                
                if result and result.success:
                    print(f"✅ Success: {result.num_waypoints} waypoints, "
                          f"time: {result.planning_time:.3f}s")
                    
                    # Trajectory data 구성
                    trajectory_data = {
                        "pair_id": i,
                        "trajectory_id": f"traj_rb{self.rigid_body_id}_{i:06d}",
                        "rigid_body": {
                            "id": self.rigid_body_id,
                            "type": self.planner.config.name
                        },
                        "environment": {
                            "name": Path(self.pointcloud_file).stem if self.pointcloud_file else "empty",
                            "ply_file": self.pointcloud_file
                        },
                        "start_pose": start_pose,
                        "goal_pose": goal_pose,
                        "path": {
                            "data": result.trajectory,
                            "format": "se3_trajectory",
                            "length": result.num_waypoints,
                            "planning_time": result.planning_time,
                            "path_length": result.path_length,
                            "timestamps": result.timestamps
                        },
                        "planning_method": "RRT-Connect",
                        "generation_info": {
                            "success": True,
                            "max_planning_time": max_planning_time,
                            "planner_settings": result.planner_settings,
                            "metadata": result.metadata
                        }
                    }
                    
                    trajectories.append(trajectory_data)
                    
                    # Individual trajectory 파일 저장
                    traj_file = output_path / f"trajectory_{i:06d}.json"
                    with open(traj_file, 'w') as f:
                        json.dump(trajectory_data, f, indent=2)
                    
                else:
                    print(f"❌ Failed: {result.metadata if result else 'Unknown error'}")
            
            except Exception as e:
                print(f"❌ Error processing pair {i}: {e}")
                self.stats["failed_plans"] += 1
        
        # 통계 계산
        self._calculate_final_stats(trajectories)
        
        return trajectories
    
    def _calculate_final_stats(self, trajectories: List[Dict[str, Any]]):
        """최종 통계 계산"""
        if self.stats["total_pairs"] > 0:
            self.stats["avg_planning_time"] = self.stats["total_planning_time"] / self.stats["total_pairs"]
        
        if trajectories:
            self.stats["avg_path_length"] = sum(t["path"]["path_length"] for t in trajectories) / len(trajectories)
            self.stats["avg_waypoints"] = sum(t["path"]["length"] for t in trajectories) / len(trajectories)
    
    def save_trajectory_batch(self, trajectories: List[Dict[str, Any]], output_file: str):
        """Trajectory batch JSON 저장"""
        batch_data = {
            "batch_info": {
                "rigid_body_id": self.rigid_body_id,
                "environment_file": self.pointcloud_file,
                "total_trajectories": len(trajectories),
                "generation_time": time.strftime('%Y-%m-%d %H:%M:%S')
            },
            "statistics": self.stats,
            "trajectories": trajectories
        }
        
        with open(output_file, 'w') as f:
            json.dump(batch_data, f, indent=2)
        
        print(f"✅ Trajectory batch saved to: {output_file}")
    
    def print_statistics(self):
        """통계 출력"""
        print(f"\n📊 SE(3) Trajectory Generation Statistics:")
        print(f"   Total pairs: {self.stats['total_pairs']}")
        print(f"   Successful: {self.stats['successful_plans']}")
        print(f"   Failed: {self.stats['failed_plans']}")
        if self.stats['total_pairs'] > 0:
            success_rate = self.stats['successful_plans'] / self.stats['total_pairs'] * 100
            print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Avg planning time: {self.stats['avg_planning_time']:.3f}s")
        print(f"   Avg path length: {self.stats['avg_path_length']:.3f}")
        print(f"   Avg waypoints: {self.stats['avg_waypoints']:.1f}")


class SE3TrajectoryHDF5Generator:
    """HDF5 직접 저장을 지원하는 SE(3) 궤적 생성기"""
    
    def __init__(self, rigid_body_id: int, hdf5_path: str, pointcloud_file: str = None):
        """
        Args:
            rigid_body_id: Rigid body ID (0-3)
            hdf5_path: HDF5 파일 경로
            pointcloud_file: 환경 PLY 파일 경로 (optional)
        """
        self.rigid_body_id = rigid_body_id
        self.hdf5_path = Path(hdf5_path)
        self.pointcloud_file = pointcloud_file
        
        # SE(3) RRT planner 생성
        self.planner = create_se3_planner(rigid_body_id, pointcloud_file)
        
        # HDF5 파일 관리
        self.hdf5_file = None
        self._initialize_hdf5()
        
        # 통계
        self.stats = {
            "total_pairs": 0,
            "successful_plans": 0,
            "failed_plans": 0,
            "total_planning_time": 0.0,
            "avg_planning_time": 0.0,
            "avg_path_length": 0.0,
            "avg_waypoints": 0.0
        }
        
        print(f"✅ SE(3) Trajectory HDF5 Generator initialized")
        print(f"   - Rigid body ID: {rigid_body_id}")
        print(f"   - HDF5 file: {self.hdf5_path}")
        print(f"   - Environment: {pointcloud_file or 'empty'}")
    
    def _initialize_hdf5(self):
        """HDF5 파일 초기화"""
        # HDF5 스키마 생성 (기존 파일이 있으면 추가 모드)
        if self.hdf5_path.exists():
            print(f"📂 Opening existing HDF5 file: {self.hdf5_path}")
            self.hdf5_file = h5py.File(self.hdf5_path, 'a')
        else:
            print(f"🚀 Creating new HDF5 file: {self.hdf5_path}")
            self.hdf5_file = create_hdf5_schema(str(self.hdf5_path))
        
        # 로봇 메타데이터 추가
        self._add_rigid_body_metadata()
        
        # 생성 설정 추가
        self._add_generation_settings()
    
    def _add_rigid_body_metadata(self):
        """로봇 메타데이터 HDF5에 추가"""
        rb_data = {
            'rigid_body_id': self.rigid_body_id,
            'name': self.planner.config.name,
            'type': 'SE3_rigid_body',
            'description': f'SE(3) rigid body configuration for planner',
            'planner_config': str(self.planner.config)
        }
        add_rigid_body_metadata(self.hdf5_file, rb_data)
    
    def _add_generation_settings(self):
        """생성 설정 HDF5에 추가"""
        settings = {
            'planner': 'RRT-Connect',
            'rigid_body_id': self.rigid_body_id,
            'data_format': '7d_quaternion',
            'conversion_method': '6d_to_7d_automatic',
            'environment_file': self.pointcloud_file or 'empty'
        }
        add_generation_settings(self.hdf5_file, settings)
    
    def load_and_convert_pose_pairs(self, pose_pairs_file: str, env_id: str) -> np.ndarray:
        """
        Pose pairs JSON 파일을 로드하고 7D 형태로 변환하여 HDF5에 저장
        
        Args:
            pose_pairs_file: Pose pairs JSON 파일 경로
            env_id: 환경 ID (예: 'circle_env_000000')
        
        Returns:
            [N, 2, 7] 형태의 변환된 pose pairs
        """
        try:
            with open(pose_pairs_file, 'r') as f:
                data = json.load(f)
            
            # pose_pairs.data에서 실제 pairs 추출
            if 'pose_pairs' in data and 'data' in data['pose_pairs']:
                pairs = data['pose_pairs']['data']
            else:
                pairs = data if isinstance(data, list) else []
            
            if not pairs:
                raise ValueError(f"No pose pairs found in {pose_pairs_file}")
            
            print(f"📥 Loading pose pairs from: {pose_pairs_file}")
            print(f"   Pairs: {len(pairs)}")
            
            # 6D → 7D 변환
            pairs_7d = np.zeros((len(pairs), 2, 7))
            
            for i, pair in enumerate(pairs):
                # init pose 변환
                init_6d = np.array(pair.get("init", [0, 0, 0, 0, 0, 0]))
                init_7d = euler_6d_to_quaternion_7d(init_6d)
                pairs_7d[i, 0, :] = init_7d
                
                # target pose 변환  
                target_6d = np.array(pair.get("target", [0, 0, 0, 0, 0, 0]))
                target_7d = euler_6d_to_quaternion_7d(target_6d)
                pairs_7d[i, 1, :] = target_7d
            
            # HDF5에 pose pairs 저장
            create_pose_pair_group(self.hdf5_file, env_id, pairs_7d)
            
            # 환경 메타데이터 추가
            env_metadata = {
                'env_id': env_id,
                'name': f'Environment {env_id}',
                'source_file': pose_pairs_file,
                'pair_count': len(pairs),
                'data_format': '7d_quaternion'
            }
            add_environment_metadata(self.hdf5_file, env_metadata)
            
            print(f"✅ Pose pairs converted and stored: {len(pairs)} pairs")
            return pairs_7d
            
        except Exception as e:
            print(f"❌ Failed to load pose pairs: {e}")
            raise
    
    def generate_trajectory_to_hdf5(self, env_id: str, pair_index: int, 
                                   start_pose: List[float], goal_pose: List[float],
                                   max_planning_time: float = 5.0) -> bool:
        """
        궤적 생성 후 HDF5에 직접 저장 (7D 형식)
        
        Args:
            env_id: 환경 ID
            pair_index: Pose pair 인덱스
            start_pose: 시작 pose [x,y,z,rx,ry,rz] (6D)
            goal_pose: 목표 pose [x,y,z,rx,ry,rz] (6D)
            max_planning_time: 최대 계획 시간
            
        Returns:
            bool: 성공 여부
        """
        try:
            # RRT 궤적 생성
            result = self.planner.plan_trajectory(start_pose, goal_pose, max_planning_time)
            
            # 통계 업데이트
            self.stats["total_planning_time"] += result.planning_time
            
            if not result.success:
                self.stats["failed_plans"] += 1
                print(f"❌ Planning failed for pair {pair_index}")
                return False
            
            self.stats["successful_plans"] += 1
            
            print(f"✅ Planning success: {result.num_waypoints} waypoints, "
                  f"time: {result.planning_time:.3f}s")
            
            # RRT 결과 (6D) → 7D 변환
            trajectory_6d = np.array(result.trajectory)  # [N, 6]
            trajectory_7d = trajectory_euler_to_quaternion(trajectory_6d)  # [N, 7]
            
            # HDF5 경로 생성
            raw_path = f"trajectories/raw/{env_id}/rb_{self.rigid_body_id}"
            
            # HDF5 그룹 생성 (존재하지 않는 경우)
            if raw_path not in self.hdf5_file:
                self.hdf5_file.create_group(raw_path)
            
            raw_group = self.hdf5_file[raw_path]
            
            # 궤적 데이터셋 생성
            traj_name = f"traj_{pair_index:06d}"
            if traj_name in raw_group:
                del raw_group[traj_name]  # 기존 데이터 교체
            
            traj_dataset = raw_group.create_dataset(
                traj_name, 
                data=trajectory_7d,
                compression='gzip',
                compression_opts=6
            )
            
            # 메타데이터 속성 추가
            traj_dataset.attrs['format'] = '[x, y, z, qw, qx, qy, qz]'
            traj_dataset.attrs['pair_index'] = pair_index
            traj_dataset.attrs['env_id'] = env_id
            traj_dataset.attrs['rigid_body_id'] = self.rigid_body_id
            traj_dataset.attrs['planning_time'] = result.planning_time
            traj_dataset.attrs['path_length'] = result.path_length
            traj_dataset.attrs['num_waypoints'] = result.num_waypoints
            traj_dataset.attrs['success'] = True
            traj_dataset.attrs['planner'] = 'RRT-Connect'
            traj_dataset.attrs['conversion'] = '6d_to_7d'
            
            # 시간 정보도 저장
            if hasattr(result, 'timestamps') and result.timestamps:
                traj_dataset.attrs['timestamps'] = result.timestamps
            
            print(f"📁 Trajectory saved to HDF5: {raw_path}/{traj_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error generating trajectory to HDF5: {e}")
            self.stats["failed_plans"] += 1
            return False
    
    def generate_trajectories_from_pose_pairs(self, pose_pairs_file: str, env_id: str, 
                                            max_planning_time: float = 5.0,
                                            start_index: int = 0, 
                                            end_index: int = None) -> int:
        """
        Pose pairs에서 궤적들을 생성하여 HDF5에 저장
        
        Args:
            pose_pairs_file: Pose pairs JSON 파일
            env_id: 환경 ID
            max_planning_time: 최대 계획 시간
            start_index: 시작 인덱스
            end_index: 끝 인덱스 (None이면 끝까지)
            
        Returns:
            int: 성공한 궤적 수
        """
        # Pose pairs 로드 및 변환
        pairs_7d = self.load_and_convert_pose_pairs(pose_pairs_file, env_id)
        
        # 인덱스 범위 설정
        total_pairs = len(pairs_7d)
        if end_index is None:
            end_index = total_pairs
        end_index = min(end_index, total_pairs)
        
        print(f"🚀 Generating trajectories for {env_id}")
        print(f"   Range: {start_index} to {end_index-1} ({end_index-start_index} pairs)")
        
        self.stats["total_pairs"] = end_index - start_index
        success_count = 0
        
        for i in range(start_index, end_index):
            print(f"\n--- Trajectory {i+1}/{total_pairs} ---")
            
            # 7D → 6D 변환 (RRT는 6D 입력 필요)
            init_7d = pairs_7d[i, 0, :]
            target_7d = pairs_7d[i, 1, :]
            
            # 간단한 7D → 6D 변환 (정확한 변환은 quaternion_7d_to_euler_6d 사용)
            init_6d = [init_7d[0], init_7d[1], init_7d[2], 0, 0, 0]  # 임시로 회전은 0
            target_6d = [target_7d[0], target_7d[1], target_7d[2], 0, 0, 0]
            
            success = self.generate_trajectory_to_hdf5(
                env_id, i, init_6d, target_6d, max_planning_time
            )
            
            if success:
                success_count += 1
        
        # 최종 통계 계산
        self._calculate_final_stats()
        
        print(f"\n🎯 Generation complete:")
        print(f"   Total processed: {end_index - start_index}")
        print(f"   Successful: {success_count}")
        print(f"   Failed: {(end_index - start_index) - success_count}")
        
        return success_count
    
    def _calculate_final_stats(self):
        """최종 통계 계산"""
        if self.stats["total_pairs"] > 0:
            self.stats["avg_planning_time"] = self.stats["total_planning_time"] / self.stats["total_pairs"]
        
        # HDF5에서 궤적 통계 계산
        if self.stats["successful_plans"] > 0:
            total_length = 0
            total_waypoints = 0
            count = 0
            
            # 모든 raw 궤적에서 통계 수집
            raw_group = self.hdf5_file.get('trajectories/raw')
            if raw_group:
                for env_key in raw_group.keys():
                    env_group = raw_group[env_key]
                    for rb_key in env_group.keys():
                        rb_group = env_group[rb_key]
                        for traj_key in rb_group.keys():
                            traj_dataset = rb_group[traj_key]
                            if 'path_length' in traj_dataset.attrs:
                                total_length += traj_dataset.attrs['path_length']
                            if 'num_waypoints' in traj_dataset.attrs:
                                total_waypoints += traj_dataset.attrs['num_waypoints']
                            count += 1
            
            if count > 0:
                self.stats["avg_path_length"] = total_length / count
                self.stats["avg_waypoints"] = total_waypoints / count
    
    def close(self):
        """HDF5 파일 닫기"""
        if self.hdf5_file is not None:
            self.hdf5_file.close()
            self.hdf5_file = None
            print(f"📁 HDF5 file closed: {self.hdf5_path}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def print_statistics(self):
        """통계 출력"""
        print(f"\n📊 SE(3) HDF5 Trajectory Generation Statistics:")
        print(f"   Total pairs: {self.stats['total_pairs']}")
        print(f"   Successful: {self.stats['successful_plans']}")
        print(f"   Failed: {self.stats['failed_plans']}")
        if self.stats['total_pairs'] > 0:
            success_rate = self.stats['successful_plans'] / self.stats['total_pairs'] * 100
            print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Avg planning time: {self.stats['avg_planning_time']:.3f}s")
        print(f"   Avg path length: {self.stats['avg_path_length']:.3f}")
        print(f"   Avg waypoints: {self.stats['avg_waypoints']:.1f}")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Generate SE(3) trajectory GT from pose pairs")
    parser.add_argument("--rigid_body_id", type=int, default=3, help="Rigid body ID (0-3)")
    parser.add_argument("--pose_pairs_file", required=True, help="Input pose pairs JSON file")
    parser.add_argument("--pointcloud_file", help="Environment PLY file (optional)")
    parser.add_argument("--output_dir", default="trajectories", help="Output directory")
    parser.add_argument("--max_planning_time", type=float, default=5.0, help="Max planning time per trajectory")
    parser.add_argument("--batch_file", help="Output batch JSON file")
    
    args = parser.parse_args()
    
    # Generator 생성
    generator = SE3TrajectoryGTGenerator(args.rigid_body_id, args.pointcloud_file)
    
    # Pose pairs 로드
    pose_pairs = generator.load_pose_pairs(args.pose_pairs_file)
    if not pose_pairs:
        print("❌ No pose pairs loaded, exiting...")
        return
    
    # Trajectories 생성
    trajectories = generator.generate_trajectories_from_poses(
        pose_pairs, 
        max_planning_time=args.max_planning_time,
        output_dir=args.output_dir
    )
    
    # Batch 파일 저장
    if args.batch_file:
        generator.save_trajectory_batch(trajectories, args.batch_file)
    
    # 통계 출력
    generator.print_statistics()


if __name__ == "__main__":
    # Check for HDF5 vs JSON mode
    if len(__import__('sys').argv) == 1:
        print("🧪 SE(3) Trajectory Generator Examples")
        print("\nChoose mode:")
        print("1. Traditional JSON mode")
        print("2. New HDF5 mode")
        
        mode = input("Enter mode (1 or 2): ").strip()
        
        if mode == "1":
            print("\n🧪 Traditional JSON Mode Example")
            
            # Test parameters
            rigid_body_id = 3
            pose_pairs_file = "../pose/pose_pairs/elongated_ellipse_poses.json"
            pointcloud_file = "../../simulation/robot_simulation/legacy/simple_endeffector_sim/data/pointcloud/circles_only/circles_only.ply"
            
            try:
                # Create generator
                generator = SE3TrajectoryGTGenerator(rigid_body_id, pointcloud_file)
                
                # Load pose pairs
                pose_pairs = generator.load_pose_pairs(pose_pairs_file)
                
                if pose_pairs:
                    # Generate first 3 trajectories for testing
                    test_pairs = pose_pairs[:3]
                    trajectories = generator.generate_trajectories_from_poses(
                        test_pairs, 
                        max_planning_time=3.0,
                        output_dir="results"
                    )
                    
                    # Save batch
                    generator.save_trajectory_batch(trajectories, "test_trajectory_batch.json")
                    generator.print_statistics()
                else:
                    print("❌ No pose pairs found for testing")
                    
            except Exception as e:
                print(f"❌ JSON example failed: {e}")
                import traceback
                traceback.print_exc()
        
        elif mode == "2":
            print("\n🧪 New HDF5 Mode Example")
            
            # Test parameters
            rigid_body_id = 3
            pose_pairs_file = "../pose/pose_pairs/elongated_ellipse_poses.json"
            pointcloud_file = "../../simulation/robot_simulation/legacy/simple_endeffector_sim/data/pointcloud/circles_only/circles_only.ply"
            hdf5_path = "trajectory_dataset.h5"
            env_id = "circle_env_000000"
            
            try:
                # Create HDF5 generator
                with SE3TrajectoryHDF5Generator(rigid_body_id, hdf5_path, pointcloud_file) as generator:
                    
                    print(f"🚀 Testing HDF5 trajectory generation...")
                    
                    # Generate trajectories for first 5 pose pairs
                    success_count = generator.generate_trajectories_from_pose_pairs(
                        pose_pairs_file,
                        env_id,
                        max_planning_time=3.0,
                        start_index=0,
                        end_index=5
                    )
                    
                    generator.print_statistics()
                    
                    print(f"\n✅ HDF5 example completed!")
                    print(f"   Successfully generated: {success_count} trajectories")
                    print(f"   HDF5 file: {hdf5_path}")
                    
            except Exception as e:
                print(f"❌ HDF5 example failed: {e}")
                import traceback
                traceback.print_exc()
        
        else:
            print("❌ Invalid mode selected")
    else:
        main() 