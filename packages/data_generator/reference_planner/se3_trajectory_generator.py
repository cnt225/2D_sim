#!/usr/bin/env python3
"""
SE(3) Trajectory GT Generator
기존 SE(3) pose pairs를 사용하여 RRT-Connect로 trajectory GT 생성

주요 기능:
- SE(3) pose pairs 로드
- RRT-Connect로 trajectory 계획
- collision_detector와 연동된 isStateValid
- Trajectory GT JSON 저장
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

# SE(3) RRT planner import
from rrt_connect import create_se3_planner, SE3TrajectoryResult


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
    # Example usage
    if len(__import__('sys').argv) == 1:
        print("🧪 SE(3) Trajectory GT Generator Example")
        
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
            print(f"❌ Example failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        main() 