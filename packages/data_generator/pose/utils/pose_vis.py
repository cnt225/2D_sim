#!/usr/bin/env python3
"""
HDF5 기반 SE(3) Pose Visualizer
통합 HDF5 파일에서 pose 데이터를 읽어서 시각화

사용법:
    from utils.pose_vis import HDF5PoseVisualizer
    
    visualizer = HDF5PoseVisualizer("unified_poses.h5")
    visualizer.visualize_poses_from_hdf5("circle_env_000000", 3, save_image=True)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import math

try:
    from ..unified_pose_manager import UnifiedPoseManager
    from ..collision_detector import RigidBodyCollisionDetector
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from unified_pose_manager import UnifiedPoseManager
    from collision_detector import RigidBodyCollisionDetector


class HDF5PoseVisualizer:
    """HDF5 기반 SE(3) 포즈 시각화기"""
    
    def __init__(self, h5_path: str, config_file: str = "config/rigid_body_configs.yaml"):
        """
        Args:
            h5_path: 통합 HDF5 파일 경로
            config_file: rigid body 설정 파일 경로
        """
        self.pose_manager = UnifiedPoseManager(h5_path)
        self.collision_detector = RigidBodyCollisionDetector(config_file)
        
    def visualize_poses_from_hdf5(self, 
                                 env_name: str,
                                 rb_id: int,
                                 save_image: bool = False,
                                 output_file: str = None,
                                 show_plot: bool = True,
                                 max_poses_to_show: int = 20,
                                 show_pairs: bool = False) -> bool:
        """
        HDF5에서 SE(3) pose들을 로드하여 시각화
        
        Args:
            env_name: 환경 이름
            rb_id: Rigid body ID
            save_image: 이미지 저장 여부
            output_file: 출력 이미지 파일명
            show_plot: 플롯 표시 여부
            max_poses_to_show: 표시할 최대 포즈 개수
            show_pairs: pose pair 화살표 표시 여부
            
        Returns:
            bool: 성공 여부
        """
        
        try:
            # HDF5에서 pose 데이터 로드
            poses, pose_metadata = self.pose_manager.get_poses(env_name, rb_id)
            if poses is None:
                print(f"❌ No pose data found for {env_name}/rb_{rb_id}")
                return False
            
            # pose_pairs 데이터 로드 (옵션)
            pairs = None
            if show_pairs:
                pairs, _ = self.pose_manager.get_pose_pairs(env_name, rb_id)
            
            # 환경 데이터 로드
            environment_points = self._load_environment_data_from_metadata(pose_metadata, env_name)
            
            # rigid body 설정 로드
            rb_config = self.collision_detector.get_rigid_body_config(rb_id)
            if rb_config is None:
                print(f"❌ Rigid body config not found for ID {rb_id}")
                return False
            
            print(f"📂 Loaded HDF5 pose data:")
            print(f"   Environment: {env_name}")
            print(f"   Rigid body: {rb_config.name}")
            print(f"   Poses: {len(poses)}")
            print(f"   Creation time: {pose_metadata.get('creation_time', 'N/A')}")
            
            # 시각화 생성
            self._create_hdf5_visualization(
                poses, pairs, environment_points, rb_config, env_name,
                save_image, output_file, show_plot, max_poses_to_show, show_pairs
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
            return False
    
    def _load_environment_data_from_metadata(self, pose_metadata: Dict[str, Any], env_name: str) -> np.ndarray:
        """메타데이터에서 환경 파일 경로를 추론하여 환경 데이터 로드"""
        
        # 환경 파일 경로 추론
        pointcloud_root = Path("/home/dhkang225/2D_sim/data/pointcloud")
        possible_paths = [
            pointcloud_root / "circles_only" / f"{env_name}.ply",
            pointcloud_root / env_name / f"{env_name}.ply",
            pointcloud_root / f"{env_name}.ply"
        ]
        
        ply_file = None
        for path in possible_paths:
            if path.exists():
                ply_file = str(path)
                break
        
        if not ply_file:
            print(f"⚠️ Environment file not found for {env_name}, using empty environment")
            return np.array([])
        
        # PLY 파일 읽기
        points = []
        try:
            with open(ply_file, 'r') as f:
                # 헤더 스킵
                line = f.readline()
                while not line.startswith('end_header'):
                    line = f.readline()
                
                # 포인트 데이터 읽기
                for line in f:
                    if line.strip():
                        coords = line.strip().split()
                        if len(coords) >= 3:
                            x, y = float(coords[0]), float(coords[1])
                            points.append([x, y])
            
            print(f"   Loaded {len(points)} environment points from {ply_file}")
            
        except Exception as e:
            print(f"⚠️ Failed to load environment points: {e}")
        
        return np.array(points)
    
    def _create_hdf5_visualization(self, 
                                  poses: np.ndarray,
                                  pairs: Optional[np.ndarray],
                                  environment_points: np.ndarray,
                                  rb_config,
                                  env_name: str,
                                  save_image: bool = False,
                                  output_file: str = None,
                                  show_plot: bool = True,
                                  max_poses_to_show: int = 20,
                                  show_pairs: bool = False) -> None:
        """HDF5 데이터를 사용한 시각화 생성"""
        
        # 표시할 포즈 수 제한
        original_count = len(poses)
        if len(poses) > max_poses_to_show:
            poses = poses[:max_poses_to_show]
            print(f"   Showing first {max_poses_to_show} poses out of {original_count}")
        
        # 플롯 설정
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # 환경 포인트 그리기
        if len(environment_points) > 0:
            ax.scatter(environment_points[:, 0], environment_points[:, 1], 
                      c='red', s=20, alpha=0.6, label='Obstacles')
        
        # SE(3) 포즈들 그리기
        self._draw_se3_poses(ax, poses, rb_config)
        
        # Pose pairs 화살표 그리기 (옵션)
        if show_pairs and pairs is not None:
            self._draw_pose_pairs(ax, pairs, max_pairs_to_show=min(5, len(pairs)))
        
        # 플롯 설정
        self._setup_hdf5_plot(ax, env_name, rb_config, len(poses), show_pairs)
        
        # 이미지 저장 또는 표시
        if save_image:
            # data/visualized/pose 디렉토리 생성
            result_dir = Path("/home/dhkang225/2D_sim/data/visualized/pose")
            result_dir.mkdir(parents=True, exist_ok=True)
            
            if output_file is None:
                suffix = "_with_pairs" if show_pairs else ""
                output_file = f"{env_name}_rb_{rb_config.id}_poses{suffix}.png"
            
            # 출력 파일 경로 설정
            if not os.path.isabs(output_file):
                output_file = result_dir / output_file
            
            plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"💾 Saved visualization to: {output_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def _draw_se3_poses(self, ax, poses: np.ndarray, rb_config) -> None:
        """SE(3) 포즈들을 타원체로 그리기"""
        
        for i, pose in enumerate(poses):
            x, y, z, roll, pitch, yaw = pose
            
            # 타원 생성 (matplotlib ellipse)
            ellipse = patches.Ellipse(
                (x, y), 
                width=2*rb_config.semi_major_axis,  # 전체 너비 (지름)
                height=2*rb_config.semi_minor_axis,  # 전체 높이 (지름) 
                angle=math.degrees(yaw),  # 각도를 도수로 변환
                facecolor=rb_config.color,
                edgecolor='black',
                alpha=0.7,
                linewidth=1
            )
            
            ax.add_patch(ellipse)
            
            # 방향 화살표 (타원체의 장축 방향)
            arrow_length = rb_config.semi_major_axis * 0.8
            arrow_x = x + arrow_length * math.cos(yaw)
            arrow_y = y + arrow_length * math.sin(yaw)
            
            ax.arrow(x, y, 
                    arrow_x - x, arrow_y - y,
                    head_width=0.05, head_length=0.05, 
                    fc='black', ec='black', alpha=0.8)
            
            # 포즈 번호 표시 (첫 10개만)
            if i < 10:
                ax.text(x, y, str(i), fontsize=8, ha='center', va='center', 
                       color='white', weight='bold')
    
    def _draw_pose_pairs(self, ax, pairs: np.ndarray, max_pairs_to_show: int = 5) -> None:
        """Pose pair들을 화살표로 그리기"""
        
        # 표시할 pair 수 제한
        if len(pairs) > max_pairs_to_show:
            pairs = pairs[:max_pairs_to_show]
        
        for i, pair in enumerate(pairs):
            # pair는 (12,) 형태: [init_pose + target_pose]
            init_pose = pair[:6]
            target_pose = pair[6:]
            
            init_x, init_y = init_pose[0], init_pose[1]
            target_x, target_y = target_pose[0], target_pose[1]
            
            # 큰 화살표로 pair 연결 표시
            ax.annotate('', xy=(target_x, target_y), xytext=(init_x, init_y),
                       arrowprops=dict(arrowstyle='->', lw=2, color='blue', alpha=0.6))
            
            # pair 번호 표시
            mid_x = (init_x + target_x) / 2
            mid_y = (init_y + target_y) / 2
            ax.text(mid_x, mid_y, f"P{i}", fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='blue', alpha=0.3),
                   color='white', weight='bold')
    
    def _setup_hdf5_plot(self, ax, env_name: str, rb_config, num_poses: int, show_pairs: bool = False) -> None:
        """HDF5 기반 플롯 설정 및 꾸미기"""
        
        # 제목 및 레이블
        title = f'SE(3) Pose Visualization (HDF5)\n'
        title += f'Environment: {env_name} | '
        title += f'Rigid Body: {rb_config.name} | '
        title += f'Poses: {num_poses}'
        if show_pairs:
            title += ' | With Pose Pairs'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        
        # 격자 및 축 설정
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 범례 요소 생성
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=8, alpha=0.6, label='Obstacles'),
            patches.Rectangle((0, 0), 1, 1, facecolor=rb_config.color, 
                            edgecolor='black', alpha=0.7, 
                            label=f'{rb_config.name} ({rb_config.semi_major_axis:.2f}×{rb_config.semi_minor_axis:.2f}m)')
        ]
        
        if show_pairs:
            legend_elements.append(
                plt.Line2D([0], [0], color='blue', lw=2, alpha=0.6, label='Pose Pairs')
            )
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        # 여백 설정
        ax.margins(0.1)
    
    def visualize_multiple_environments(self, 
                                      env_names: List[str],
                                      rb_id: int,
                                      save_image: bool = False,
                                      output_file: str = "multi_env_comparison.png",
                                      show_plot: bool = True,
                                      max_poses_per_env: int = 10) -> bool:
        """여러 환경의 포즈 비교 시각화"""
        
        if len(env_names) > 4:
            print("Warning: Maximum 4 environments supported for comparison")
            env_names = env_names[:4]
        
        # 서브플롯 설정
        cols = min(2, len(env_names))
        rows = (len(env_names) + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
        
        if len(env_names) == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if len(env_names) > 1 else [axes]
        else:
            axes = axes.flatten()
        
        rb_config = self.collision_detector.get_rigid_body_config(rb_id)
        if rb_config is None:
            print(f"❌ Rigid body config not found for ID {rb_id}")
            return False
        
        success_count = 0
        
        for i, env_name in enumerate(env_names):
            ax = axes[i]
            
            try:
                # 데이터 로드
                poses, pose_metadata = self.pose_manager.get_poses(env_name, rb_id)
                if poses is None:
                    ax.text(0.5, 0.5, f"No data\n{env_name}", ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f"{env_name} (No Data)")
                    continue
                
                environment_points = self._load_environment_data_from_metadata(pose_metadata, env_name)
                
                # 표시할 포즈 수 제한
                if len(poses) > max_poses_per_env:
                    poses = poses[:max_poses_per_env]
                
                # 환경 포인트
                if len(environment_points) > 0:
                    ax.scatter(environment_points[:, 0], environment_points[:, 1], 
                             c='red', s=20, alpha=0.6)
                
                # SE(3) 포즈들
                self._draw_se3_poses(ax, poses, rb_config)
                
                # 서브플롯 설정
                ax.set_title(f"{env_name}\n{len(poses)} poses", fontsize=12, fontweight='bold')
                ax.set_xlabel('X (meters)')
                ax.set_ylabel('Y (meters)')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
                ax.margins(0.1)
                
                success_count += 1
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error\n{env_name}\n{str(e)[:50]}", ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10)
                ax.set_title(f"{env_name} (Error)")
        
        # 빈 서브플롯 숨기기
        for i in range(len(env_names), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_image:
            result_dir = Path("/home/dhkang225/2D_sim/data/visualized/pose")
            result_dir.mkdir(parents=True, exist_ok=True)
            
            if not os.path.isabs(output_file):
                output_file = result_dir / output_file
            
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"💾 Saved multi-environment comparison to: {output_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return success_count > 0


if __name__ == "__main__":
    # 간단한 테스트
    print("🧪 Testing HDF5PoseVisualizer...")
    
    visualizer = HDF5PoseVisualizer("/home/dhkang225/2D_sim/data/pose/unified_poses.h5")
    
    # 단일 환경 시각화 테스트
    success = visualizer.visualize_poses_from_hdf5(
        env_name="circle_env_000000",
        rb_id=3,
        save_image=True,
        show_plot=False,
        max_poses_to_show=10
    )
    
    if success:
        print("✅ Single environment visualization test passed")
    else:
        print("❌ Single environment visualization test failed")
    
    print("🎉 Test completed")
