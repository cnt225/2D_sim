#!/usr/bin/env python3
"""
SE(3) Pose Visualizer
저장된 SE(3) pose JSON 파일들을 환경과 함께 시각화

사용법:
    python pose_visualizer.py <pose_json_file>
    
예시:
    python pose_visualizer.py ../../data/pose/circles_only_rb_0_poses.json
    python pose_visualizer.py ../../data/pose/circle_envs/env_0001_rb_2_poses.json --save_image
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Dict, Any, Tuple
import math

try:
    from .collision_detector import RigidBodyCollisionDetector
except ImportError:
    from collision_detector import RigidBodyCollisionDetector


class SE3PoseVisualizer:
    """SE(3) 포즈 시각화기"""
    
    def __init__(self, config_file: str = "config/rigid_body_configs.yaml"):
        """
        Args:
            config_file: rigid body 설정 파일 경로
        """
        self.collision_detector = RigidBodyCollisionDetector(config_file)
        
    def visualize_poses_from_file(self, 
                                json_file: str, 
                                save_image: bool = False,
                                output_file: str = None,
                                show_plot: bool = True,
                                max_poses_to_show: int = 20) -> None:
        """
        JSON 파일에서 SE(3) pose들을 로드하여 시각화
        
        Args:
            json_file: pose JSON 파일 경로
            save_image: 이미지 저장 여부
            output_file: 출력 이미지 파일명
            show_plot: 플롯 표시 여부
            max_poses_to_show: 표시할 최대 포즈 개수
        """
        
        # JSON 파일 로드
        pose_data = self._load_pose_data(json_file)
        
        # 환경 데이터 로드
        environment_points = self._load_environment_data(pose_data)
        
        # 시각화
        self._create_visualization(pose_data, environment_points, save_image, output_file, show_plot, max_poses_to_show)
    
    def _load_pose_data(self, json_file: str) -> Dict[str, Any]:
        """JSON 파일에서 SE(3) pose 데이터 로드"""
        
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Pose file not found: {json_file}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        print(f"📂 Loaded SE(3) pose data from: {json_file}")
        print(f"   Environment: {data['environment']['name']}")
        print(f"   Rigid body: {data['rigid_body']['metadata']['name']}")
        print(f"   Poses: {data['poses']['count']}")
        print(f"   Format: {data['poses']['format']}")
        
        return data
    
    def _load_environment_data(self, pose_data: Dict[str, Any]) -> np.ndarray:
        """환경 PLY 파일에서 포인트 데이터 로드"""
        
        ply_file = pose_data['environment']['ply_file']
        
        # 상대 경로 처리
        if not os.path.exists(ply_file):
            # 다른 가능한 경로들 시도
            possible_paths = [
                ply_file,
                os.path.join("../../", ply_file),
                ply_file.replace("../../", "")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    ply_file = path
                    break
            else:
                raise FileNotFoundError(f"Environment file not found: {ply_file}")
        
        # PLY 파일 읽기
        points = []
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
        
        print(f"   Loaded {len(points)} environment points")
        return np.array(points)
    
    def _create_visualization(self, 
                            pose_data: Dict[str, Any], 
                            environment_points: np.ndarray,
                            save_image: bool = False,
                            output_file: str = None,
                            show_plot: bool = True,
                            max_poses_to_show: int = 20) -> None:
        """SE(3) 포즈들의 시각화 생성"""
        
        # 데이터 추출
        poses = pose_data['poses']['data']
        rigid_body_metadata = pose_data['rigid_body']['metadata']
        environment_name = pose_data['environment']['name']
        
        # 표시할 포즈 수 제한
        if len(poses) > max_poses_to_show:
            poses = poses[:max_poses_to_show]
            print(f"   Showing first {max_poses_to_show} poses out of {len(pose_data['poses']['data'])}")
        
        # 플롯 설정
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # 환경 포인트 그리기
        if len(environment_points) > 0:
            ax.scatter(environment_points[:, 0], environment_points[:, 1], 
                      c='red', s=20, alpha=0.6, label='Obstacles')
        
        # SE(3) 포즈들 그리기
        self._draw_se3_poses(ax, poses, rigid_body_metadata)
        
        # 플롯 설정
        self._setup_plot(ax, environment_name, rigid_body_metadata, len(poses))
        
        # 이미지 저장 또는 표시
        if save_image:
            # data/result 디렉토리 확인 및 생성
            result_dir = Path("../../data/result")
            result_dir.mkdir(parents=True, exist_ok=True)
            
            if output_file is None:
                output_file = f"{environment_name}_rb_{rigid_body_metadata['id']}_visualization.png"
            
            # 출력 파일 경로를 data/result 하위로 설정
            if not str(output_file).startswith('../../data/result/'):
                output_file = result_dir / output_file
            
            plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"💾 Saved visualization to: {output_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def _draw_se3_poses(self, ax, poses: List[List[float]], rigid_body_metadata: Dict[str, Any]) -> None:
        """SE(3) 포즈들을 타원체로 그리기"""
        
        semi_major = rigid_body_metadata['semi_major_axis']
        semi_minor = rigid_body_metadata['semi_minor_axis']
        color = rigid_body_metadata['color']
        
        for i, pose in enumerate(poses):
            x, y, z, roll, pitch, yaw = pose
            
            # 타원 생성 (matplotlib ellipse)
            ellipse = patches.Ellipse(
                (x, y), 
                width=2*semi_major,  # 전체 너비 (지름)
                height=2*semi_minor,  # 전체 높이 (지름) 
                angle=math.degrees(yaw),  # 각도를 도수로 변환
                facecolor=color,
                edgecolor='black',
                alpha=0.7,
                linewidth=1
            )
            
            ax.add_patch(ellipse)
            
            # 방향 화살표 (타원체의 장축 방향)
            arrow_length = semi_major * 0.8
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
    
    def _setup_plot(self, ax, environment_name: str, rigid_body_metadata: Dict[str, Any], num_poses: int) -> None:
        """플롯 설정 및 꾸미기"""
        
        # 제목 및 레이블
        ax.set_title(f'SE(3) Pose Visualization\n'
                    f'Environment: {environment_name} | '
                    f'Rigid Body: {rigid_body_metadata["name"]} | '
                    f'Poses: {num_poses}', 
                    fontsize=14, fontweight='bold')
        
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        
        # 격자 및 축 설정
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 범례
        # 샘플 타원체 생성 (범례용)
        sample_ellipse = patches.Ellipse((0, 0), 0, 0, 
                                       facecolor=rigid_body_metadata['color'], 
                                       edgecolor='black', alpha=0.7)
        ax.add_patch(sample_ellipse)
        
        # 범례 추가
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=8, alpha=0.6, label='Obstacles'),
            patches.Rectangle((0, 0), 1, 1, facecolor=rigid_body_metadata['color'], 
                            edgecolor='black', alpha=0.7, 
                            label=f'{rigid_body_metadata["name"]} ({rigid_body_metadata["semi_major_axis"]}×{rigid_body_metadata["semi_minor_axis"]}m)')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        # 여백 설정
        ax.margins(0.1)
    
    def create_pose_comparison(self, 
                             json_files: List[str],
                             save_image: bool = False,
                             output_file: str = "pose_comparison.png",
                             show_plot: bool = True) -> None:
        """여러 rigid body의 포즈 비교 시각화"""
        
        if len(json_files) > 3:
            print("Warning: Maximum 3 files supported for comparison")
            json_files = json_files[:3]
        
        fig, axes = plt.subplots(1, len(json_files), figsize=(6*len(json_files), 6))
        if len(json_files) == 1:
            axes = [axes]
        
        for i, json_file in enumerate(json_files):
            # 데이터 로드
            pose_data = self._load_pose_data(json_file)
            environment_points = self._load_environment_data(pose_data)
            
            # 서브플롯에 그리기
            ax = axes[i]
            
            # 환경 포인트
            if len(environment_points) > 0:
                ax.scatter(environment_points[:, 0], environment_points[:, 1], 
                          c='red', s=20, alpha=0.6)
            
            # SE(3) 포즈들 (최대 10개)
            poses = pose_data['poses']['data'][:10]
            rigid_body_metadata = pose_data['rigid_body']['metadata']
            
            self._draw_se3_poses(ax, poses, rigid_body_metadata)
            self._setup_plot(ax, pose_data['environment']['name'], 
                           rigid_body_metadata, len(poses))
        
        plt.tight_layout()
        
        if save_image:
            # data/result 디렉토리 확인 및 생성
            result_dir = Path("../../data/result")
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # 출력 파일 경로를 data/result 하위로 설정
            if not str(output_file).startswith('../../data/result/'):
                output_file = result_dir / output_file
            
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"💾 Saved comparison to: {output_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()


# 호환성을 위한 별칭
PoseVisualizer = SE3PoseVisualizer


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='Visualize SE(3) poses from JSON files')
    
    parser.add_argument('json_file', type=str,
                       help='SE(3) pose JSON file path')
    parser.add_argument('--save_image', action='store_true',
                       help='Save visualization as image')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output image file name')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not display the plot')
    parser.add_argument('--max_poses', type=int, default=20,
                       help='Maximum number of poses to display (default: 20)')
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_arguments()
    
    try:
        # 시각화기 초기화
        visualizer = SE3PoseVisualizer()
        
        # 시각화 생성
        visualizer.visualize_poses_from_file(
            json_file=args.json_file,
            save_image=args.save_image,
            output_file=args.output_file,
            show_plot=not args.no_show,
            max_poses_to_show=args.max_poses
        )
        
        print("🎉 Visualization completed!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 