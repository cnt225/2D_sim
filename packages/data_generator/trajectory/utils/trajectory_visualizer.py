#!/usr/bin/env python3
"""
HDF5 Based Trajectory Visualizer
HDF5 파일에서 궤적 데이터를 읽어 환경과 함께 시각화

사용법:
    python trajectory_visualizer.py <env_name> <pair_id> [options]
    
예시:
    python trajectory_visualizer.py circle_env_000000 test_pair_000 --output-dir /path/to/output
    python trajectory_visualizer.py circle_env_000000 test_pair_000 --show-smoothed --save-animation
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_trajectory_from_hdf5(env_name: str, pair_id: str) -> Dict[str, Any]:
    """
    HDF5 파일에서 특정 궤적 데이터 로드
    
    Args:
        env_name: 환경 이름 (예: circle_env_000000)
        pair_id: 궤적 ID (예: test_pair_000)
        
    Returns:
        trajectory_data: 궤적 데이터 딕셔너리
    """
    from trajectory_data_manager import TrajectoryDataManager
    
    try:
        # TrajectoryDataManager 초기화
        manager = TrajectoryDataManager(env_name)
        
        # 궤적 데이터 로드
        pair_data = manager.get_pose_pair(pair_id)
        
        if pair_data is None:
            raise ValueError(f"Trajectory pair '{pair_id}' not found in environment '{env_name}'")
        
        print(f"✅ Loaded trajectory data: {env_name}/{pair_id}")
        print(f"   Raw trajectory: {len(pair_data['raw_trajectory'])} points")
        print(f"   Smooth trajectory: {len(pair_data['smooth_trajectory'])} points")
        print(f"   Method: {pair_data['metadata'].get('generation_method', 'N/A')}")
        print(f"   Planning time: {pair_data['metadata'].get('generation_time', 'N/A')}s")
        
        return pair_data
        
    except Exception as e:
        raise RuntimeError(f"Failed to load trajectory data: {e}")

def load_environment_pointcloud(env_name: str) -> np.ndarray:
    """
    환경 이름으로부터 포인트클라우드 파일 로드
    
    Args:
        env_name: 환경 이름
        
    Returns:
        points: 환경 포인트클라우드 (N x 2)
    """
    # 가능한 환경 파일 경로들
    project_root = Path(__file__).parent.parent.parent.parent
    possible_paths = [
        project_root / f"data/pointcloud/circles_only/{env_name}.ply",
        project_root / f"data/pointcloud/{env_name}/{env_name}.ply",
        project_root / f"data/pointcloud/{env_name}.ply"
    ]
    
    for ply_path in possible_paths:
        if ply_path.exists():
            print(f"📁 Loading environment: {ply_path}")
            return load_ply_file(str(ply_path))
    
    print(f"⚠️ Environment file not found for: {env_name}")
    return np.array([])

def load_ply_file(ply_file: str) -> np.ndarray:
    """PLY 파일에서 포인트 데이터 로드"""
    
    if not os.path.exists(ply_file):
        print(f"Warning: PLY file not found: {ply_file}")
        return np.array([])
    
    points = []
    try:
        with open(ply_file, 'r') as f:
            in_header = True
            for line in f:
                if in_header:
                    if line.strip() == 'end_header':
                        in_header = False
                    continue
                
                parts = line.strip().split()
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    points.append([x, y])
    
    except Exception as e:
        print(f"Error loading PLY file: {e}")
        return np.array([])
    
    return np.array(points)

class HDF5TrajectoryVisualizer:
    """HDF5 기반 궤적 시각화기"""
    
    def __init__(self, output_dir: str = None):
        """
        초기화
        
        Args:
            output_dir: 출력 디렉토리 (기본값: /home/dhkang225/2D_sim/data/visualized/trajectory)
        """
        if output_dir is None:
            self.output_dir = Path("/home/dhkang225/2D_sim/data/visualized/trajectory")
        else:
            self.output_dir = Path(output_dir)
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {self.output_dir}")
        
        self.fig = None
        self.ax = None
        
    def visualize_trajectory(self, 
                           env_name: str,
                           pair_id: str,
                           show_raw: bool = True,
                           show_smooth: bool = True,
                           save_image: bool = True,
                           show_plot: bool = False,
                           figsize: Tuple[int, int] = (12, 8)) -> str:
        """
        궤적 시각화 (정적 이미지)
        
        Args:
            env_name: 환경 이름
            pair_id: 궤적 ID  
            show_raw: 원본 궤적 표시 여부
            show_smooth: 스무딩된 궤적 표시 여부
            save_image: 이미지 저장 여부
            show_plot: 플롯 화면 표시 여부
            figsize: 그림 크기
            
        Returns:
            output_file: 저장된 파일 경로
        """
        # 데이터 로드
        trajectory_data = load_trajectory_from_hdf5(env_name, pair_id)
        environment_points = load_environment_pointcloud(env_name)
        
        # 그래프 설정
        self.fig, self.ax = plt.subplots(figsize=figsize)
        
        # 환경 그리기
        if len(environment_points) > 0:
            self.ax.scatter(environment_points[:, 0], environment_points[:, 1], 
                          c='red', s=1, alpha=0.6, label='Environment Obstacles')
        
        # 궤적 데이터 추출
        raw_trajectory = trajectory_data['raw_trajectory']
        smooth_trajectory = trajectory_data['smooth_trajectory']
        metadata = trajectory_data['metadata']
        
        # 시작점과 끝점
        start_pose = metadata['start_pose']
        end_pose = metadata['end_pose']
        
        # 원본 궤적 그리기
        if show_raw and len(raw_trajectory) > 0:
            self.ax.plot(raw_trajectory[:, 0], raw_trajectory[:, 1], 
                        'b-o', linewidth=2, markersize=3, alpha=0.7, 
                        label=f'Raw RRT ({len(raw_trajectory)} pts)')
        
        # 스무딩된 궤적 그리기  
        if show_smooth and len(smooth_trajectory) > 0 and len(smooth_trajectory) > 2:
            self.ax.plot(smooth_trajectory[:, 0], smooth_trajectory[:, 1], 
                        'g-', linewidth=3, alpha=0.8, 
                        label=f'B-spline Smooth ({len(smooth_trajectory)} pts)')
        
        # 시작점과 끝점 표시
        self.ax.plot(start_pose[0], start_pose[1], 
                    'go', markersize=10, markeredgecolor='black', linewidth=2, label='Start')
        self.ax.plot(end_pose[0], end_pose[1], 
                    'ro', markersize=10, markeredgecolor='black', linewidth=2, label='Goal')
        
        # 방향 화살표
        arrow_len = 0.3
        self.ax.arrow(start_pose[0], start_pose[1], 
                     arrow_len * np.cos(start_pose[2]), arrow_len * np.sin(start_pose[2]),
                     head_width=0.1, head_length=0.1, fc='green', ec='green')
        self.ax.arrow(end_pose[0], end_pose[1], 
                     arrow_len * np.cos(end_pose[2]), arrow_len * np.sin(end_pose[2]),
                     head_width=0.1, head_length=0.1, fc='red', ec='red')
        
        # 그래프 설정
        self.ax.grid(True, alpha=0.3)
        self.ax.axis('equal')
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        
        # 제목 설정
        method = metadata.get('generation_method', 'Unknown')
        planning_time = metadata.get('generation_time', 0.0)
        path_length = metadata.get('path_length', 0.0)
        
        self.ax.set_title(f'Trajectory: {env_name}/{pair_id}\n'
                         f'{method}, Time: {planning_time:.3f}s, Length: {path_length:.3f}')
        self.ax.legend()
        
        plt.tight_layout()
        
        # 저장
        output_file = None
        if save_image:
            output_file = self.output_dir / f"{env_name}_{pair_id}_trajectory.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"✅ Trajectory visualization saved: {output_file}")
        
        # 표시
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return str(output_file) if output_file else None
    
    def create_trajectory_animation(self,
                                  env_name: str, 
                                  pair_id: str,
                                  use_smooth: bool = False,
                                  fps: int = 10,
                                  save_animation: bool = True,
                                  show_animation: bool = False) -> str:
        """
        궤적 애니메이션 생성
        
        Args:
            env_name: 환경 이름
            pair_id: 궤적 ID
            use_smooth: 스무딩된 궤적 사용 여부
            fps: 프레임 레이트
            save_animation: 애니메이션 저장 여부
            show_animation: 애니메이션 화면 표시 여부
            
        Returns:
            output_file: 저장된 파일 경로
        """
        # 데이터 로드
        trajectory_data = load_trajectory_from_hdf5(env_name, pair_id)
        environment_points = load_environment_pointcloud(env_name)
        
        # 사용할 궤적 선택
        if use_smooth and len(trajectory_data['smooth_trajectory']) > 2:
            trajectory = trajectory_data['smooth_trajectory']
            traj_type = "smooth"
        else:
            trajectory = trajectory_data['raw_trajectory'] 
            traj_type = "raw"
        
        print(f"🎬 Creating animation with {len(trajectory)} points ({traj_type} trajectory)")
        
        # 그래프 설정
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # 환경 그리기
        if len(environment_points) > 0:
            self.ax.scatter(environment_points[:, 0], environment_points[:, 1], 
                          c='red', s=1, alpha=0.6, label='Environment')
        
        # 전체 경로 미리 그리기 (희미하게)
        self.ax.plot(trajectory[:, 0], trajectory[:, 1], 
                    'lightblue', linewidth=1, alpha=0.3, label='Full Path')
        
        # 축 범위 설정
        all_x = trajectory[:, 0]
        all_y = trajectory[:, 1]
        if len(environment_points) > 0:
            all_x = np.concatenate([all_x, environment_points[:, 0]])
            all_y = np.concatenate([all_y, environment_points[:, 1]])
        
        margin = 1.0
        self.ax.set_xlim(np.min(all_x) - margin, np.max(all_x) + margin)
        self.ax.set_ylim(np.min(all_y) - margin, np.max(all_y) + margin)
        
        # 그래프 설정
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_title(f'Trajectory Animation: {env_name}/{pair_id} ({traj_type})')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        # 애니메이션 요소들
        current_point, = self.ax.plot([], [], 'bo', markersize=8, label='Current Position')
        trail_line, = self.ax.plot([], [], 'darkblue', linewidth=3, alpha=0.8, label='Trail')
        
        # 방향 화살표 (SE(2) 궤적인 경우)
        arrow_line = None
        if trajectory.shape[1] >= 3:  # yaw 정보가 있는 경우
            arrow_line, = self.ax.plot([], [], 'b-', linewidth=3, alpha=0.9)
        
        # 시간 텍스트
        time_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                               verticalalignment='top', 
                               bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        def animate(frame):
            """애니메이션 함수"""
            if frame >= len(trajectory):
                return current_point, trail_line, time_text
            
            # 현재 위치
            x, y = trajectory[frame, 0], trajectory[frame, 1]
            current_point.set_data([x], [y])
            
            # 궤적 추적
            trail_x = trajectory[:frame+1, 0]
            trail_y = trajectory[:frame+1, 1]
            trail_line.set_data(trail_x, trail_y)
            
            # 방향 화살표 (SE(2)인 경우)
            if arrow_line is not None and trajectory.shape[1] >= 3:
                yaw = trajectory[frame, 2]
                arrow_len = 0.3
                dx = arrow_len * np.cos(yaw)
                dy = arrow_len * np.sin(yaw)
                arrow_line.set_data([x, x + dx], [y, y + dy])
            
            # 시간 정보
            time_text.set_text(f'Frame: {frame+1}/{len(trajectory)}\n'
                              f'Position: [{x:.2f}, {y:.2f}]')
            
            if arrow_line is not None:
                return current_point, trail_line, arrow_line, time_text
            else:
                return current_point, trail_line, time_text
        
        # 애니메이션 생성
        anim = animation.FuncAnimation(self.fig, animate, frames=len(trajectory), 
                                     interval=1000//fps, blit=False, repeat=True)
        
        # 저장
        output_file = None
        if save_animation:
            output_file = self.output_dir / f"{env_name}_{pair_id}_{traj_type}_animation.gif"
            print(f"💾 Saving animation: {output_file}")
            anim.save(str(output_file), writer='pillow', fps=fps)
            print(f"✅ Animation saved: {output_file}")
        
        # 표시
        if show_animation:
            plt.show()
        else:
            plt.close()
        
        return str(output_file) if output_file else None

def parse_arguments():
    """명령행 인수 파싱"""
    
    parser = argparse.ArgumentParser(description='Visualize trajectories from HDF5 files')
    
    parser.add_argument('env_name', type=str, 
                       help='Environment name (e.g., circle_env_000000)')
    parser.add_argument('pair_id', type=str,
                       help='Trajectory pair ID (e.g., test_pair_000)')
    
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for visualizations')
    parser.add_argument('--show-raw', action='store_true', default=True,
                       help='Show raw RRT trajectory (default: True)')
    parser.add_argument('--show-smooth', action='store_true', default=True,
                       help='Show smoothed trajectory (default: True)')
    parser.add_argument('--hide-raw', action='store_true',
                       help='Hide raw trajectory')
    parser.add_argument('--hide-smooth', action='store_true', 
                       help='Hide smoothed trajectory')
    
    parser.add_argument('--save-animation', action='store_true',
                       help='Create and save animation')
    parser.add_argument('--animation-smooth', action='store_true',
                       help='Use smoothed trajectory for animation')
    parser.add_argument('--fps', type=int, default=10,
                       help='Animation FPS (default: 10)')
    
    parser.add_argument('--show', action='store_true',
                       help='Show plots on screen')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save images')
    
    return parser.parse_args()

def main():
    """메인 함수"""
    
    args = parse_arguments()
    
    try:
        # 옵션 처리
        show_raw = args.show_raw and not args.hide_raw
        show_smooth = args.show_smooth and not args.hide_smooth
        save_image = not args.no_save
        
        print(f"🎨 Visualizing trajectory: {args.env_name}/{args.pair_id}")
        print(f"   Show raw: {show_raw}")
        print(f"   Show smooth: {show_smooth}")
        print(f"   Save animation: {args.save_animation}")
        
        # 시각화기 생성
        visualizer = HDF5TrajectoryVisualizer(args.output_dir)
        
        # 정적 시각화
        static_file = visualizer.visualize_trajectory(
            env_name=args.env_name,
            pair_id=args.pair_id,
            show_raw=show_raw,
            show_smooth=show_smooth,
            save_image=save_image,
            show_plot=args.show
        )
        
        # 애니메이션 생성 (요청된 경우)
        if args.save_animation:
            animation_file = visualizer.create_trajectory_animation(
                env_name=args.env_name,
                pair_id=args.pair_id,
                use_smooth=args.animation_smooth,
                fps=args.fps,
                save_animation=True,
                show_animation=args.show
            )
        
        print("✅ Visualization completed!")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)