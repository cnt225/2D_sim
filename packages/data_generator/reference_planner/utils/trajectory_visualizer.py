#!/usr/bin/env python3
"""
Trajectory Visualizer
생성된 궤적 JSON 파일들을 환경과 함께 시각화하고 애니메이션으로 출력

사용법:
    python trajectory_visualizer.py <trajectory_json_file>
    
예시:
    python trajectory_visualizer.py reference_planner/data_export/test_trajectory.json --save_animation
    python trajectory_visualizer.py reference_planner/data_export/test_trajectory.json --save_frames
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

def load_rigid_body_config(rigid_body_id: int) -> Dict[str, Any]:
    """
    rigid_body ID로부터 robot_geometries.yaml에서 설정 읽어오기
    
    Args:
        rigid_body_id: Rigid body ID (0-3)
        
    Returns:
        rigid_body_config: 설정 딕셔너리
    """
    try:
        # robot_geometries.yaml 파일 경로
        config_path = Path(__file__).parent.parent.parent.parent / "simulation/config/robot_geometries.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'rigid_bodies' in config and rigid_body_id in config['rigid_bodies']:
            return config['rigid_bodies'][rigid_body_id]
        else:
            print(f"⚠️ Rigid body ID {rigid_body_id} not found in config")
            return {}
            
    except Exception as e:
        print(f"⚠️ Failed to load robot geometry config: {e}")
        return {}

# 로봇 기하학 및 forward kinematics 계산
def compute_forward_kinematics(joint_angles: List[float], 
                              link_lengths: List[float]) -> List[Tuple[float, float]]:
    """
    Forward kinematics 계산하여 모든 링크의 끝점 위치 반환
    
    Args:
        joint_angles: [θ1, θ2, θ3] joint angles in radians
        link_lengths: [l1, l2, l3] link lengths
        
    Returns:
        positions: [(x0,y0), (x1,y1), (x2,y2), (x3,y3)] 
                  base, joint1, joint2, end-effector 위치
    """
    positions = [(0.0, 0.0)]  # Base position
    
    x, y = 0.0, 0.0
    angle_sum = 0.0
    
    for i, (angle, length) in enumerate(zip(joint_angles, link_lengths)):
        angle_sum += angle
        x += length * np.cos(angle_sum)
        y += length * np.sin(angle_sum)
        positions.append((x, y))
    
    return positions


def draw_robot_configuration(ax, joint_angles: List[float], 
                           link_lengths: List[float], 
                           link_widths: List[float],
                           link_shape: str = "ellipse",
                           color: str = 'blue',
                           alpha: float = 1.0,
                           linewidth: float = 2.0):
    """
    로봇 구성을 그리기
    
    Args:
        ax: matplotlib axes
        joint_angles: 관절 각도
        link_lengths: 링크 길이
        link_widths: 링크 너비  
        link_shape: 링크 형태 ("ellipse" or "rectangle")
        color: 색상
        alpha: 투명도
        linewidth: 선 두께
    """
    positions = compute_forward_kinematics(joint_angles, link_lengths)
    
    # 링크 그리기
    for i in range(len(positions) - 1):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        
        # 링크 중심선
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=alpha)
        
        # 링크 형태 (간단히 원으로 표시)
        if link_shape == "ellipse":
            circle = plt.Circle((x2, y2), link_widths[i]/2, 
                              color=color, alpha=alpha*0.3, fill=True)
            ax.add_patch(circle)
    
    # Base 표시
    base_circle = plt.Circle(positions[0], 0.1, color='black', alpha=1.0)
    ax.add_patch(base_circle)
    
    # End-effector 표시  
    end_circle = plt.Circle(positions[-1], 0.05, color='red', alpha=1.0)
    ax.add_patch(end_circle)


class TrajectoryVisualizer:
    """궤적 시각화기"""
    
    def __init__(self):
        """초기화"""
        self.fig = None
        self.ax = None
        
    def load_trajectory_data(self, json_file: str) -> Dict[str, Any]:
        """궤적 JSON 파일 로드 (SE(3) 및 Legacy 3-link 지원)"""
        
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Trajectory file not found: {json_file}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        print(f"📂 Loaded trajectory data from: {json_file}")
        
        # SE(3) trajectory 구조 감지
        if 'path' in data and 'data' in data['path']:
            print(f"   Type: SE(3) rigid body trajectory")
            print(f"   Trajectory ID: {data.get('trajectory_id', 'N/A')}")
            print(f"   Rigid Body: {data.get('rigid_body', {}).get('type', 'N/A')}")
            print(f"   Environment: {data.get('environment', {}).get('name', 'N/A')}")
            print(f"   Waypoints: {len(data['path']['data'])}")
            
            # SE(3) 구조를 Legacy 형식으로 변환하여 호환성 유지
            data['trajectory_type'] = 'SE3'
            data['success'] = True  # SE(3) trajectory는 성공한 것으로 간주
            
        # Legacy 3-link 구조 감지  
        elif 'trajectory' in data and 'joint_angles' in data['trajectory']:
            print(f"   Type: Legacy 3-link robot arm")
            print(f"   Success: {data.get('success', 'N/A')}")
            print(f"   Waypoints: {data['trajectory'].get('num_waypoints', len(data['trajectory']['joint_angles']))}")
            print(f"   Duration: {data['trajectory'].get('total_duration', 'N/A')}s")
            print(f"   Robot: {data.get('robot', {}).get('link_shape', 'N/A')} links")
            data['trajectory_type'] = 'Legacy'
            
        else:
            raise ValueError(f"Unknown trajectory format in {json_file}")
        
        return data
    
    def load_environment_data(self, ply_file: str) -> np.ndarray:
        """환경 PLY 파일에서 포인트 데이터 로드"""
        
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
    
    def visualize_static_trajectory(self, 
                                  trajectory_data: Dict[str, Any],
                                  environment_points: np.ndarray = None,
                                  save_image: bool = False,
                                  output_file: str = None,
                                  show_plot: bool = True) -> None:
        """정적 궤적 시각화 (SE(3) 및 Legacy 지원)"""
        
        # 그래프 설정
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        # 환경 그리기
        if environment_points is not None and len(environment_points) > 0:
            self.ax.scatter(environment_points[:, 0], environment_points[:, 1], 
                          c='red', s=1, alpha=0.6, label='Environment')
        
        # Trajectory 타입에 따른 처리
        trajectory_type = trajectory_data.get('trajectory_type', 'Legacy')
        
        if trajectory_type == 'SE3':
            self._visualize_se3_static(trajectory_data)
        else:
            self._visualize_legacy_static(trajectory_data)
        
        # 그래프 설정 완료
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        if save_image:
            if output_file is None:
                trajectory_id = trajectory_data.get('trajectory_id', 'trajectory')
                output_file = f"../results/visualized/{trajectory_id}_static.png"
            
            # results/visualized 폴더 생성 확인
            import os
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"💾 Static visualization saved: {output_file}")
        
        if show_plot:
            plt.show()
        
        plt.close()
    
    def _visualize_se3_static(self, trajectory_data: Dict[str, Any]) -> None:
        """SE(3) rigid body trajectory 정적 시각화"""
        
        # SE(3) trajectory 정보
        rigid_body = trajectory_data['rigid_body']
        poses = trajectory_data['path']['data']  # SE(3) poses
        start_pose = trajectory_data.get('start_pose', poses[0])
        goal_pose = trajectory_data.get('goal_pose', poses[-1])
        
        print(f"🎨 Drawing {len(poses)} SE(3) poses for {rigid_body['type']}")
        
        # Trajectory path 그리기
        x_coords = [pose[0] for pose in poses]
        y_coords = [pose[1] for pose in poses]
        
        self.ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7, label='Trajectory Path')
        self.ax.scatter(x_coords, y_coords, c='blue', s=20, alpha=0.6, label='Waypoints')
        
        # Start and Goal poses 강조 (실제 rigid body 크기 사용)
        self._draw_se3_pose(start_pose, color='green', label='Start', size='large', 
                           rigid_body_info=rigid_body)
        self._draw_se3_pose(goal_pose, color='red', label='Goal', size='large',
                           rigid_body_info=rigid_body)
        
        # 중간 poses 그리기 (몇 개만)
        step = max(1, len(poses) // 8)  # 최대 8개 중간 pose
        for i in range(step, len(poses)-step, step):
            self._draw_se3_pose(poses[i], color='orange', alpha=0.3, size='small',
                               rigid_body_info=rigid_body)
            
        # 타이틀 설정
        env_name = trajectory_data.get('environment', {}).get('name', 'unknown')
        self.ax.set_title(f'SE(3) Trajectory: {rigid_body["type"]} in {env_name}')
    
    def _draw_se3_pose(self, pose: List[float], color: str = 'blue', 
                       label: str = None, alpha: float = 0.8, size: str = 'medium',
                       rigid_body_info: Dict[str, Any] = None) -> None:
        """SE(3) pose를 ellipse + orientation arrow로 그리기"""
        
        x, y, z, roll, pitch, yaw = pose
        
        # Ellipse 크기 설정 - rigid_body ID로부터 실제 크기 읽어오기
        if rigid_body_info and 'id' in rigid_body_info:
            # rigid_body ID로부터 설정 읽어오기
            config = load_rigid_body_config(rigid_body_info['id'])
            if config:
                width = config.get('semi_major_axis', 1.0) * 2  # diameter
                height = config.get('semi_minor_axis', 0.5) * 2  # diameter
                arrow_len = max(width, height) * 0.6  # 크기에 비례한 화살표
                print(f"🎯 Using actual rigid body size: {width/2:.1f}×{height/2:.1f}m (ID: {rigid_body_info['id']})")
            else:
                # Fallback to old method
                width = rigid_body_info.get('semi_major_axis', 1.0) * 2
                height = rigid_body_info.get('semi_minor_axis', 0.5) * 2
                arrow_len = max(width, height) * 0.6
                print(f"🎯 Using fallback rigid body size: {width/2:.1f}×{height/2:.1f}m")
        else:
            # Fallback: 기존 하드코딩된 크기들
            if size == 'large':
                width, height = 0.3, 0.2
                arrow_len = 0.4
            elif size == 'small':
                width, height = 0.15, 0.1
                arrow_len = 0.2
            else:  # medium
                width, height = 0.2, 0.15
                arrow_len = 0.3
            print(f"⚠️ Using hardcoded size: {width}×{height}m")
        
        # Ellipse 그리기 (rigid body representation)
        from matplotlib.patches import Ellipse
        ellipse = Ellipse((x, y), width, height, angle=np.degrees(yaw), 
                         facecolor=color, alpha=alpha, edgecolor='black', linewidth=1)
        self.ax.add_patch(ellipse)
        
        # Orientation arrow 그리기
        dx = arrow_len * np.cos(yaw)
        dy = arrow_len * np.sin(yaw)
        self.ax.arrow(x, y, dx, dy, head_width=0.05, head_length=0.05, 
                     fc=color, ec=color, alpha=alpha*1.2)
        
        # Label 추가
        if label:
            self.ax.text(x, y-0.4, label, ha='center', va='top', fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
    
    def _visualize_legacy_static(self, trajectory_data: Dict[str, Any]) -> None:
        """Legacy 3-link robot arm trajectory 정적 시각화"""
        
        # 로봇 정보
        robot = trajectory_data['robot']
        link_lengths = robot['link_lengths']
        link_widths = robot['link_widths']
        link_shape = robot['link_shape']
        
        # 궤적 정보
        joint_angles = trajectory_data['trajectory']['joint_angles']
        timestamps = trajectory_data['trajectory']['timestamps']
        
        # 시작과 끝 구성 강조 표시
        start_config = joint_angles[0]
        end_config = joint_angles[-1]
        
        draw_robot_configuration(self.ax, start_config, link_lengths, link_widths,
                                link_shape, color='green', alpha=0.8, linewidth=3.0)
        draw_robot_configuration(self.ax, end_config, link_lengths, link_widths,
                                link_shape, color='red', alpha=0.8, linewidth=3.0)
        
        # 중간 구성들 (일부만 표시)
        step = max(1, len(joint_angles) // 10)  # 최대 10개 중간 프레임
        for i in range(0, len(joint_angles), step):
            if i == 0 or i == len(joint_angles) - 1:
                continue  # 시작/끝은 이미 그렸음
            alpha = 0.2
            draw_robot_configuration(self.ax, joint_angles[i], link_lengths, link_widths,
                                   link_shape, color='blue', alpha=alpha, linewidth=1.0)
        
        # End-effector 궤적 그리기
        ee_positions = []
        for config in joint_angles:
            positions = compute_forward_kinematics(config, link_lengths)
            ee_positions.append(positions[-1])  # End-effector position
        
        ee_x = [pos[0] for pos in ee_positions]
        ee_y = [pos[1] for pos in ee_positions]
        self.ax.plot(ee_x, ee_y, 'purple', linewidth=2, alpha=0.7, label='End-effector Path')
        
        # 축 설정
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title(f"Robot Trajectory Visualization\n"
                         f"Duration: {trajectory_data['trajectory']['total_duration']:.2f}s, "
                         f"Waypoints: {len(joint_angles)}")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        # 범위 설정
        all_x = ee_x.copy()
        all_y = ee_y.copy()
        if environment_points is not None and len(environment_points) > 0:
            all_x.extend(environment_points[:, 0].tolist())
            all_y.extend(environment_points[:, 1].tolist())
        
        if all_x and all_y:
            margin = 1.0
            self.ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            self.ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
        plt.tight_layout()
        
        # 이미지 저장
        if save_image:
            result_dir = Path("data/results/trajectories")
            result_dir.mkdir(parents=True, exist_ok=True)
            
            if output_file is None:
                output_file = "trajectory_visualization.png"
            
            if not str(output_file).startswith('data/results/trajectories/'):
                output_file = result_dir / output_file
            
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"💾 Saved visualization to: {output_file}")
        
        # 플롯 표시
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def create_animation(self, 
                        trajectory_data: Dict[str, Any],
                        environment_points: np.ndarray = None,
                        save_animation: bool = False,
                        output_file: str = None,
                        fps: int = 10) -> None:
        """궤적 애니메이션 생성 (SE(3) 및 Legacy 지원)"""
        
        # 그래프 설정
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        # 환경 그리기
        if environment_points is not None and len(environment_points) > 0:
            self.ax.scatter(environment_points[:, 0], environment_points[:, 1], 
                          c='red', s=1, alpha=0.6, label='Environment')
        
        # Trajectory 타입에 따른 처리
        trajectory_type = trajectory_data.get('trajectory_type', 'Legacy')
        
        if trajectory_type == 'SE3':
            self._create_se3_animation(trajectory_data, save_animation, output_file, fps)
        else:
            self._create_legacy_animation(trajectory_data, save_animation, output_file, fps)
    
    def _create_se3_animation(self, trajectory_data: Dict[str, Any], 
                             save_animation: bool, output_file: str, fps: int) -> None:
        """SE(3) rigid body trajectory 애니메이션 생성"""
        
        # SE(3) trajectory 정보
        rigid_body = trajectory_data['rigid_body']
        poses = trajectory_data['path']['data']
        env_name = trajectory_data.get('environment', {}).get('name', 'unknown')
        
        print(f"🎬 Creating animation with {len(poses)} SE(3) poses")
        
        # 전체 trajectory path 미리 그리기
        x_coords = [pose[0] for pose in poses]
        y_coords = [pose[1] for pose in poses]
        self.ax.plot(x_coords, y_coords, 'lightblue', linewidth=1, alpha=0.5, label='Full Path')
        
        # 축 범위 설정
        margin = 1.0
        self.ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
        self.ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
        
        # 제목 및 라벨 설정
        self.ax.set_title(f'SE(3) Animation: {rigid_body["type"]} in {env_name}')
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        # 애니메이션용 아티스트들
        from matplotlib.patches import Ellipse
        
        # Current pose ellipse (실제 rigid body 크기 사용)
        rb_width = rigid_body.get('semi_major_axis', 1.0) * 2  # diameter
        rb_height = rigid_body.get('semi_minor_axis', 0.5) * 2  # diameter
        current_ellipse = Ellipse((0, 0), rb_width, rb_height, angle=0, 
                                 facecolor='blue', alpha=0.8, edgecolor='black')
        self.ax.add_patch(current_ellipse)
        print(f"🎬 Animation using actual rigid body size: {rb_width/2:.1f}×{rb_height/2:.1f}m")
        
        # Current pose arrow (disabled)
        # current_arrow = self.ax.annotate('', xy=(0, 0), xytext=(0, 0),
        #                                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        current_arrow = None  # Disabled orientation arrow
        
        # Trail points
        trail_line, = self.ax.plot([], [], 'darkblue', linewidth=2, alpha=0.8, label='Current Trail')
        
        # Time text
        time_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                               verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white'))
        
        def animate(frame):
            """애니메이션 함수"""
            if frame >= len(poses):
                return current_ellipse, trail_line, time_text
            
            # Current pose
            x, y, z, roll, pitch, yaw = poses[frame]
            
            # Update ellipse
            current_ellipse.center = (x, y)
            current_ellipse.angle = np.degrees(yaw)
            
            # Update arrow (rigid body 크기에 비례) - disabled
            # arrow_len = max(rb_width, rb_height) * 0.6
            # dx = arrow_len * np.cos(yaw)
            # dy = arrow_len * np.sin(yaw)
            # current_arrow.xy = (x + dx, y + dy)
            # current_arrow.xytext = (x, y)
            
            # Update trail
            trail_x = [poses[i][0] for i in range(min(frame + 1, len(poses)))]
            trail_y = [poses[i][1] for i in range(min(frame + 1, len(poses)))]
            trail_line.set_data(trail_x, trail_y)
            
            # Update time text  
            time_text.set_text(f'Frame: {frame+1}/{len(poses)}\nPose: [{x:.2f}, {y:.2f}, {np.degrees(yaw):.1f}°]')
            
            return current_ellipse, trail_line, time_text
        
        # 애니메이션 생성
        import matplotlib.animation as animation
        anim = animation.FuncAnimation(self.fig, animate, frames=len(poses), 
                                     interval=1000//fps, blit=False, repeat=True)
        
        if save_animation:
            if output_file is None:
                trajectory_id = trajectory_data.get('trajectory_id', 'trajectory')
                output_file = f"../results/visualized/{trajectory_id}_animation.gif"
            
            # results/visualized 폴더 생성 확인
            import os
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            print(f"💾 Saving animation: {output_file}")
            anim.save(output_file, writer='pillow', fps=fps)
            print(f"✅ Animation saved: {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def _create_legacy_animation(self, trajectory_data: Dict[str, Any],
                                save_animation: bool, output_file: str, fps: int) -> None:
        """Legacy 3-link robot arm trajectory 애니메이션 생성"""
        
        # 로봇 정보
        robot = trajectory_data['robot']
        link_lengths = robot['link_lengths']
        link_widths = robot['link_widths']
        link_shape = robot['link_shape']
        
        # 궤적 정보
        joint_angles = trajectory_data['trajectory']['joint_angles']
        timestamps = trajectory_data['trajectory']['timestamps']
        
        # 축 설정
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Robot Trajectory Animation')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        # 범위 설정 (전체 궤적 기준)
        all_ee_positions = []
        for config in joint_angles:
            positions = compute_forward_kinematics(config, link_lengths)
            all_ee_positions.append(positions[-1])
        
        all_x = [pos[0] for pos in all_ee_positions]
        all_y = [pos[1] for pos in all_ee_positions]
        
        if environment_points is not None and len(environment_points) > 0:
            all_x.extend(environment_points[:, 0].tolist())
            all_y.extend(environment_points[:, 1].tolist())
        
        margin = 1.0
        self.ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        self.ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
        # 애니메이션 요소들
        robot_lines = []
        robot_circles = []
        trail_line, = self.ax.plot([], [], 'purple', linewidth=2, alpha=0.5, label='Trail')
        time_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, fontsize=12,
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 초기화 함수
        def init():
            return robot_lines + robot_circles + [trail_line, time_text]
        
        # 애니메이션 함수
        def animate(frame):
            # 이전 로봇 그래픽 지우기
            for line in robot_lines:
                line.remove()
            for circle in robot_circles:
                circle.remove()
            robot_lines.clear()
            robot_circles.clear()
            
            # 현재 로봇 구성 그리기
            config = joint_angles[frame]
            positions = compute_forward_kinematics(config, link_lengths)
            
            # 링크 그리기
            for i in range(len(positions) - 1):
                x1, y1 = positions[i]
                x2, y2 = positions[i + 1]
                line, = self.ax.plot([x1, x2], [y1, y2], 'blue', linewidth=3)
                robot_lines.append(line)
                
                # 조인트 원
                if link_shape == "ellipse":
                    circle = plt.Circle((x2, y2), link_widths[i]/2, 
                                      color='blue', alpha=0.3, fill=True)
                    self.ax.add_patch(circle)
                    robot_circles.append(circle)
            
            # Base와 end-effector
            base_circle = plt.Circle(positions[0], 0.1, color='black', alpha=1.0)
            end_circle = plt.Circle(positions[-1], 0.05, color='red', alpha=1.0)
            self.ax.add_patch(base_circle)
            self.ax.add_patch(end_circle)
            robot_circles.extend([base_circle, end_circle])
            
            # Trail 업데이트
            trail_x = [all_ee_positions[i][0] for i in range(frame + 1)]
            trail_y = [all_ee_positions[i][1] for i in range(frame + 1)]
            trail_line.set_data(trail_x, trail_y)
            
            # 시간 업데이트
            current_time = timestamps[frame] if frame < len(timestamps) else 0
            time_text.set_text(f'Time: {current_time:.2f}s\nFrame: {frame+1}/{len(joint_angles)}')
            
            return robot_lines + robot_circles + [trail_line, time_text]
        
        # 애니메이션 생성
        anim = animation.FuncAnimation(self.fig, animate, init_func=init,
                                     frames=len(joint_angles), interval=1000//fps,
                                     blit=False, repeat=True)
        
        # 애니메이션 저장
        if save_animation:
            result_dir = Path("data/results/trajectories")
            result_dir.mkdir(parents=True, exist_ok=True)
            
            if output_file is None:
                output_file = "trajectory_animation.mp4"
            
            if not str(output_file).startswith('data/results/trajectories/'):
                output_file = result_dir / output_file
            
            print(f"🎬 Saving animation to: {output_file}")
            print(f"   This may take a while...")
            
            # FFmpeg writer 설정
            writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='TrajectoryVisualizer'),
                                          bitrate=1800)
            anim.save(output_file, writer=writer)
            print(f"✅ Animation saved!")
        
        plt.show()


def parse_arguments():
    """명령행 인수 파싱"""
    
    parser = argparse.ArgumentParser(description='Visualize robot trajectories')
    
    parser.add_argument('trajectory_file', type=str, 
                       help='Path to trajectory JSON file')
    
    # 사용자 제안: --mode static|dynamic 통합 방식
    parser.add_argument('--mode', type=str, choices=['static', 'dynamic'], 
                       default='static',
                       help='Visualization mode: static (image) or dynamic (animation)')
    
    parser.add_argument('--output', type=str, default=None,
                       help='Output file name (auto-generated if not specified)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Animation FPS for dynamic mode (default: 10)')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not show plots')
    
    # Legacy options for backward compatibility
    parser.add_argument('--save_image', action='store_true',
                       help='DEPRECATED: Use --mode static instead')
    parser.add_argument('--save_animation', action='store_true',
                       help='DEPRECATED: Use --mode dynamic instead')
    parser.add_argument('--save_frames', action='store_true',
                       help='Save individual animation frames (dynamic mode only)')
    
    return parser.parse_args()


def main():
    """메인 함수"""
    
    args = parse_arguments()
    
    # 파일 존재 확인
    if not os.path.exists(args.trajectory_file):
        print(f"Error: Trajectory file not found: {args.trajectory_file}")
        return 1
    
    try:
        # Legacy 옵션을 새로운 mode로 변환
        if args.save_image and not args.save_animation:
            args.mode = 'static'
            print("⚠️ --save_image is deprecated. Using --mode static")
        elif args.save_animation:
            args.mode = 'dynamic'
            print("⚠️ --save_animation is deprecated. Using --mode dynamic")
        
        print(f"🎨 Visualization mode: {args.mode}")
        
        # 시각화기 생성
        visualizer = TrajectoryVisualizer()
        
        # 궤적 데이터 로드
        trajectory_data = visualizer.load_trajectory_data(args.trajectory_file)
        
        # 환경 데이터 로드 (trajectory JSON에서 환경 정보 사용)
        environment_points = np.array([])
        
        # Trajectory JSON에서 환경 정보 추출
        env_info = trajectory_data.get('environment', {})
        env_name = env_info.get('name')
        
        if env_name:
            # 환경 이름을 이용해 root/data/pointcloud에서 찾기
            # trajectory_visualizer.py가 utils/ 폴더에서 실행되므로 경로 조정
            env_ply_path = f"../../../../data/pointcloud/circle_envs_10k/{env_name}.ply"
            
            if os.path.exists(env_ply_path):
                environment_points = visualizer.load_environment_data(env_ply_path)
                print(f"📁 Using environment: {env_name} from {env_ply_path}")
            else:
                print(f"⚠️ Environment file not found: {env_ply_path}")
                # 백업: trajectory JSON의 원래 경로도 시도
                legacy_ply_file = env_info.get('ply_file')
                if legacy_ply_file and os.path.exists(legacy_ply_file):
                    environment_points = visualizer.load_environment_data(legacy_ply_file)
                    print(f"📁 Using legacy path: {legacy_ply_file}")
                else:
                    print(f"⚠️ Legacy path also failed: {legacy_ply_file}")
                    environment_points = np.array([])
        else:
            # 환경 파일 경로 추정 (fallback)
            possible_env_files = [
                "data/pointcloud/circles_only/circles_only.ply",
                "data/pointcloud/random_hard_01/random_hard_01.ply"
            ]
            
            for env_file in possible_env_files:
                if os.path.exists(env_file):
                    environment_points = visualizer.load_environment_data(env_file)
                    print(f"📁 Using fallback environment: {env_file}")
                    break
        
        # Mode에 따른 시각화 실행
        if args.mode == 'static':
            print("📊 Generating static trajectory visualization...")
            visualizer.visualize_static_trajectory(
                trajectory_data=trajectory_data,
                environment_points=environment_points,
                save_image=True,  # static mode는 항상 저장
                output_file=args.output,
                show_plot=not args.no_show
            )
        elif args.mode == 'dynamic':
            print("🎬 Generating dynamic trajectory animation...")
            visualizer.create_animation(
                trajectory_data=trajectory_data,
                environment_points=environment_points,
                save_animation=True,  # dynamic mode는 항상 저장
                output_file=args.output,
                fps=args.fps
            )
        
        print(f"✅ {args.mode.title()} visualization completed!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 