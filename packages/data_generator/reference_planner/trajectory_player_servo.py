#!/usr/bin/env python3
"""
Trajectory Player - Servo Control Version
서보 제어 방식으로 부드러운 각도 제어를 구현한 OMPL 궤적 재생기

사용법:
    python trajectory_player_servo.py <trajectory_json_file> [options]
"""

import os
import sys
import json
import argparse
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# 루트 디렉토리를 sys.path에 추가하여 모듈 import 가능하게 함
root_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(root_dir)

# Box2D 시뮬레이션 관련 import
import pygame
import Box2D
from robot_simulation.core.env import make_world
from robot_simulation.core.render import draw_world
from robot_simulation.config_loader import get_config

# 비디오 기록용
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not available. Video recording disabled.")


class TrajectoryPlayerServo:
    """OMPL 궤적 재생기 - 서보 제어 방식"""
    
    def __init__(self):
        """초기화"""
        # config.yaml 파일의 절대 경로 설정
        config_path = os.path.join(root_dir, "config.yaml")
        self.config = get_config(config_path)
        
        # Pygame 초기화
        pygame.init()
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("OMPL Trajectory Player - Servo Control")
        
        # Box2D 시뮬레이션 컴포넌트
        self.world = None
        self.links = None
        self.obstacles = None
        self.joints = None
        
        # 서보 제어를 위한 변수들
        self.current_joint_angles = [0.0, 0.0, 0.0]
        self.target_joint_angles = [0.0, 0.0, 0.0]
        self.interpolation_factor = 0.1  # 부드러운 보간 계수
        
        # 궤적 시각화를 위한 변수들
        self.trajectory_points = []  # End-effector 경로 저장
        
        # 비디오 기록용
        self.video_writer = None
        self.frame_count = 0
        
    def load_trajectory_data(self, json_file: str) -> Dict[str, Any]:
        """궤적 JSON 파일 로드"""
        
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Trajectory file not found: {json_file}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        print(f"📂 OMPL Trajectory Data Loaded: {json_file}")
        print(f"   Success: {data['success']}")
        if 'planning_time_seconds' in data:
            print(f"   Planning Time: {data['planning_time_seconds']:.4f}s")
        if 'planning_time' in data:
            print(f"   Planning Time: {data['planning_time']:.4f}s")
        print(f"   Waypoints: {data['trajectory']['num_waypoints']}")
        print(f"   Total Duration: {data['trajectory']['total_duration']:.2f}s")
        print(f"   Robot: {data['robot']['link_shape']} links")
        
        # Environment info output (flexible handling)
        env_info = data.get('environment', {})
        if 'file_path' in env_info:
            print(f"   Environment: {env_info['file_path']}")
        elif 'num_points' in env_info:
            print(f"   Environment: pointcloud ({env_info['num_points']} points)")
        else:
            print(f"   Environment: No info available")
        
        return data
    
    def setup_simulation(self, trajectory_data: Dict[str, Any]) -> None:
        """Box2D 시뮬레이션 환경 설정"""
        
        # 로봇 정보 추출
        robot_config = trajectory_data['robot']
        robot_id = robot_config['robot_id']
        
        print(f"🤖 Setting up Box2D simulation... Robot ID {robot_id}")
        
        # Determine environment file - use trajectory environment (with fallback)
        env_config = trajectory_data.get('environment', {})
        
        # Use environment file path if available, otherwise use default environment
        if 'file_path' in env_config:
            env_file = Path(env_config['file_path']).stem  # Remove .ply extension
        else:
            # Default environment - use circles_only
            env_file = "circles_only"
            print("⚠️  No environment info found, using default environment (circles_only)")
        
        print(f"🌍 Environment: {env_file}")
        
        # Create Box2D world, robot, and obstacles
        self.world, self.links, self.obstacles = make_world(
            geometry_id=robot_id,
            env_file=env_file
        )
        
        # Store joint references
        self.joints = list(self.world.joints)
        
        print(f"✅ Box2D simulation setup complete")
        print(f"   Robot links: {len(self.links)}")
        print(f"   Joints: {len(self.joints)}")
        print(f"   Obstacles: {len(self.obstacles)}")
        print(f"🎛️  Using SERVO CONTROL mode (direct angle interpolation)")
    
    def set_robot_configuration_servo(self, target_angles: List[float], smooth: bool = True) -> None:
        """서보 제어 방식으로 로봇 각도 설정"""
        if len(self.joints) != len(target_angles):
            print(f"Warning: Joint count mismatch. Expected {len(target_angles)}, got {len(self.joints)}")
            return
        
        # 목표 각도 업데이트
        self.target_joint_angles = target_angles.copy()
        
        if smooth:
            # 부드러운 보간을 통한 각도 업데이트
            for i in range(len(self.current_joint_angles)):
                angle_diff = self.target_joint_angles[i] - self.current_joint_angles[i]
                self.current_joint_angles[i] += angle_diff * self.interpolation_factor
        else:
            # 즉시 목표 각도로 설정
            self.current_joint_angles = target_angles.copy()
        
        # Box2D 조인트에 각도 직접 적용 (서보 제어)
        self._apply_servo_angles()
    
    def _apply_servo_angles(self) -> None:
        """현재 각도를 Box2D 조인트에 직접 적용 (진짜 서보 제어 - Motor 사용)"""
        for i, (joint, target_angle) in enumerate(zip(self.joints, self.current_joint_angles)):
            # 현재 각도와 목표 각도의 차이 계산
            current_angle = joint.angle
            angle_diff = target_angle - current_angle
            
            # Box2D의 내장 Motor 기능 사용 (진짜 서보 제어)
            if abs(angle_diff) < 0.001:  # 목표에 거의 도달했으면 정지
                joint.motorSpeed = 0.0
                joint.enableMotor = True
                joint.maxMotorTorque = 1000.0  # 위치 유지를 위한 토크
            else:
                # 목표 각도로 이동하기 위한 속도 계산
                # 각도 차이에 비례한 속도 (빠른 수렴을 위해)
                desired_speed = angle_diff * 5.0  # 비례 계수
                
                # 속도 제한 (너무 빠르지 않게)
                max_speed = 10.0  # rad/s
                desired_speed = np.clip(desired_speed, -max_speed, max_speed)
                
                # Motor 설정
                joint.motorSpeed = desired_speed
                joint.enableMotor = True
                joint.maxMotorTorque = 500.0  # 충분한 토크
    
    def get_current_joint_angles(self) -> List[float]:
        """현재 조인트 각도 반환"""
        if not self.joints:
            return [0.0, 0.0, 0.0]
        
        return [joint.angle for joint in self.joints]
    
    def get_servo_target_angles(self) -> List[float]:
        """서보 목표 각도 반환"""
        return self.current_joint_angles.copy()
    
    def get_end_effector_position(self) -> Tuple[float, float]:
        """End-effector 위치 반환"""
        if not self.links:
            return (0.0, 0.0)
        
        end_effector_pos = self.links[-1].worldCenter
        return (end_effector_pos.x, end_effector_pos.y)
    
    def calculate_trajectory_path(self, joint_angles: List[List[float]]) -> List[Tuple[float, float]]:
        """궤적의 모든 waypoint에 대한 end-effector 위치 계산"""
        trajectory_points = []
        
        # 로봇 기하학 정보 (3-link robot)
        link_lengths = [3.0, 2.5, 2.0]  # 링크 길이
        
        for angles in joint_angles:
            if len(angles) >= 3:
                q1, q2, q3 = angles[0], angles[1], angles[2]
                
                # Forward kinematics 계산
                x = (link_lengths[0] * np.cos(q1) + 
                     link_lengths[1] * np.cos(q1 + q2) + 
                     link_lengths[2] * np.cos(q1 + q2 + q3))
                
                y = (link_lengths[0] * np.sin(q1) + 
                     link_lengths[1] * np.sin(q1 + q2) + 
                     link_lengths[2] * np.sin(q1 + q2 + q3))
                
                trajectory_points.append((x, y))
        
        return trajectory_points
    
    def draw_target_pose_landmarks(self, screen, target_joint_angles: List[float]):
        """Target pose의 모든 joint 위치를 빨간 점으로 표시 (landmark)"""
        from robot_simulation.core.render import PPM, ORIGIN
        
        # Forward kinematics로 target pose의 모든 joint 위치 계산
        link_lengths = [3.0, 2.5, 2.0]
        θ1, θ2, θ3 = target_joint_angles
        
        # 각 조인트의 절대 각도
        angle1 = θ1
        angle2 = θ1 + θ2
        angle3 = θ1 + θ2 + θ3
        
        # 각 joint 위치 계산
        positions = [
            (0.0, 0.0),  # Base (고정)
            (link_lengths[0] * np.cos(angle1), link_lengths[0] * np.sin(angle1)),  # Joint 1
            (link_lengths[0] * np.cos(angle1) + link_lengths[1] * np.cos(angle2),
             link_lengths[0] * np.sin(angle1) + link_lengths[1] * np.sin(angle2)),  # Joint 2
            (link_lengths[0] * np.cos(angle1) + link_lengths[1] * np.cos(angle2) + link_lengths[2] * np.cos(angle3),
             link_lengths[0] * np.sin(angle1) + link_lengths[1] * np.sin(angle2) + link_lengths[2] * np.sin(angle3))  # End-effector
        ]
        
        # 각 joint 위치를 빨간 점으로 표시
        for i, (x, y) in enumerate(positions):
            screen_x = int(ORIGIN[0] + x * PPM)
            screen_y = int(ORIGIN[1] - y * PPM)
            
            if i == 0:  # Base - 검은 점으로 표시
                pygame.draw.circle(screen, (50, 50, 50), (screen_x, screen_y), 6)
                pygame.draw.circle(screen, (255, 255, 255), (screen_x, screen_y), 3)
            elif i == 3:  # End-effector - 더 큰 빨간 점
                pygame.draw.circle(screen, (255, 255, 0), (screen_x, screen_y), 10, 2)  # 노란 테두리
                pygame.draw.circle(screen, (255, 0, 0), (screen_x, screen_y), 7)  # 빨간 중심
            else:  # 중간 joints - 중간 크기 빨간 점
                pygame.draw.circle(screen, (255, 255, 0), (screen_x, screen_y), 8, 2)  # 노란 테두리
                pygame.draw.circle(screen, (255, 0, 0), (screen_x, screen_y), 5)  # 빨간 중심
    
    def draw_trajectory_path(self, screen, trajectory_points: List[Tuple[float, float]], current_waypoint: int):
        """궤적 경로를 회색 점들로 표시 (참고용)"""
        from robot_simulation.core.render import PPM, ORIGIN
        
        # 전체 end-effector 경로를 작은 회색 점으로 표시 (참고용)
        for i, (x, y) in enumerate(trajectory_points):
            screen_x = int(ORIGIN[0] + x * PPM)
            screen_y = int(ORIGIN[1] - y * PPM)
            
            if i <= current_waypoint:
                # 지나간 경로는 어두운 회색
                color = (80, 80, 80)
                radius = 1
            else:
                # 앞으로 갈 경로는 밝은 회색
                color = (120, 120, 120)
                radius = 2
            
            pygame.draw.circle(screen, color, (screen_x, screen_y), radius)
    
    def play_trajectory(self, 
                       trajectory_data: Dict[str, Any],
                       record_video: bool = False,
                       output_path: str = None,
                       fps: int = 60,
                       speed_factor: float = 1.0,
                       show_controls: bool = True,
                       servo_smooth: bool = True,
                       interpolation_factor: float = 0.1) -> bool:
        """
        OMPL 궤적 재생 - 서보 제어 방식
        
        Args:
            trajectory_data: 궤적 데이터
            record_video: 비디오 기록 여부
            output_path: 출력 경로
            fps: 프레임 레이트
            speed_factor: 재생 속도 (1.0 = 실시간)
            show_controls: 제어 정보 표시 여부
            servo_smooth: 서보 부드러운 제어 여부
            interpolation_factor: 보간 계수 (0.01-1.0)
            
        Returns:
            성공 여부
        """
        
        # 서보 제어 파라미터 설정
        self.interpolation_factor = interpolation_factor
        
        # 비디오 기록 설정
        if record_video and HAS_CV2:
            if output_path is None:
                output_dir = Path("visualized")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # 파일명에 timestamp와 servo 표시 추가
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                smooth_str = "smooth" if servo_smooth else "direct"
                output_path = output_dir / f"trajectory_servo_{smooth_str}_{timestamp}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                str(output_path), fourcc, fps, 
                (self.screen_width, self.screen_height)
            )
            print(f"🎬 Starting video recording: {output_path}")
        
        # 궤적 정보 추출
        joint_angles = trajectory_data['trajectory']['joint_angles']
        timestamps = trajectory_data['trajectory']['timestamps']
        total_duration = trajectory_data['trajectory']['total_duration']
        
        # 시작/목표 구성 (안전하게 추출)
        start_goal = trajectory_data.get('start_goal', {})
        start_config = start_goal.get('start_config', joint_angles[0] if joint_angles else [0, 0, 0])
        goal_config = start_goal.get('goal_config', joint_angles[-1] if joint_angles else [0, 0, 0])
        
        # 궤적 경로 미리 계산
        trajectory_points = self.calculate_trajectory_path(joint_angles)
        
        print(f"🎯 Starting OMPL trajectory playback - SERVO CONTROL")
        print(f"   Waypoints: {len(joint_angles)}")
        print(f"   Total time: {total_duration:.2f}s")
        print(f"   Start: {[f'{x:.3f}' for x in start_config]}")
        print(f"   Goal: {[f'{x:.3f}' for x in goal_config]}")
        print(f"   Speed: {speed_factor}x")
        print(f"   Servo mode: {'Smooth' if servo_smooth else 'Direct'}")
        print(f"   Interpolation: {interpolation_factor}")
        print(f"   Controls: ESC=quit, SPACE=pause, R=restart")
        
        # 시뮬레이션 루프
        clock = pygame.time.Clock()
        current_waypoint = 0
        running = True
        paused = False
        
        trajectory_start_time = time.time()
        pause_start_time = 0.0
        total_pause_time = 0.0
        
        # 초기 위치 설정 (서보 제어)
        if joint_angles:
            self.current_joint_angles = joint_angles[0].copy()
            self.set_robot_configuration_servo(joint_angles[0], smooth=False)  # 초기는 즉시 설정
            
            # 몇 프레임 동안 초기 위치로 안정화
            for _ in range(30):
                self.set_robot_configuration_servo(joint_angles[0], smooth=servo_smooth)
                self.world.Step(1.0/fps, 10, 10)
        
        while running and current_waypoint < len(joint_angles):
            # 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # Pause/resume toggle
                        if paused:
                            paused = False
                            total_pause_time += time.time() - pause_start_time
                            print("▶️ Resumed")
                        else:
                            paused = True
                            pause_start_time = time.time()
                            print("⏸️ Paused")
                    elif event.key == pygame.K_r:
                        # Restart
                        current_waypoint = 0
                        trajectory_start_time = time.time()
                        total_pause_time = 0.0
                        print("🔄 Restarted")
            
            if paused:
                # Pause message display
                self.screen.fill((30, 30, 30))
                draw_world(self.screen, self.world, self.screen_width, self.screen_height)
                
                # 궤적 경로 표시 (일시정지 중에도)
                self.draw_trajectory_path(self.screen, trajectory_points, current_waypoint)
                
                font = pygame.font.Font(None, 48)
                pause_text = font.render("PAUSED - Press SPACE to resume", True, (255, 255, 0))
                text_rect = pause_text.get_rect(center=(self.screen_width//2, self.screen_height//2))
                self.screen.blit(pause_text, text_rect)
                
                pygame.display.flip()
                clock.tick(fps)
                continue
            
            # 현재 시간 계산 (일시정지 시간 제외)
            elapsed_time = (time.time() - trajectory_start_time - total_pause_time) * speed_factor
            
            # 현재 waypoint 결정
            while (current_waypoint < len(timestamps) - 1 and 
                   elapsed_time >= timestamps[current_waypoint + 1]):
                current_waypoint += 1
            
            # 로봇 구성 업데이트 (서보 제어)
            if current_waypoint < len(joint_angles):
                target_angles = joint_angles[current_waypoint]
                self.set_robot_configuration_servo(target_angles, smooth=servo_smooth)
            
            # 물리 시뮬레이션 스텝
            TIME_STEP = 1.0 / fps
            self.world.Step(TIME_STEP, 10, 10)
            
            # 렌더링
            self.screen.fill((30, 30, 30))  # 어두운 배경
            draw_world(self.screen, self.world, self.screen_width, self.screen_height)
            
            # 궤적 경로 표시 (빨간 점들)
            self.draw_trajectory_path(self.screen, trajectory_points, current_waypoint)
            
            # Trajectory information display
            if show_controls:
                current_time = timestamps[current_waypoint] if current_waypoint < len(timestamps) else 0
                actual_angles = self.get_current_joint_angles()
                servo_targets = self.get_servo_target_angles()
                target_config = joint_angles[current_waypoint] if current_waypoint < len(joint_angles) else [0, 0, 0]
                ee_pos = self.get_end_effector_position()
                
                info_text = [
                    f"OMPL RRT-Connect Trajectory - SERVO CONTROL",
                    f"Time: {current_time:.2f}s / {total_duration:.2f}s ({current_time/total_duration*100:.1f}%)",
                    f"Waypoint: {current_waypoint + 1} / {len(joint_angles)}",
                    f"Speed: {speed_factor}x | Mode: {'Smooth' if servo_smooth else 'Direct'}",
                    f"Target angles: [{target_config[0]:.3f}, {target_config[1]:.3f}, {target_config[2]:.3f}]",
                    f"Servo targets: [{servo_targets[0]:.3f}, {servo_targets[1]:.3f}, {servo_targets[2]:.3f}]",
                    f"Actual angles: [{actual_angles[0]:.3f}, {actual_angles[1]:.3f}, {actual_angles[2]:.3f}]",
                    f"End-effector: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f})",
                    f"Controls: ESC=quit | SPACE=pause | R=restart"
                ]
                
                font = pygame.font.Font(None, 26)
                for i, text in enumerate(info_text):
                    if i == 0:  # Title
                        color = (255, 255, 0)
                    elif i == len(info_text) - 1:  # Control info
                        color = (150, 150, 150)
                    elif i == 4:  # Target angles
                        color = (0, 255, 0)  # Green
                    elif i == 5:  # Servo targets
                        color = (255, 165, 0)  # Orange
                    elif i == 6:  # Actual angles
                        color = (255, 100, 100)  # Light red
                    else:
                        color = (255, 255, 255)
                    
                    text_surface = font.render(text, True, color)
                    self.screen.blit(text_surface, (10, 10 + i * 23))
            
            pygame.display.flip()
            
            # 비디오 프레임 기록
            if self.video_writer:
                frame = pygame.surfarray.array3d(self.screen)
                frame = np.transpose(frame, (1, 0, 2))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.video_writer.write(frame)
            
            self.frame_count += 1
            clock.tick(fps)
        
        # Completion message
        print(f"🎉 OMPL trajectory playback completed!")
        print(f"   Total frames: {self.frame_count}")
        
        # Cleanup
        if self.video_writer:
            self.video_writer.release()
            print(f"✅ Video saved: {output_path}")
        
        return True
    
    def cleanup(self):
        """정리"""
        if self.video_writer:
            self.video_writer.release()
        pygame.quit()


def parse_arguments():
    """명령행 인수 파싱"""
    
    parser = argparse.ArgumentParser(description='Play OMPL trajectory with servo control')
    
    parser.add_argument('trajectory_file', type=str, 
                       help='Path to OMPL trajectory JSON file')
    parser.add_argument('--record_video', action='store_true',
                       help='Record simulation video')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video file path')
    parser.add_argument('--fps', type=int, default=60,
                       help='Simulation FPS (default: 60)')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Playback speed factor (default: 1.0)')
    parser.add_argument('--no_controls', action='store_true',
                       help='Hide control information overlay')
    parser.add_argument('--servo_mode', choices=['smooth', 'direct'], default='smooth',
                       help='Servo control mode: smooth or direct (default: smooth)')
    parser.add_argument('--interpolation', type=float, default=0.1,
                       help='Interpolation factor for smooth mode (0.01-1.0, default: 0.1)')
    
    return parser.parse_args()


def main():
    """메인 함수"""
    
    args = parse_arguments()
    
    # 파일 존재 확인
    if not os.path.exists(args.trajectory_file):
        print(f"❌ Error: Trajectory file not found: {args.trajectory_file}")
        return 1
    
    try:
        # 궤적 재생기 생성
        player = TrajectoryPlayerServo()
        
        # 궤적 데이터 로드
        trajectory_data = player.load_trajectory_data(args.trajectory_file)
        
        # 성공적인 궤적인지 확인
        if not trajectory_data.get('success', False):
            print(f"❌ Error: Trajectory planning failed.")
            return 1
        
        # Box2D 시뮬레이션 설정
        player.setup_simulation(trajectory_data)
        
        print(f"🎮 Using real Box2D physics simulation with SERVO CONTROL")
        
        # 궤적 재생
        success = player.play_trajectory(
            trajectory_data=trajectory_data,
            record_video=args.record_video,
            output_path=args.output,
            fps=args.fps,
            speed_factor=args.speed,
            show_controls=not args.no_controls,
            servo_smooth=(args.servo_mode == 'smooth'),
            interpolation_factor=args.interpolation
        )
        
        if success:
            print(f"🎉 Trajectory playback successful!")
        else:
            print(f"❌ Trajectory playback failed!")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        try:
            player.cleanup()
        except:
            pass


if __name__ == "__main__":
    sys.exit(main()) 