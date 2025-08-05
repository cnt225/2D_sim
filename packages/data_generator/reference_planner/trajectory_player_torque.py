#!/usr/bin/env python3
"""
OMPL Trajectory Player
실제 OMPL로 생성된 궤적을 Box2D 시뮬레이션에서 재생하고 영상으로 기록

사용법:
    python ompl_trajectory_player.py <trajectory_json_file> [options]
    
예시:
    python ompl_trajectory_player.py reference_planner/data_export/test_trajectory.json --record_video
    python ompl_trajectory_player.py reference_planner/data_export/test_trajectory.json --record_video --fps 30 --speed 0.5
"""

import os
import sys
import json
import argparse
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

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


class OMPLTrajectoryPlayer:
    """OMPL 궤적 재생기"""
    
    def __init__(self):
        """초기화"""
        self.config = get_config()
        
        # Pygame 초기화
        pygame.init()
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("OMPL RRT-Connect Trajectory Player")
        
        # Box2D 시뮬레이션 컴포넌트
        self.world = None
        self.links = None
        self.obstacles = None
        self.joints = None
        
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
    
    def set_robot_configuration(self, joint_angles: List[float]) -> None:
        """로봇을 특정 관절 각도로 설정 (부드러운 제어)"""
        if len(self.joints) != len(joint_angles):
            print(f"Warning: Joint count mismatch. Expected {len(joint_angles)}, got {len(self.joints)}")
            return
        
        # 각 조인트의 각도를 직접 설정 (부드러운 PD 제어)
        for i, (joint, target_angle) in enumerate(zip(self.joints, joint_angles)):
            # 현재 각도와의 차이
            current_angle = joint.angle
            angle_diff = target_angle - current_angle
            
            # 부드러운 PD 제어 (떨림 방지)
            kp = 300.0  # 적당한 비례 게인 (500→300)
            kd = 80.0   # 더 강한 댐핑 (50→80)
            
            # 각속도 얻기
            angular_velocity = joint.bodyB.angularVelocity
            
            # PD 제어
            torque = kp * angle_diff - kd * angular_velocity
            
            # 토크 제한 (더 부드럽게)
            max_torque = 150.0  # 토크 감소 (200→150)
            torque = np.clip(torque, -max_torque, max_torque)
            
            # 작은 각도 차이에서는 토크 스케일링
            if abs(angle_diff) < 0.01:  # 1도 미만
                torque *= 0.5  # 토크 절반으로 감소
            
            # 토크 적용
            joint.bodyB.ApplyTorque(torque, wake=True)
    
    def get_current_joint_angles(self) -> List[float]:
        """현재 조인트 각도 반환 (개별 각도)"""
        if not self.joints:
            return [0.0, 0.0, 0.0]
        
        return [joint.angle for joint in self.joints]
    
    def get_end_effector_position(self) -> Tuple[float, float]:
        """End-effector 위치 반환"""
        if not self.links:
            return (0.0, 0.0)
        
        end_effector_pos = self.links[-1].worldCenter
        return (end_effector_pos.x, end_effector_pos.y)
    
    def play_trajectory(self, 
                       trajectory_data: Dict[str, Any],
                       record_video: bool = False,
                       output_path: str = None,
                       fps: int = 60,
                       speed_factor: float = 1.0,
                       show_controls: bool = True) -> bool:
        """
        OMPL 궤적 재생
        
        Args:
            trajectory_data: 궤적 데이터
            record_video: 비디오 기록 여부
            output_path: 출력 경로
            fps: 프레임 레이트
            speed_factor: 재생 속도 (1.0 = 실시간)
            show_controls: 제어 정보 표시 여부
            
        Returns:
            성공 여부
        """
        
        # 비디오 기록 설정
        if record_video and HAS_CV2:
            if output_path is None:
                output_dir = Path("data/results/simulation_videos")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # 파일명에 timestamp 추가
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = output_dir / f"ompl_trajectory_{timestamp}.mp4"
            
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
        
        print(f"🎯 Starting OMPL trajectory playback")
        print(f"   Waypoints: {len(joint_angles)}")
        print(f"   Total time: {total_duration:.2f}s")
        print(f"   Start: {[f'{x:.3f}' for x in start_config]}")
        print(f"   Goal: {[f'{x:.3f}' for x in goal_config]}")
        print(f"   Speed: {speed_factor}x")
        print(f"   Controls: ESC=quit, SPACE=pause, R=restart")
        
        # 시뮬레이션 루프
        clock = pygame.time.Clock()
        current_waypoint = 0
        running = True
        paused = False
        
        trajectory_start_time = time.time()
        pause_start_time = 0.0
        total_pause_time = 0.0
        
        # 초기 위치 설정
        if joint_angles:
            self.set_robot_configuration(joint_angles[0])
            
            # 몇 프레임 동안 초기 위치로 안정화
            for _ in range(30):
                self.set_robot_configuration(joint_angles[0])
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
                # 일시정지 상태에서는 화면만 갱신
                self.screen.fill((30, 30, 30))
                draw_world(self.screen, self.world, self.screen_width, self.screen_height)
                
                # Pause message display
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
            
            # 로봇 구성 업데이트
            if current_waypoint < len(joint_angles):
                target_angles = joint_angles[current_waypoint]
                self.set_robot_configuration(target_angles)
            
            # 물리 시뮬레이션 스텝
            TIME_STEP = 1.0 / fps
            self.world.Step(TIME_STEP, 10, 10)
            
            # 렌더링
            self.screen.fill((30, 30, 30))  # 어두운 배경
            draw_world(self.screen, self.world, self.screen_width, self.screen_height)
            
            # Trajectory information display
            if show_controls:
                current_time = timestamps[current_waypoint] if current_waypoint < len(timestamps) else 0
                current_config = self.get_current_joint_angles()
                target_config = joint_angles[current_waypoint] if current_waypoint < len(joint_angles) else [0, 0, 0]
                ee_pos = self.get_end_effector_position()
                
                info_text = [
                    f"OMPL RRT-Connect Trajectory Playback",
                    f"Time: {current_time:.2f}s / {total_duration:.2f}s ({current_time/total_duration*100:.1f}%)",
                    f"Waypoint: {current_waypoint + 1} / {len(joint_angles)}",
                    f"Speed: {speed_factor}x",
                    f"Target angles: [{target_config[0]:.3f}, {target_config[1]:.3f}, {target_config[2]:.3f}]",
                    f"Current angles: [{current_config[0]:.3f}, {current_config[1]:.3f}, {current_config[2]:.3f}]",
                    f"End-effector: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f})",
                    f"Controls: ESC=quit | SPACE=pause | R=restart"
                ]
                
                font = pygame.font.Font(None, 28)
                for i, text in enumerate(info_text):
                    if i == 0:  # Title
                        color = (255, 255, 0)
                    elif i == len(info_text) - 1:  # Control info
                        color = (150, 150, 150)
                    else:
                        color = (255, 255, 255)
                    
                    text_surface = font.render(text, True, color)
                    self.screen.blit(text_surface, (10, 10 + i * 25))
            
            # End-effector 위치 강조 표시
            ee_pos = self.get_end_effector_position()
            from robot_simulation.core.render import PPM, ORIGIN
            screen_x = ORIGIN[0] + ee_pos[0] * PPM
            screen_y = ORIGIN[1] - ee_pos[1] * PPM
            
            # End-effector 위치에 빨간 원 표시
            pygame.draw.circle(self.screen, (255, 0, 0), (int(screen_x), int(screen_y)), 10, 3)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(screen_x), int(screen_y)), 4)
            
            # 목표 위치 표시 (있는 경우)
            if 'goal_position' in trajectory_data:
                goal_pos = trajectory_data['goal_position']
                goal_screen_x = ORIGIN[0] + goal_pos[0] * PPM
                goal_screen_y = ORIGIN[1] - goal_pos[1] * PPM
                pygame.draw.circle(self.screen, (0, 255, 0), (int(goal_screen_x), int(goal_screen_y)), 8, 2)
            
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
    
    parser = argparse.ArgumentParser(description='Play OMPL trajectory in Box2D simulation')
    
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
        player = OMPLTrajectoryPlayer()
        
        # 궤적 데이터 로드
        trajectory_data = player.load_trajectory_data(args.trajectory_file)
        
        # 성공적인 궤적인지 확인
        if not trajectory_data.get('success', False):
            print(f"❌ Error: Trajectory planning failed.")
            return 1
        
        # Box2D 시뮬레이션 설정
        player.setup_simulation(trajectory_data)
        
        print(f"🎮 Using real Box2D physics simulation")
        
        # 궤적 재생
        success = player.play_trajectory(
            trajectory_data=trajectory_data,
            record_video=args.record_video,
            output_path=args.output,
            fps=args.fps,
            speed_factor=args.speed,
            show_controls=not args.no_controls
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