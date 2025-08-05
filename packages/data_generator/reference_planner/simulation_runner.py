#!/usr/bin/env python3
"""
Simulation Runner
생성된 궤적을 Box2D 시뮬레이션에서 실행하고 영상으로 기록

사용법:
    python simulation_runner.py <trajectory_json_file>
    
예시:
    python simulation_runner.py reference_planner/data_export/test_trajectory.json --record_video
    python simulation_runner.py reference_planner/data_export/test_trajectory.json --save_frames
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
from robot_simulation.core.env import make_world, list_available_pointclouds
from robot_simulation.core.render import draw_world
from robot_simulation.config_loader import get_config

# 비디오 기록용
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not available. Video recording disabled.")


class TrajectorySimulationRunner:
    """궤적 시뮬레이션 실행기 - 실제 Box2D 통합"""
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        Args:
            config_file: 설정 파일 경로
        """
        self.config = get_config()
        
        # Pygame 초기화
        pygame.init()
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Robot Trajectory Simulation - OMPL RRT-Connect")
        
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
        
        print(f"📂 Loaded trajectory data from: {json_file}")
        print(f"   Success: {data['success']}")
        print(f"   Waypoints: {data['trajectory']['num_waypoints']}")
        print(f"   Duration: {data['trajectory']['total_duration']:.2f}s")
        print(f"   Robot: {data['robot']['link_shape']} links")
        
        return data
    
    def setup_simulation(self, trajectory_data: Dict[str, Any]) -> None:
        """Box2D 시뮬레이션 환경 설정"""
        
        # 로봇 정보 추출
        robot_config = trajectory_data['robot']
        robot_id = robot_config['robot_id']
        
        print(f"🤖 Setting up Box2D simulation for Robot ID {robot_id}")
        
        # 환경 파일 결정 (circles_only 환경 사용)
        env_file = "circles_only"  # 기본 환경
        
        # Box2D 월드, 로봇, 장애물 생성
        self.world, self.links, self.obstacles = make_world(
            geometry_id=robot_id,
            env_file=env_file
        )
        
        # 조인트 참조 저장
        self.joints = self.world.joints
        
        print(f"✅ Box2D simulation setup complete")
        print(f"   Robot links: {len(self.links)}")
        print(f"   Joints: {len(self.joints)}")
        print(f"   Obstacles: {len(self.obstacles)}")
    
    def set_robot_configuration(self, joint_angles: List[float]) -> None:
        """로봇을 특정 관절 각도로 설정"""
        if len(self.joints) != len(joint_angles):
            print(f"Warning: Joint count mismatch. Expected {len(joint_angles)}, got {len(self.joints)}")
            return
        
        # 각 조인트의 각도를 직접 설정
        for i, (joint, target_angle) in enumerate(zip(self.joints, joint_angles)):
            # 현재 각도
            current_angle = joint.angle
            
            # 각도 차이 계산
            angle_diff = target_angle - current_angle
            
            # 조인트에 토크 적용하여 목표 각도로 이동
            # 강한 토크를 적용하여 빠르게 수렴
            kp = 100.0  # Proportional gain
            torque = kp * angle_diff
            
            # 토크 제한
            max_torque = 50.0
            torque = np.clip(torque, -max_torque, max_torque)
            
            joint.bodyB.ApplyTorque(torque, wake=True)
    
    def get_current_joint_angles(self) -> List[float]:
        """현재 조인트 각도 반환 (절대 각도)"""
        if not self.joints:
            return [0.0, 0.0, 0.0]
        
        # 절대 각도 계산 (누적)
        angles = []
        cumulative_angle = 0.0
        
        for joint in self.joints:
            cumulative_angle += joint.angle
            angles.append(cumulative_angle)
        
        return angles
    
    def run_trajectory_simulation(self, 
                                trajectory_data: Dict[str, Any],
                                record_video: bool = False,
                                save_frames: bool = False,
                                output_path: str = None,
                                fps: int = 60,
                                speed_factor: float = 1.0) -> bool:
        """
        궤적 시뮬레이션 실행
        
        Args:
            trajectory_data: 궤적 데이터
            record_video: 비디오 기록 여부
            save_frames: 프레임 저장 여부
            output_path: 출력 경로
            fps: 프레임 레이트
            speed_factor: 재생 속도 (1.0 = 실시간)
            
        Returns:
            성공 여부
        """
        
        # 비디오 기록 설정
        if record_video and HAS_CV2:
            if output_path is None:
                output_dir = Path("data/results/simulation_videos")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "ompl_trajectory_simulation.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                str(output_path), fourcc, fps, 
                (self.screen_width, self.screen_height)
            )
            print(f"🎬 Recording video to: {output_path}")
        
        # 프레임 저장 설정
        frame_dir = None
        if save_frames:
            frame_dir = Path("data/results/trajectory_frames")
            frame_dir.mkdir(parents=True, exist_ok=True)
            print(f"📸 Saving frames to: {frame_dir}")
        
        # 궤적 정보
        joint_angles = trajectory_data['trajectory']['joint_angles']
        timestamps = trajectory_data['trajectory']['timestamps']
        total_duration = trajectory_data['trajectory']['total_duration']
        
        print(f"🎯 Starting OMPL trajectory simulation...")
        print(f"   Waypoints: {len(joint_angles)}")
        print(f"   Duration: {total_duration:.2f}s")
        print(f"   Speed factor: {speed_factor}x")
        
        # 시뮬레이션 루프
        clock = pygame.time.Clock()
        current_waypoint = 0
        running = True
        
        trajectory_start_time = time.time()
        
        while running and current_waypoint < len(joint_angles):
            # 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # 스페이스바로 일시정지/재생
                        input("Press Enter to continue...")
            
            # 현재 시간 계산
            elapsed_time = (time.time() - trajectory_start_time) * speed_factor
            
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
            
            # 궤적 정보 표시
            current_time = timestamps[current_waypoint] if current_waypoint < len(timestamps) else 0
            current_config = self.get_current_joint_angles()
            target_config = joint_angles[current_waypoint] if current_waypoint < len(joint_angles) else [0, 0, 0]
            
            info_text = [
                f"OMPL RRT-Connect Trajectory Playback",
                f"Time: {current_time:.2f}s / {total_duration:.2f}s",
                f"Waypoint: {current_waypoint + 1} / {len(joint_angles)}",
                f"Speed: {speed_factor}x",
                f"Target config: [{target_config[0]:.3f}, {target_config[1]:.3f}, {target_config[2]:.3f}]",
                f"Current config: [{current_config[0]:.3f}, {current_config[1]:.3f}, {current_config[2]:.3f}]",
                f"ESC: quit, SPACE: pause"
            ]
            
            font = pygame.font.Font(None, 28)
            for i, text in enumerate(info_text):
                color = (255, 255, 255) if i != 0 else (255, 255, 0)  # 제목은 노란색
                text_surface = font.render(text, True, color)
                self.screen.blit(text_surface, (10, 10 + i * 25))
            
            # End-effector 위치 표시
            if self.links:
                end_effector_pos = self.links[-1].worldCenter
                ee_x, ee_y = end_effector_pos
                
                # 월드 좌표를 화면 좌표로 변환
                from robot_simulation.core.render import PPM, ORIGIN
                screen_x = ORIGIN[0] + ee_x * PPM
                screen_y = ORIGIN[1] - ee_y * PPM
                
                # End-effector 위치에 빨간 점 표시
                pygame.draw.circle(self.screen, (255, 0, 0), (int(screen_x), int(screen_y)), 8)
                pygame.draw.circle(self.screen, (255, 255, 255), (int(screen_x), int(screen_y)), 3)
                
                # End-effector 위치 텍스트
                ee_text = f"End-effector: ({ee_x:.3f}, {ee_y:.3f})"
                ee_surface = font.render(ee_text, True, (255, 0, 0))
                self.screen.blit(ee_surface, (10, 10 + len(info_text) * 25))
            
            pygame.display.flip()
            
            # 비디오 프레임 기록
            if self.video_writer:
                frame = pygame.surfarray.array3d(self.screen)
                frame = np.transpose(frame, (1, 0, 2))  # pygame는 (width, height) 순서
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.video_writer.write(frame)
            
            # 프레임 저장
            if frame_dir:
                frame_filename = frame_dir / f"frame_{self.frame_count:06d}.png"
                pygame.image.save(self.screen, str(frame_filename))
            
            self.frame_count += 1
            
            # FPS 제한
            clock.tick(fps)
        
        # 정리
        if self.video_writer:
            self.video_writer.release()
            print(f"✅ Video saved: {output_path}")
        
        if save_frames:
            print(f"✅ {self.frame_count} frames saved to: {frame_dir}")
        
        print(f"🎉 OMPL trajectory simulation completed!")
        return True
    
    def cleanup(self):
        """정리"""
        if self.video_writer:
            self.video_writer.release()
        pygame.quit()


def parse_arguments():
    """명령행 인수 파싱"""
    
    parser = argparse.ArgumentParser(description='Run OMPL trajectory simulation with Box2D')
    
    parser.add_argument('trajectory_file', type=str, 
                       help='Path to trajectory JSON file')
    parser.add_argument('--record_video', action='store_true',
                       help='Record simulation video')
    parser.add_argument('--save_frames', action='store_true',
                       help='Save individual frames')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video file path')
    parser.add_argument('--fps', type=int, default=60,
                       help='Simulation FPS (default: 60)')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Playback speed factor (default: 1.0)')
    parser.add_argument('--config', type=str, default="config.yaml",
                       help='Configuration file path')
    
    return parser.parse_args()


def main():
    """메인 함수"""
    
    args = parse_arguments()
    
    # 파일 존재 확인
    if not os.path.exists(args.trajectory_file):
        print(f"Error: Trajectory file not found: {args.trajectory_file}")
        return 1
    
    try:
        # 시뮬레이션 러너 생성
        runner = TrajectorySimulationRunner(args.config)
        
        # 궤적 데이터 로드
        trajectory_data = runner.load_trajectory_data(args.trajectory_file)
        
        # Box2D 시뮬레이션 설정
        runner.setup_simulation(trajectory_data)
        
        print(f"🎮 Using Real Box2D Physics Simulation")
        
        # 시뮬레이션 실행
        success = runner.run_trajectory_simulation(
            trajectory_data=trajectory_data,
            record_video=args.record_video,
            save_frames=args.save_frames,
            output_path=args.output,
            fps=args.fps,
            speed_factor=args.speed
        )
        
        if success:
            print(f"🎉 Simulation completed successfully!")
        else:
            print(f"❌ Simulation failed!")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        try:
            runner.cleanup()
        except:
            pass


if __name__ == "__main__":
    sys.exit(main()) 