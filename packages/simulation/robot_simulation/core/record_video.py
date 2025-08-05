# record_video.py - 시뮬레이션 비디오 녹화
import pygame, sys
import numpy as np
import argparse
import cv2
from robot_simulation.core.env import make_world, list_available_pointclouds
from robot_simulation.core.simulation import RobotSimulation
from robot_simulation.core.render import draw_world

# 명령행 인자 파싱
def parse_args():
    parser = argparse.ArgumentParser(description='Robot arm simulation with video recording')
    parser.add_argument('--target', nargs=2, type=float, default=[5.0, 5.0],
                        help='Target position (x y), default: 5.0 5.0')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Recording duration in seconds, default: 10.0')
    parser.add_argument('--output', type=str, default='robot_simulation.mp4',
                        help='Output video filename, default: robot_simulation.mp4')
    parser.add_argument('--fps', type=int, default=60,
                        help='Video FPS, default: 60')
    parser.add_argument('--env', type=str, default=None,
                        help='Pointcloud environment file (.ply). If not specified, uses static environment')
    parser.add_argument('--geometry', type=int, default=0,
                        help='Robot geometry ID (0-5), default: 0')
    parser.add_argument('--policy', choices=['potential_field', 'potential_field_pd', 'rmp'], 
                        default='potential_field_pd',
                        help='Control policy (default: potential_field_pd)')
    
    return parser.parse_args()

# target 위치 시각화 함수
def draw_target(screen, target, PPM=50.0, ORIGIN=(100, 500)):
    # 월드 좌표를 화면 좌표로 변환
    x, y = target
    sx = ORIGIN[0] + x * PPM
    sy = ORIGIN[1] - y * PPM
    
    # 빨간 원으로 target 그리기
    pygame.draw.circle(screen, (255, 0, 0), (int(sx), int(sy)), 8)
    # 중심점 표시
    pygame.draw.circle(screen, (255, 255, 255), (int(sx), int(sy)), 2)

def main():
    # 1) 명령행 인자 처리
    args = parse_args()
    target = np.array(args.target)
    duration = args.duration
    output_file = args.output
    fps = args.fps
    
    # Environment validation
    if args.env and args.env != 'static':
        import os
        
        # Convert environment name to full path: env_name -> data/env_name/env_name.ply
        env_name = args.env
        if env_name.endswith('.ply'):
            env_name = env_name[:-4]  # Remove .ply extension if provided
        
        # Construct the expected path
        pointcloud_path = os.path.join("pointcloud", "data", env_name, f"{env_name}.ply")
        
        if not os.path.exists(pointcloud_path):
            print(f"Error: Pointcloud file not found: {pointcloud_path}")
            print("Available environments:")
            
            # 데이터 디렉토리 설정  
            data_dir = os.path.join("data", "pointcloud")
            if os.path.exists(data_dir):
                loader = PointcloudLoader(data_dir)
                available_envs = loader.list_available_environments()
                
                if available_envs:
                    for env_name in sorted(available_envs):
                        print(f"  - {env_name}")
                else:
                    print("  (No environment folders found)")
            else:
                print("  (pointcloud/data directory not found)")
            
            print(f"\nUsage: --env <environment_name>")
            print(f"Example: --env circles_only")
            sys.exit(1)
        
        # Update args.env to use the environment name (not full path)
        args.env = env_name
    
    # Determine environment type
    env_type = 'static' if args.env == 'static' else 'pointcloud'
    
    # simulation_videos 폴더 생성
    import os
    results_dir = "data/results/simulation_videos"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 출력 파일 경로를 simulation_videos 폴더로 설정
    output_path = os.path.join(results_dir, output_file)
    
    print(f"Environment: {env_type}")
    if env_type == 'pointcloud':
        print(f"Pointcloud file: {args.env}")
    print(f"Target position: {target}")
    print(f"Recording duration: {duration} seconds")
    print(f"Output file: {output_path}")
    print(f"FPS: {fps}")

    # 2) 초기화
    SCREEN_W, SCREEN_H = 800, 600
    TIME_STEP = 1.0 / fps

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    
    # Set window caption based on environment type
    if env_type == 'pointcloud':
        caption = f"Robot Arm Simulation - Recording - Pointcloud: {args.env} - Target: {target}"
    else:
        caption = f"Robot Arm Simulation - Recording - Static Env - Target: {target}"
    pygame.display.set_caption(caption)
    
    clock = pygame.time.Clock()

    # 3) 월드·로봇·장애물 생성
    world, links, obstacles = make_world(
        geometry_id=args.geometry,
        env_file=args.env
    )

    # 시뮬레이션 객체 생성
    simulation = RobotSimulation(world, links, obstacles, target, args.policy)

    # 4) 비디오 작성기 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (SCREEN_W, SCREEN_H))

    # 5) 녹화 변수
    total_frames = int(duration * fps)
    current_frame = 0

    print(f"Starting recording... Total frames: {total_frames}")

    # 6) 메인 루프
    running = True
    while running and current_frame < total_frames:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        # 시뮬레이션 한 스텝 실행
        step_info = simulation.step(debug=False)  # 비디오 녹화시에는 디버그 출력 끄기

        # 렌더링
        screen.fill((30, 30, 30))
        draw_world(screen, world, SCREEN_W, SCREEN_H)
        
        # target 위치 시각화
        draw_target(screen, target)
        
        # 정보 텍스트 표시
        font = pygame.font.Font(None, 36)
        if env_type == 'pointcloud':
            info_text = f"Target: ({target[0]:.1f}, {target[1]:.1f}) | PC: {args.env} | Frame: {current_frame}/{total_frames}"
        else:
            info_text = f"Target: ({target[0]:.1f}, {target[1]:.1f}) | Static Env | Frame: {current_frame}/{total_frames}"
        text_surface = font.render(info_text, True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))
        
        # 진행률 표시
        progress = current_frame / total_frames
        progress_text = f"Recording: {progress*100:.1f}%"
        progress_surface = font.render(progress_text, True, (255, 255, 0))
        screen.blit(progress_surface, (10, 50))
        
        # 화면을 numpy 배열로 변환
        pygame_image = pygame.surfarray.array3d(screen)
        # pygame은 (width, height, 3), opencv는 (height, width, 3) 형식
        opencv_image = np.transpose(pygame_image, (1, 0, 2))
        # RGB를 BGR로 변환 (opencv 형식)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        
        # 비디오 프레임 작성
        video_writer.write(opencv_image)
        
        pygame.display.flip()
        clock.tick(fps)
        
        current_frame += 1
        
        # 진행률 출력 (10% 단위)
        if current_frame % (total_frames // 10) == 0:
            print(f"Recording progress: {current_frame}/{total_frames} frames ({progress*100:.0f}%)")

    # 7) 정리
    video_writer.release()
    pygame.quit()
    
    print(f"Recording completed! Video saved as: {output_path}")
    print(f"Total frames recorded: {current_frame}")

if __name__ == "__main__":
    main()
