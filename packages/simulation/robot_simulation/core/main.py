# main.py - SE(3) Rigid Body Simulation
import pygame, sys
import argparse
import numpy as np
from robot_simulation.core.env import make_world, list_available_pointclouds
from robot_simulation.core.simulation import RobotSimulation
from robot_simulation.config_loader import get_rigid_body_config, is_valid_rigid_body_id
from robot_simulation.core.render import draw_world
try:
    from data_generator.pose.pose_pair_loader import SE3PosePairLoader
    SE3_POSE_SUPPORT = True
except ImportError:
    print("Warning: SE(3) pose pair loader not available. SE(3) functionality limited.")
    SE3_POSE_SUPPORT = False

# 명령행 인자 파싱
def parse_args():
    parser = argparse.ArgumentParser(description='SE(3) Rigid Body simulation with target pose control')
    
    # Pose 관련 옵션들
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--target-pose', nargs=6, type=float, default=None,
                       help='Target SE(3) pose (x y z roll pitch yaw). Example: 5.0 3.0 0.0 0.0 0.0 1.57')
    group.add_argument('--pose-pair', nargs=2, type=str, default=None, metavar=('ENV_NAME', 'INDEX'),
                       help='Use SE(3) pose pair from data. ENV_NAME: environment name, INDEX: pair index (or "random")')
    
    parser.add_argument('--init-pose', nargs=6, type=float, default=None,
                        help='Initial SE(3) pose (x y z roll pitch yaw), only used with --target-pose')
    parser.add_argument('--env', type=str, default=None,
                        help='Pointcloud environment file (.ply). If not specified, uses static environment')
    parser.add_argument('--geometry', type=int, default=0,
                        help='Rigid body geometry ID (0-3), default: 0. Use --list-geometries to see available options')
    parser.add_argument('--policy', choices=['position_control', 'velocity_control'], 
                        default='position_control',
                        help='Control policy (default: position_control)')
    parser.add_argument('--list-geometries', action='store_true',
                        help='List available rigid body geometries and exit')
    parser.add_argument('--list-pose-pairs', action='store_true',
                        help='List available SE(3) pose pair environments and exit')
    
    return parser.parse_args()


def draw_se3_target_pose(screen, simulation, PPM=50.0, ORIGIN=(100, 500)):
    """SE(3) Target pose의 robot geometry를 빨간 경계로 시각화"""
    import pygame
    
    # Target pose vertices 가져오기 (simulation에서 계산)
    target_vertices = simulation.get_target_vertices()
    
    if not target_vertices:
        return
    
    # 월드 좌표를 화면 좌표로 변환
    screen_vertices = []
    for wx, wy in target_vertices:
        sx = ORIGIN[0] + wx * PPM
        sy = ORIGIN[1] - wy * PPM
        screen_vertices.append((int(sx), int(sy)))
    
    # Target robot geometry outline 그리기 (빨간 경계)
    if len(screen_vertices) >= 3:
        # 반투명 빨간 면 그리기
        target_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        pygame.draw.polygon(target_surface, (255, 0, 0, 50), screen_vertices)  # 반투명 빨간색
        screen.blit(target_surface, (0, 0))
        
        # 빨간 경계선 그리기
        pygame.draw.polygon(screen, (255, 0, 0), screen_vertices, 3)  # 굵은 빨간 경계
        
        # Target pose 중심점 표시
        target_pose = simulation.target_pose
        target_x, target_y = target_pose[:2]
        center_sx = ORIGIN[0] + target_x * PPM
        center_sy = ORIGIN[1] - target_y * PPM
        pygame.draw.circle(screen, (255, 0, 0), (int(center_sx), int(center_sy)), 5)
        
        # Target orientation 화살표 표시 (yaw 방향)
        target_yaw = target_pose[5]
        arrow_length = 30
        arrow_end_x = center_sx + arrow_length * np.cos(target_yaw)
        arrow_end_y = center_sy - arrow_length * np.sin(target_yaw)  # y축 반전
        pygame.draw.line(screen, (255, 100, 100), 
                        (int(center_sx), int(center_sy)), 
                        (int(arrow_end_x), int(arrow_end_y)), 4)
        
        # 화살표 머리 그리기
        arrow_head_length = 8
        arrow_head_angle = 0.5  # radians
        
        # 왼쪽 화살표 머리
        head1_x = arrow_end_x - arrow_head_length * np.cos(target_yaw - arrow_head_angle)
        head1_y = arrow_end_y + arrow_head_length * np.sin(target_yaw - arrow_head_angle)
        pygame.draw.line(screen, (255, 100, 100), 
                        (int(arrow_end_x), int(arrow_end_y)), 
                        (int(head1_x), int(head1_y)), 3)
        
        # 오른쪽 화살표 머리
        head2_x = arrow_end_x - arrow_head_length * np.cos(target_yaw + arrow_head_angle)
        head2_y = arrow_end_y + arrow_head_length * np.sin(target_yaw + arrow_head_angle)
        pygame.draw.line(screen, (255, 100, 100), 
                        (int(arrow_end_x), int(arrow_end_y)), 
                        (int(head2_x), int(head2_y)), 3)


# 1) 명령행 인자 처리
args = parse_args()

# Handle --list-pose-pairs flag
if args.list_pose_pairs:
    print("Available SE(3) Pose Pair Data:")
    
    if SE3_POSE_SUPPORT:
        se3_loader = SE3PosePairLoader()
        available_se3 = se3_loader.list_available_pairs()
        if available_se3:
            for env_name in available_se3:
                se3_loader.print_summary(env_name)
                print()
        else:
            print("No SE(3) pose pairs found")
    else:
        print("SE(3) pose pair loader not available")
    
    sys.exit(0)

# Handle --list-geometries flag
if args.list_geometries:
    from robot_simulation.config_loader import get_config, list_rigid_bodies
    config = get_config()
    
    print("Available SE(3) Rigid Body Geometries:")
    rigid_bodies = list_rigid_bodies()
    for rb_id, rb_config in rigid_bodies.items():
        name = rb_config['name']
        desc = rb_config['description']
        semi_major = rb_config['semi_major_axis']
        semi_minor = rb_config['semi_minor_axis']
        aspect_ratio = semi_major / semi_minor if semi_minor > 0 else 'N/A'
        print(f"  {rb_id}: {name} - {desc}")
        print(f"      Size: {semi_major}×{semi_minor}m | Aspect Ratio: {aspect_ratio:.1f}")
        print()
    
    sys.exit(0)

# Determine poses based on arguments
init_pose = None
target_pose = None
env_from_pose = None

if args.pose_pair:
    # Use SE(3) pose pair data
    env_name, pair_index_str = args.pose_pair
    
    if SE3_POSE_SUPPORT:
        print(f"Loading SE(3) pose pairs for rigid body...")
        loader = SE3PosePairLoader()
        available_envs = loader.list_available_pairs()
        
        if env_name not in available_envs:
            print(f"Error: SE(3) environment '{env_name}' not found in pose pair data.")
            print(f"Available SE(3) environments: {available_envs}")
            sys.exit(1)
        
        # Get SE(3) pose pair
        if pair_index_str.lower() == "random":
            all_pairs = loader.get_all_pairs(env_name)
            if not all_pairs:
                print(f"Error: No pose pairs available for {env_name}")
                sys.exit(1)
            import random
            pair = random.choice(all_pairs)
            init_pose, target_pose = pair['init'], pair['target']
            print(f"Using random SE(3) pose pair from {env_name}")
        else:
            try:
                pair_index = int(pair_index_str)
                init_pose, target_pose = loader.get_pose_pair(env_name, pair_index)
                print(f"Using SE(3) pose pair {pair_index} from {env_name}")
            except ValueError:
                print(f"Error: Invalid pair index '{pair_index_str}'. Use integer or 'random'.")
                sys.exit(1)
            except IndexError:
                pair_count = loader.get_pair_count(env_name)
                print(f"Error: Pair index {pair_index} out of range [0, {pair_count-1}]")
                sys.exit(1)
        
        # Determine environment file from SE(3) metadata
        metadata = loader.get_metadata(env_name)
        env_from_pose = metadata.get('environment', {}).get('ply_file')
        
    else:
        print("Error: SE(3) pose pair loader not available")
        sys.exit(1)
    
    # Set robot geometry for pose pairs
    robot_geometry = args.geometry

elif args.target_pose:
    # Use manually specified SE(3) poses
    target_pose = np.array(args.target_pose)
    init_pose = np.array(args.init_pose) if args.init_pose else None
    robot_geometry = args.geometry
    
    print(f"Using manual SE(3) target pose: {target_pose}")
    
    if init_pose is not None:
        print(f"Using manual init pose: {init_pose}")
    
else:
    # Default SE(3) poses for rigid body
    robot_geometry = args.geometry
    target_pose = np.array([5.0, 3.0, 0.0, 0.0, 0.0, 1.57])  # x, y, z, roll, pitch, yaw
    print(f"Using default SE(3) target pose: {target_pose}")

# Validate rigid body ID
if not is_valid_rigid_body_id(robot_geometry):
    print(f"Error: Invalid rigid body ID {robot_geometry}. Valid range: 0-3")
    sys.exit(1)

# Convert poses to numpy arrays
target_pose = np.array(target_pose)
if init_pose is not None:
    init_pose = np.array(init_pose)

# Determine environment to use
env_file = None
if args.env:
    env_file = args.env
elif env_from_pose:
    env_file = env_from_pose

# Validate environment file if specified
if env_file and env_file != 'static':
    import os
    
    # If it's not a full path, try to construct one
    if not env_file.startswith('data/'):
        env_name = env_file
        if env_name.endswith('.ply'):
            env_name = env_name[:-4]  # Remove .ply extension if provided
        
        # Try different possible paths
        possible_paths = [
            os.path.join("data", "pointcloud", env_name, f"{env_name}.ply"),
            os.path.join("data", "pointcloud", "circles_only", f"{env_name}.ply"),
            os.path.join("data", "pointcloud", f"{env_name}.ply"),
            env_name,  # As is
            f"{env_name}.ply"  # Add .ply extension
        ]
        
        env_file = None
        for path in possible_paths:
            if os.path.exists(path):
                env_file = path
                break
        
        if env_file is None:
            print(f"Error: Environment file not found for '{args.env or env_from_pose}'")
            print(f"Tried paths: {possible_paths}")
            sys.exit(1)

# Determine environment type
env_type = 'static' if env_file in [None, 'static'] else 'pointcloud'

print(f"Target pose: {target_pose}")
if init_pose is not None:
    print(f"Init pose: {init_pose}")
if env_type == 'pointcloud':
    print(f"Environment file: {env_file}")
print(f"Robot geometry: {robot_geometry}")

# 2) 초기화
SCREEN_W, SCREEN_H = 800, 600
FPS = 60
TIME_STEP = 1.0 / FPS

pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption(f"SE(3) Rigid Body Simulation - Target: {target_pose[:3]}")
clock  = pygame.time.Clock()

# 3) 월드·로봇·장애물 생성
world, robot_body, obstacles, robot_config = make_world(
    geometry_id=robot_geometry,
    env_file=env_file,
    initial_pose=init_pose
)

# 4) 시뮬레이션 객체 생성
simulation = RobotSimulation(world, robot_body, obstacles, robot_config, target_pose, init_pose, args.policy)
print(f"Created SE(3) simulation with {robot_config['name']}")

# 5) 메인 루프
while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit(); sys.exit()

    # 시뮬레이션 한 스텝 실행
    simulation.step()
    
    # 렌더링
    draw_world(screen, world, SCREEN_W, SCREEN_H)
    
    # SE(3) Rigid Body 시각화 및 정보 수집
    current_pose = simulation.get_current_pose()
    pose_error = simulation.get_pose_error()
    
    # Target SE(3) pose 시각화 (빨간 경계)
    draw_se3_target_pose(screen, simulation)
    
    # 정보 텍스트 표시
    font = pygame.font.Font(None, 30)
    
    # 첫 번째 줄: 환경 및 설정 정보
    if args.pose_pair:
        env_name = args.pose_pair[0]
        if 'pair_index' in locals():
            pair_info = f"SE(3) Pair: {env_name}[{pair_index}]"
        else:
            pair_info = f"SE(3) Pair: {env_name}[random]"
    else:
        pair_info = "SE(3) Manual"
    info_text = f"Mode: {pair_info} | Policy: {simulation.policy_type} | Env: {env_file or 'static'}"
    text_surface = font.render(info_text, True, (255, 255, 255))
    screen.blit(text_surface, (10, 10))
    
    # 두 번째 줄: Target SE(3) pose
    target_text = f"Target: x={target_pose[0]:.2f}, y={target_pose[1]:.2f}, yaw={target_pose[5]:.2f}"
    target_surface = font.render(target_text, True, (255, 0, 0))
    screen.blit(target_surface, (10, 40))
    
    # 세 번째 줄: Current SE(3) pose 및 error
    current_text = f"Current: x={current_pose[0]:.2f}, y={current_pose[1]:.2f}, yaw={current_pose[5]:.2f} | Error: {pose_error:.3f}"
    current_surface = font.render(current_text, True, (255, 255, 255))
    screen.blit(current_surface, (10, 70))
    
    # 네 번째 줄: Init SE(3) pose (있는 경우)
    if init_pose is not None:
        init_text = f"Init: x={init_pose[0]:.2f}, y={init_pose[1]:.2f}, yaw={init_pose[5]:.2f}"
        init_surface = font.render(init_text, True, (0, 255, 0))
        screen.blit(init_surface, (10, 100))
    
    pygame.display.flip()
    clock.tick(FPS)