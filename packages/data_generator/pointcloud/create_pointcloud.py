#!/usr/bin/env python3
"""
create_pointcloud.py - 포인트클라우드 생성 전용 스크립트
환경 설정과 포인트클라우드 추출을 담당

지원하는 환경 타입:
- static: 고정된 3개 장애물 환경
- random: 랜덤 생성 다양한 장애물 환경
"""
import argparse
import sys
import os
import datetime

# 상위 디렉토리를 path에 추가하여 모듈 import 가능하도록 설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointcloud import PointcloudExtractor
from robot_simulation.core.env import make_world
from pointcloud.random_environment_generator import create_random_environment
from pointcloud.circle_environment_generator import create_circle_environment


def parse_args():
    parser = argparse.ArgumentParser(description='Create pointcloud from environment (static or random)')
    
    # 출력 설정
    parser.add_argument('output_name', type=str,
                        help='Output pointcloud name (without .ply extension)')
    
    # 환경 타입 설정
    parser.add_argument('--env-type', choices=['static', 'random', 'circles'], default='static',
                        help='Environment type: static (3 fixed obstacles), random (various shapes), or circles (circle-only) (default: static)')
    
    # 기본 환경 설정
    parser.add_argument('--link-shape', choices=['rectangle', 'ellipse'], default='rectangle',
                        help='Link shape for environment generation (default: rectangle)')
    
    # 랜덤 환경 설정
    parser.add_argument('--difficulty', choices=['tutorial', 'easy', 'medium', 'hard', 'expert', 'random'], default='medium',
                        help='Environment difficulty (default: medium)')
    parser.add_argument('--num-obstacles', type=int, default=None,
                        help='Number of obstacles for random environment (overrides difficulty)')
    parser.add_argument('--obstacle-types', nargs='+', 
                        choices=['rectangle', 'circle', 'polygon', 'curve'],
                        default=['rectangle', 'circle', 'polygon', 'curve'],
                        help='Obstacle types for random environment (default: all types)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible environments')
    
    # 포인트클라우드 추출 설정
    parser.add_argument('--resolution', type=float, default=0.05,
                        help='Pointcloud resolution in meters (default: 0.05)')
    parser.add_argument('--noise-level', type=float, default=0.01,
                        help='Sensor noise level (default: 0.01)')
    parser.add_argument('--workspace-bounds', nargs=4, type=float, 
                        default=[-1, 11, -1, 11],
                        help='Workspace bounds: min_x max_x min_y max_y (default: -1 11 -1 11)')
    
    # 클러스터링 설정 (미리 메타데이터에 저장)
    parser.add_argument('--clustering-eps', type=float, default=0.3,
                        help='DBSCAN clustering epsilon for reconstruction (default: 0.3)')
    parser.add_argument('--min-samples', type=int, default=5,
                        help='DBSCAN minimum samples for reconstruction (default: 5)')
    parser.add_argument('--obstacle-type', choices=['polygon', 'circle', 'auto'], default='auto',
                        help='Obstacle type for reconstruction (default: auto)')
    
    # 시각화 및 저장 옵션
    parser.add_argument('--visualize', action='store_true',
                        help='Show pointcloud visualization after extraction')
    parser.add_argument('--save-image', action='store_true',
                        help='Save environment visualization as JPG image')
    parser.add_argument('--save-scene', action='store_true',
                        help='Save both PLY and scene image')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Creating pointcloud: {args.output_name}.ply")
    print(f"Environment type: {args.env_type}")
    print(f"Resolution: {args.resolution}m")
    print(f"Noise level: {args.noise_level}")
    print(f"Workspace bounds: {args.workspace_bounds}")
    
    if args.env_type == 'random':
        print(f"Difficulty: {args.difficulty}")
        print(f"Obstacle types: {args.obstacle_types}")
        if args.seed is not None:
            print(f"Seed: {args.seed}")
    elif args.env_type == 'circles':
        print(f"Difficulty: {args.difficulty}")
        if args.seed is not None:
            print(f"Seed: {args.seed}")
    print()
    
    # 1. 환경 생성
    if args.env_type == 'static':
        print("Generating static environment...")
        world, links, obstacles = make_world(0, 'static')  # geometry_id=0 for static
        environment_metadata = {
            'environment_type': 'static',
            'num_obstacles': len(obstacles)
        }
    elif args.env_type == 'random':
        print("Generating random environment...")
        world, obstacles, environment_metadata = create_random_environment(
            difficulty=args.difficulty,
            obstacle_types=args.obstacle_types,
            seed=args.seed
        )
        if args.num_obstacles is not None:
            print(f"Requested {args.num_obstacles} obstacles, generated {len(obstacles)}")
        
        # 랜덤 환경에는 로봇이 포함되지 않으므로 더미 links 생성
        links = []
    else:  # circles
        print("Generating circle-only environment...")
        world, obstacles, environment_metadata = create_circle_environment(
            difficulty=args.difficulty,
            seed=args.seed
        )
        
        # 원형 환경에는 로봇이 포함되지 않으므로 더미 links 생성
        links = []
    
    print(f"Created environment with {len(obstacles)} obstacles")
    
    # 2. 포인트클라우드 추출기 생성
    extractor = PointcloudExtractor(
        resolution=args.resolution, 
        noise_level=args.noise_level
    )
    
    # 3. 포인트클라우드 추출
    print("Extracting pointcloud...")
    workspace_bounds = tuple(args.workspace_bounds)
    
    points = extractor.extract_from_world(world, workspace_bounds)
    print(f"Extracted {len(points)} points")
    
    # 4. 메타데이터 준비 (클러스터링 설정 + 환경 정보 포함)
    metadata = {
        'env_type': args.env_type,
        'link_shape': args.link_shape,
        'resolution': args.resolution,
        'noise_level': args.noise_level,
        'workspace_bounds': workspace_bounds,
        'clustering_eps': args.clustering_eps,
        'min_samples': args.min_samples,
        'obstacle_type': args.obstacle_type,
        'num_points': len(points),
        'num_obstacles': len(obstacles),
        'generation_timestamp': datetime.datetime.now().isoformat()
    }
    
    # 랜덤 환경의 경우 추가 메타데이터 포함
    if args.env_type == 'random':
        metadata.update({
            'difficulty': args.difficulty,
            'obstacle_types_used': args.obstacle_types,
            'seed': args.seed,
            'environment_details': environment_metadata
        })
    elif args.env_type == 'circles':
        metadata.update({
            'difficulty': args.difficulty,
            'seed': args.seed,
            'environment_details': environment_metadata
        })
    
    # 5. 파일 저장
    print("Saving pointcloud...")
    filepath = extractor.save_pointcloud(points, args.output_name, metadata=metadata)
    print(f"Saved: {filepath}")
    print(f"Metadata saved with clustering settings")
    
    # 6. 이미지 저장 (옵션)
    if args.save_image or args.save_scene:
        print("Saving environment visualization...")
        try:
            image_path = save_environment_image(world, obstacles, args.output_name, 
                                              workspace_bounds, args.env_type)
            print(f"Environment image saved: {image_path}")
        except Exception as e:
            print(f"Image saving failed: {e}")
    
    # 7. 시각화 (옵션)
    if args.visualize:
        print("Showing visualization...")
        try:
            if args.env_type == 'static':
                title = f"Static Environment: {args.output_name}"
            else:
                title = f"Random Environment: {args.output_name} ({args.difficulty})"
            
            extractor.visualize_pointcloud(points, title=title)
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    print("\nPointcloud creation completed!")
    print(f"Use with: python main.py --env {args.output_name}.ply --target X Y")
    
    if args.env_type == 'random':
        print(f"Environment details: {args.difficulty} difficulty with {len(obstacles)} obstacles")
        if args.seed is not None:
            print(f"Reproducible with seed: {args.seed}")


def save_environment_image(world, obstacles, output_name, workspace_bounds, env_type):
    """환경을 이미지로 저장"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 작업공간 경계 설정
        min_x, max_x, min_y, max_y = workspace_bounds
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect('equal')
        
        # 로봇 베이스 표시
        ax.plot(0, 0, 'ro', markersize=10, label='Robot Base')
        
        # 장애물 그리기
        colors = ['gray', 'darkgray', 'lightgray', 'dimgray']
        
        for i, obstacle in enumerate(obstacles):
            color = colors[i % len(colors)]
            
            # 장애물의 fixture 정보 가져오기
            if obstacle.fixtures:
                fixture = obstacle.fixtures[0]
                pos = obstacle.position
                
                if hasattr(fixture.shape, 'radius'):
                    # 원형 장애물
                    circle = patches.Circle((pos.x, pos.y), fixture.shape.radius, 
                                          facecolor=color, edgecolor='black', alpha=0.7)
                    ax.add_patch(circle)
                    
                elif hasattr(fixture.shape, 'vertices'):
                    # 다각형 장애물
                    vertices = []
                    for vertex in fixture.shape.vertices:
                        # 로컬 좌표를 월드 좌표로 변환
                        world_point = obstacle.GetWorldPoint(vertex)
                        vertices.append([world_point.x, world_point.y])
                    
                    polygon = patches.Polygon(vertices, facecolor=color, 
                                            edgecolor='black', alpha=0.7)
                    ax.add_patch(polygon)
        
        # 제목과 라벨
        if env_type == 'static':
            title = f'Static Environment: {output_name}'
        else:
            title = f'Random Environment: {output_name}'
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 환경 정보 텍스트 박스
        info_text = f'Obstacles: {len(obstacles)}\nWorkspace: [{min_x}, {max_x}] × [{min_y}, {max_y}]'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 이미지 저장
        image_path = os.path.join("pointcloud", "data", f"{output_name}_scene.jpg")
        plt.savefig(image_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return image_path
        
    except ImportError:
        raise Exception("matplotlib not available for image saving")
    except Exception as e:
        raise Exception(f"Failed to save image: {e}")


if __name__ == "__main__":
    main()
