#!/usr/bin/env python3
"""
Batch Circle Environment Generator
(--count)개의 서로 다른 원형 장애물 환경을 배치 생성하는 스크립트

사용법:
    python batch_generate_circles.py --count 10000 --output-dir circle_envs_10k
    python batch_generate_circles.py --count 1000 --difficulties easy medium hard
    python batch_generate_circles.py --count 100 --start-index 5000
"""

import argparse
import os
import sys
import time
import json
from typing import List, Optional
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointcloud.circle_environment_generator import create_circle_environment
from pointcloud import PointcloudExtractor


def parse_args():
    parser = argparse.ArgumentParser(description='Batch generate circle-only environments')
    
    # 생성 설정
    parser.add_argument('--count', type=int, default=10000,
                        help='Number of environments to generate (default: 10000)')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Starting index for environment naming (default: 0)')
    parser.add_argument('--output-dir', type=str, default='circle_environments',
                        help='Output directory name (default: circle_environments)')
    
    # 난이도 설정
    parser.add_argument('--difficulties', nargs='+', 
                        choices=['tutorial', 'easy', 'medium', 'hard', 'expert', 'random'],
                        default=['tutorial', 'easy', 'medium', 'hard', 'expert', 'random'],
                        help='Difficulty levels to include (default: all)')
    parser.add_argument('--difficulty-weights', nargs='+', type=float,
                        default=[0.05, 0.2, 0.3, 0.25, 0.1, 0.1],
                        help='Weights for difficulty distribution (default: 0.05 0.2 0.3 0.25 0.1 0.1)')
    
    # 포인트클라우드 설정
    parser.add_argument('--resolution', type=float, default=0.05,
                        help='Pointcloud resolution (default: 0.05)')
    parser.add_argument('--noise-level', type=float, default=0.01,
                        help='Sensor noise level (default: 0.01)')
    parser.add_argument('--workspace-bounds', nargs=4, type=float,
                        default=[-1, 11, -1, 11],
                        help='Workspace bounds: min_x max_x min_y max_y (default: -1 11 -1 11)')
    
    # 클러스터링 설정
    parser.add_argument('--clustering-eps', type=float, default=0.3,
                        help='DBSCAN clustering epsilon (default: 0.3)')
    parser.add_argument('--min-samples', type=int, default=5,
                        help='DBSCAN minimum samples (default: 5)')
    parser.add_argument('--obstacle-type', choices=['polygon', 'circle', 'auto'], default='auto',
                        help='Obstacle type for reconstruction (default: auto)')
    
    # 성능 설정
    parser.add_argument('--parallel', type=int, default=4,
                        help='Number of parallel processes (default: 4)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for progress reporting (default: 100)')
    
    # 기타 옵션
    parser.add_argument('--save-images', action='store_true',
                        help='Save environment visualization images')
    parser.add_argument('--seed-base', type=int, default=42,
                        help='Base seed for random generation (default: 42)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show configuration without generating')
    
    return parser.parse_args()


def generate_single_environment(args: tuple) -> dict:
    """단일 환경 생성 (멀티프로세싱용)"""
    (index, difficulty, seed, output_dir, 
     resolution, noise_level, workspace_bounds,
     clustering_eps, min_samples, obstacle_type,
     save_images) = args
    
    try:
        # 환경 생성
        world, obstacles, environment_metadata = create_circle_environment(
            difficulty=difficulty,
            seed=seed
        )
        
        if len(obstacles) == 0:
            return {
                'index': index,
                'success': False,
                'error': 'No obstacles generated',
                'filename': None
            }
        
        # 포인트클라우드 추출
        extractor = PointcloudExtractor(
            resolution=resolution,
            noise_level=noise_level,
            data_dir=output_dir
        )
        
        points = extractor.extract_from_world(world, workspace_bounds)
        
        if len(points) == 0:
            return {
                'index': index,
                'success': False,
                'error': 'No points extracted',
                'filename': None
            }
        
        # 파일명 생성
        filename = f"circle_env_{index:06d}"
        
        # 메타데이터 준비
        metadata = {
            'env_type': 'circles',
            'difficulty': difficulty,
            'resolution': resolution,
            'noise_level': noise_level,
            'workspace_bounds': workspace_bounds,
            'clustering_eps': clustering_eps,
            'min_samples': min_samples,
            'obstacle_type': obstacle_type,
            'num_points': len(points),
            'num_obstacles': len(obstacles),
            'seed': seed,
            'environment_details': environment_metadata
        }
        
        # 저장
        ply_path = extractor.save_pointcloud(points, filename, metadata=metadata)
        
        # 이미지 저장 (옵션)
        if save_images:
            try:
                save_environment_image(world, obstacles, filename, 
                                     workspace_bounds, output_dir)
            except Exception as e:
                # 이미지 저장 실패는 치명적이지 않음
                pass
        
        return {
            'index': index,
            'success': True,
            'error': None,
            'filename': filename,
            'obstacles': len(obstacles),
            'points': len(points),
            'difficulty': difficulty,
            'config': environment_metadata.get('config', {})
        }
        
    except Exception as e:
        return {
            'index': index,
            'success': False,
            'error': str(e),
            'filename': None
        }


def save_environment_image(world, obstacles, filename: str, workspace_bounds, output_dir: str):
    """환경 이미지 저장"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 작업공간 경계 설정
        min_x, max_x, min_y, max_y = workspace_bounds
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect('equal')
        
        # 로봇 베이스 표시
        ax.plot(0, 0, 'ro', markersize=8, label='Robot Base')
        
        # 장애물 그리기
        for obstacle in obstacles:
            if obstacle.fixtures:
                fixture = obstacle.fixtures[0]
                pos = obstacle.position
                
                if hasattr(fixture.shape, 'radius'):
                    # 원형 장애물
                    circle = patches.Circle((pos.x, pos.y), fixture.shape.radius,
                                          facecolor='lightblue', edgecolor='darkblue', alpha=0.7)
                    ax.add_patch(circle)
        
        ax.set_title(f'Circle Environment: {filename}')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 이미지 저장
        image_path = os.path.join(output_dir, f"{filename}_scene.jpg")
        plt.savefig(image_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Failed to save image for {filename}: {e}")


def main():
    args = parse_args()
    
    # 출력 디렉토리 생성
    output_dir = os.path.join("data", "pointcloud", args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Batch Circle Environment Generation ===")
    print(f"Target count: {args.count}")
    print(f"Start index: {args.start_index}")
    print(f"Output directory: {output_dir}")
    print(f"Difficulties: {args.difficulties}")
    print(f"Difficulty weights: {args.difficulty_weights}")
    print(f"Resolution: {args.resolution}")
    print(f"Parallel processes: {args.parallel}")
    print(f"Save images: {args.save_images}")
    
    if args.dry_run:
        print("\nDry run mode - configuration shown above")
        return
    
    # 난이도 분포 계산
    if len(args.difficulty_weights) != len(args.difficulties):
        # 가중치가 맞지 않으면 균등 분포 사용
        args.difficulty_weights = [1.0 / len(args.difficulties)] * len(args.difficulties)
    
    # 각 환경에 대한 설정 생성
    tasks = []
    for i in range(args.count):
        index = args.start_index + i
        
        # 난이도 선택 (가중치 적용)
        difficulty = random.choices(args.difficulties, weights=args.difficulty_weights)[0]
        
        # 시드 생성 (재현 가능하도록)
        seed = args.seed_base + index
        
        task_args = (
            index, difficulty, seed, output_dir,
            args.resolution, args.noise_level, tuple(args.workspace_bounds),
            args.clustering_eps, args.min_samples, args.obstacle_type,
            args.save_images
        )
        tasks.append(task_args)
    
    print(f"\nStarting generation of {len(tasks)} environments...")
    
    # 진행 상황 추적
    start_time = time.time()
    completed = 0
    success_count = 0
    failed_envs = []
    difficulty_stats = {d: 0 for d in args.difficulties}
    
    # 멀티프로세싱으로 생성
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        # 작업 제출
        future_to_index = {executor.submit(generate_single_environment, task): task[0] 
                          for task in tasks}
        
        # 결과 수집
        for future in as_completed(future_to_index):
            result = future.result()
            completed += 1
            
            if result['success']:
                success_count += 1
                difficulty_stats[result['difficulty']] += 1
                
                if completed % args.batch_size == 0 or completed == len(tasks):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (len(tasks) - completed) / rate if rate > 0 else 0
                    
                    print(f"Progress: {completed}/{len(tasks)} ({completed/len(tasks)*100:.1f}%) "
                          f"- Success: {success_count} - Rate: {rate:.1f} env/s - ETA: {eta:.0f}s")
            else:
                failed_envs.append({
                    'index': result['index'],
                    'error': result['error']
                })
                print(f"Failed env_{result['index']:06d}: {result['error']}")
    
    # 결과 요약
    total_time = time.time() - start_time
    print(f"\n=== Generation Complete ===")
    print(f"Total time: {total_time:.1f}s")
    print(f"Successfully generated: {success_count}/{args.count} ({success_count/args.count*100:.1f}%)")
    print(f"Average rate: {args.count/total_time:.1f} environments/second")
    
    print(f"\nDifficulty distribution:")
    for difficulty, count in difficulty_stats.items():
        print(f"  {difficulty}: {count} ({count/success_count*100:.1f}%)")
    
    if failed_envs:
        print(f"\nFailed environments: {len(failed_envs)}")
        for fail in failed_envs[:10]:  # 처음 10개만 표시
            print(f"  env_{fail['index']:06d}: {fail['error']}")
        if len(failed_envs) > 10:
            print(f"  ... and {len(failed_envs) - 10} more")
    
    # 요약 메타데이터 저장
    summary = {
        'generation_info': {
            'total_requested': args.count,
            'successfully_generated': success_count,
            'failed_count': len(failed_envs),
            'start_index': args.start_index,
            'generation_time_seconds': total_time,
            'generation_rate_per_second': args.count / total_time
        },
        'configuration': {
            'difficulties': args.difficulties,
            'difficulty_weights': args.difficulty_weights,
            'resolution': args.resolution,
            'noise_level': args.noise_level,
            'workspace_bounds': args.workspace_bounds,
            'clustering_eps': args.clustering_eps,
            'min_samples': args.min_samples,
            'obstacle_type': args.obstacle_type
        },
        'difficulty_distribution': difficulty_stats,
        'failed_environments': failed_envs
    }
    
    summary_path = os.path.join(output_dir, 'generation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    print(f"Environments saved in: {output_dir}")


if __name__ == "__main__":
    main() 