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
import shutil
import tempfile
from typing import List, Optional, Dict, Any
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
from pathlib import Path

# 현재 디렉토리에서 import
from circle_environment_generator import create_circle_environment
from utils.pointcloud_extractor import PointcloudExtractor


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


def safe_save_pointcloud(extractor: PointcloudExtractor, points, filename: str, 
                        metadata: Optional[Dict] = None, temp_dir: Optional[str] = None) -> str:
    """안전한 포인트클라우드 파일 저장 (버퍼링 제어 및 검증 포함)"""
    if temp_dir is None:
        temp_dir = extractor.data_dir
    
    # 임시 파일에 먼저 저장
    temp_ply_path = os.path.join(temp_dir, f"{filename}_temp.ply")
    final_ply_path = os.path.join(extractor.data_dir, f"{filename}.ply")
    
    try:
        # PLY 파일 저장 (안전한 방식)
        with open(temp_ply_path, 'w', buffering=1) as f:  # 라인 버퍼링
            # PLY 헤더
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            f.flush()  # 헤더 강제 쓰기
            
            # 포인트 데이터 (배치 처리로 메모리 효율성 향상)
            batch_size = 1000
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                for point in batch:
                    f.write(f"{point[0]:.6f} {point[1]:.6f} 0.000000\n")
                f.flush()  # 배치마다 강제 쓰기
            
            f.flush()  # 최종 강제 쓰기
            os.fsync(f.fileno())  # 커널 버퍼까지 강제 동기화
        
        # 파일 검증
        if not validate_ply_file(temp_ply_path, len(points)):
            raise ValueError(f"PLY file validation failed for {filename}")
        
        # 최종 위치로 원자적 이동
        shutil.move(temp_ply_path, final_ply_path)
        
        # 메타데이터 저장 (별도 처리)
        if metadata is not None:
            import datetime
            metadata['generation_timestamp'] = datetime.datetime.now().isoformat()
            metadata['file_validation'] = True
            
            temp_meta_path = os.path.join(temp_dir, f"{filename}_meta_temp.json")
            final_meta_path = os.path.join(extractor.data_dir, f"{filename}_meta.json")
            
            with open(temp_meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            shutil.move(temp_meta_path, final_meta_path)
        
        return final_ply_path
        
    except Exception as e:
        # 임시 파일 정리
        for temp_file in [temp_ply_path, temp_ply_path.replace('.ply', '_meta.json')]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        raise e


def validate_ply_file(ply_path: str, expected_points: int) -> bool:
    """PLY 파일 검증"""
    try:
        with open(ply_path, 'r') as f:
            lines = f.readlines()
        
        # 기본 구조 검증
        if len(lines) < 8:  # 최소 헤더 라인 수
            return False
            
        if not lines[0].strip() == 'ply':
            return False
        
        if not lines[1].strip() == 'format ascii 1.0':
            return False
        
        # 포인트 수 검증
        header_line = lines[3].strip()
        if not header_line.startswith('element vertex'):
            return False
            
        try:
            header_points = int(header_line.split()[-1])
        except (IndexError, ValueError):
            return False
        
        # 실제 데이터 라인 수 확인 (헤더 8줄 제외)
        actual_data_lines = len(lines) - 8
        
        # 포인트 수 일치 확인
        if header_points != expected_points:
            print(f"❌ Header mismatch: header={header_points}, expected={expected_points}")
            return False
            
        if actual_data_lines != expected_points:
            print(f"❌ Data mismatch: data_lines={actual_data_lines}, expected={expected_points}")
            return False
        
        # 마지막 라인이 완전한지 확인
        last_line = lines[-1].strip()
        if last_line:
            parts = last_line.split()
            if len(parts) != 3:  # x, y, z 좌표
                print(f"❌ Incomplete last line: '{last_line}'")
                return False
                
            # 좌표가 숫자인지 확인
            try:
                for part in parts:
                    float(part)
            except ValueError:
                print(f"❌ Invalid coordinates in last line: '{last_line}'")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ PLY validation error: {e}")
        return False


def generate_single_environment(args: tuple) -> dict:
    """단일 환경 생성 (멀티프로세싱용 - 안전한 파일 쓰기 적용)"""
    (index, difficulty, seed, output_dir, 
     resolution, noise_level, workspace_bounds,
     clustering_eps, min_samples, obstacle_type,
     save_images) = args
    
    # 프로세스별 임시 디렉토리 생성 (파일 충돌 방지)
    process_id = os.getpid()
    temp_dir = tempfile.mkdtemp(prefix=f"env_gen_{process_id}_")
    
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
                'filename': None,
                'validation_info': None
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
                'filename': None,
                'validation_info': None
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
            'environment_details': environment_metadata,
            'process_id': process_id,
            'temp_dir_used': temp_dir
        }
        
        # 안전한 파일 저장 (임시 디렉토리 사용)
        ply_path = safe_save_pointcloud(extractor, points, filename, 
                                       metadata=metadata, temp_dir=temp_dir)
        
        # 생성된 파일 재검증
        final_ply_path = os.path.join(output_dir, f"{filename}.ply")
        validation_result = validate_ply_file(final_ply_path, len(points))
        
        if not validation_result:
            return {
                'index': index,
                'success': False,
                'error': f'Final file validation failed for {filename}',
                'filename': filename,
                'validation_info': {'final_validation': False, 'expected_points': len(points)}
            }
        
        # 이미지 저장 (옵션)
        image_saved = False
        if save_images:
            try:
                save_environment_image(world, obstacles, filename, 
                                     workspace_bounds, output_dir)
                image_saved = True
            except Exception as e:
                # 이미지 저장 실패는 치명적이지 않음
                print(f"Warning: Image save failed for {filename}: {e}")
        
        return {
            'index': index,
            'success': True,
            'error': None,
            'filename': filename,
            'obstacles': len(obstacles),
            'points': len(points),
            'difficulty': difficulty,
            'config': environment_metadata.get('config', {}),
            'validation_info': {
                'final_validation': validation_result,
                'expected_points': len(points),
                'image_saved': image_saved,
                'temp_dir': temp_dir
            }
        }
        
    except Exception as e:
        return {
            'index': index,
            'success': False,
            'error': f"Generation failed: {str(e)}",
            'filename': None,
            'validation_info': {'error_type': type(e).__name__, 'temp_dir': temp_dir}
        }
    
    finally:
        # 임시 디렉토리 정리
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to cleanup temp dir {temp_dir}: {e}")


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
    
    # 출력 디렉토리 생성 (시뮬레이션 환경 경로)
    base_data_dir = "/home/dhkang225/2D_sim/data/pointcloud"
    output_dir = os.path.join(base_data_dir, args.output_dir)
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
    validation_failures = []
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
                
                # 검증 정보 확인
                validation_info = result.get('validation_info', {})
                if not validation_info.get('final_validation', True):
                    validation_failures.append({
                        'index': result['index'],
                        'filename': result['filename'],
                        'validation_info': validation_info
                    })
                
                if completed % args.batch_size == 0 or completed == len(tasks):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (len(tasks) - completed) / rate if rate > 0 else 0
                    
                    print(f"Progress: {completed}/{len(tasks)} ({completed/len(tasks)*100:.1f}%) "
                          f"- Success: {success_count} - Rate: {rate:.1f} env/s - ETA: {eta:.0f}s")
            else:
                failed_envs.append({
                    'index': result['index'],
                    'error': result['error'],
                    'validation_info': result.get('validation_info')
                })
                print(f"❌ Failed env_{result['index']:06d}: {result['error']}")
    
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
        print(f"\n❌ Failed environments: {len(failed_envs)}")
        for fail in failed_envs[:10]:  # 처음 10개만 표시
            print(f"  env_{fail['index']:06d}: {fail['error']}")
        if len(failed_envs) > 10:
            print(f"  ... and {len(failed_envs) - 10} more")
    
    if validation_failures:
        print(f"\n⚠️  Validation issues: {len(validation_failures)}")
        for fail in validation_failures[:5]:  # 처음 5개만 표시
            print(f"  {fail['filename']}: validation failed")
        if len(validation_failures) > 5:
            print(f"  ... and {len(validation_failures) - 5} more")
    
    # 요약 메타데이터 저장
    summary = {
        'generation_info': {
            'total_requested': args.count,
            'successfully_generated': success_count,
            'failed_count': len(failed_envs),
            'validation_failures': len(validation_failures),
            'start_index': args.start_index,
            'generation_time_seconds': total_time,
            'generation_rate_per_second': args.count / total_time,
            'file_safety_improvements': True
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
        'failed_environments': failed_envs,
        'validation_failures': validation_failures,
        'improvements_applied': {
            'safe_file_writing': True,
            'temp_directory_isolation': True,
            'file_validation': True,
            'atomic_file_moves': True,
            'buffer_control': True
        }
    }
    
    summary_path = os.path.join(output_dir, 'generation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    print(f"Environments saved in: {output_dir}")


if __name__ == "__main__":
    main() 