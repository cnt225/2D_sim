#!/usr/bin/env python3
"""
10,000개 환경에 대한 RRT + B-spline 궤적 생성 (첫 번째 pose pair만 사용)
"""

import subprocess
import time
import json
from pathlib import Path
import sys

def main():
    print('🚀 Starting mass trajectory generation for 10,000 environments...')
    print('   Using only the FIRST pose pair from each environment')
    
    total_envs = 10000
    success_count = 0
    failed_count = 0
    start_time = time.time()
    
    # 출력 디렉토리 생성
    Path('../../../data/trajectories/circle_envs_10k_bsplined').mkdir(parents=True, exist_ok=True)
    Path('../../../data/trajectories/circle_envs_10k_temp').mkdir(parents=True, exist_ok=True)
    
    for i in range(0, 10000):  # 0부터 9999까지
        env_id = f'circle_env_{i:06d}'
        
        print(f'\n=== Processing {env_id} ({i+1}/10000) ===')
        
        # 파일 경로 설정
        pose_pairs_file = f'../../../data/pose_pairs/circle_envs_10k/{env_id}_rb_3_pairs.json'
        pointcloud_file = f'../../../data/pointcloud/circle_envs_10k/{env_id}.ply'
        output_dir = '../../../data/trajectories/circle_envs_10k_temp'
        
        # 파일 존재 확인
        if not Path(pose_pairs_file).exists():
            print(f'❌ Pose pairs file not found: {pose_pairs_file}')
            failed_count += 1
            continue
            
        if not Path(pointcloud_file).exists():
            print(f'❌ Pointcloud file not found: {pointcloud_file}')
            failed_count += 1
            continue
        
        try:
            # 첫 번째 pose pair만 추출해서 임시 파일 생성
            with open(pose_pairs_file, 'r') as f:
                data = json.load(f)
            
            if 'pose_pairs' not in data or 'data' not in data['pose_pairs']:
                print(f'❌ Invalid pose pairs format: {pose_pairs_file}')
                failed_count += 1
                continue
                
            pairs = data['pose_pairs']['data']
            if len(pairs) == 0:
                print(f'❌ No pose pairs found: {pose_pairs_file}')
                failed_count += 1
                continue
            
            # 첫 번째 pair만 사용
            first_pair_data = {
                'source_file': data['source_file'],
                'environment': data['environment'],
                'pose_pairs': {
                    'data': [pairs[0]],  # 첫 번째만
                    'count': 1,
                    'format': 'se3_pose_pairs',
                    'description': 'Single first SE(3) pose pair for mass generation'
                },
                'generation_info': {
                    'source_poses': data['generation_info']['source_poses'],
                    'generated_pairs': 1,
                    'generation_method': 'first_pair_only'
                }
            }
            
            # 임시 첫 번째 pair 파일 저장
            temp_pair_file = f'{output_dir}/{env_id}_first_pair.json'
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(temp_pair_file, 'w') as f:
                json.dump(first_pair_data, f, indent=2)
            
            # RRT 궤적 생성
            print(f'🔧 Generating RRT trajectory for first pose pair...')
            cmd = [
                'python', 'se3_trajectory_generator.py',
                '--rigid_body_id', '3',
                '--pose_pairs_file', temp_pair_file,
                '--pointcloud_file', pointcloud_file,
                '--output_dir', output_dir,
                '--max_planning_time', '15.0'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # 임시 pair 파일 삭제
            Path(temp_pair_file).unlink(missing_ok=True)
            
            if result.returncode != 0:
                print(f'❌ RRT generation failed for {env_id}')
                print(f'   Error: {result.stderr.strip()[:100]}...')
                failed_count += 1
                continue
            
            # 생성된 RRT 궤적 파일 찾기 (최신 trajectory 파일)
            output_path = Path(output_dir)
            rrt_files = [f for f in output_path.glob('trajectory_*.json') if f.name != temp_pair_file.split('/')[-1]]
            
            if not rrt_files:
                print(f'❌ RRT trajectory file not found for {env_id}')
                failed_count += 1
                continue
            
            # 가장 최근 파일 사용
            rrt_file = str(sorted(rrt_files, key=lambda x: x.stat().st_mtime)[-1])
            print(f'✅ RRT trajectory generated: {Path(rrt_file).name}')
            
            # B-spline 스무딩 적용
            print(f'🎯 Applying B-spline smoothing...')
            try:
                from bspline_smoothing import create_bsplined_trajectory_file
                
                output_file = create_bsplined_trajectory_file(
                    rrt_file,
                    output_dir='../../../data/trajectories/circle_envs_10k_bsplined',
                    degree=3,
                    smoothing_factor=0.1,
                    density_multiplier=2
                )
                print(f'✅ B-spline smoothing completed: {Path(output_file).name}')
                success_count += 1
                
                # 임시 RRT 파일 삭제
                Path(rrt_file).unlink(missing_ok=True)
                
            except Exception as e:
                print(f'❌ B-spline smoothing failed: {str(e)[:100]}...')
                failed_count += 1
                # RRT 파일은 보존해서 나중에 재시도 가능하게
                
        except subprocess.TimeoutExpired:
            print(f'❌ RRT generation timeout for {env_id}')
            failed_count += 1
            # 임시 파일들 정리
            Path(f'{output_dir}/{env_id}_first_pair.json').unlink(missing_ok=True)
            
        except Exception as e:
            print(f'❌ Unexpected error for {env_id}: {str(e)[:100]}...')
            failed_count += 1
        
        # 진행 상황 출력 (100개마다)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (total_envs - i - 1)
            
            print(f'\n📊 Progress Report:')
            print(f'   Processed: {i+1}/{total_envs} ({(i+1)*100/total_envs:.1f}%)')
            print(f'   Success: {success_count}, Failed: {failed_count}')
            if i >= 0:
                print(f'   Success rate: {success_count*100/(i+1):.1f}%')
            print(f'   Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)')
            print(f'   Remaining: {remaining:.0f}s ({remaining/60:.1f}m)')
            print(f'   ETA: {(elapsed + remaining)/3600:.1f}h')
            
            # 임시 체크포인트 저장
            checkpoint = {
                'processed': i + 1,
                'success_count': success_count,
                'failed_count': failed_count,
                'elapsed_time': elapsed,
                'timestamp': time.time()
            }
            with open('generation_checkpoint.json', 'w') as f:
                json.dump(checkpoint, f, indent=2)
    
    # 최종 통계
    total_time = time.time() - start_time
    print(f'\n🎉 Mass trajectory generation completed!')
    print(f'   Total environments: {total_envs}')
    print(f'   Successful: {success_count}')
    print(f'   Failed: {failed_count}')
    print(f'   Success rate: {success_count*100/total_envs:.1f}%')
    print(f'   Total time: {total_time:.0f}s ({total_time/60:.1f}m = {total_time/3600:.1f}h)')
    if success_count > 0:
        print(f'   Average time per success: {total_time/success_count:.1f}s')
    print(f'\n📁 Output directory: ../../../data/trajectories/circle_envs_10k_bsplined')
    
    # 최종 리포트 저장
    final_report = {
        'total_environments': total_envs,
        'successful': success_count,
        'failed': failed_count,
        'success_rate': success_count*100/total_envs if total_envs > 0 else 0,
        'total_time_seconds': total_time,
        'total_time_hours': total_time/3600,
        'avg_time_per_success': total_time/success_count if success_count > 0 else 0,
        'output_directory': '../../../data/trajectories/circle_envs_10k_bsplined',
        'timestamp': time.time(),
        'settings': {
            'range_value': 0.05,
            'bspline_degree': 3,
            'smoothing_factor': 0.1,
            'density_multiplier': 2,
            'max_planning_time': 15.0,
            'pose_pairs_used': 'first_only'
        }
    }
    
    with open('final_generation_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f'✅ Final report saved: final_generation_report.json')

if __name__ == '__main__':
    main()