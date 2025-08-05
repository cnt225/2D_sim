#!/usr/bin/env python3
"""
11개 환경 (00-10)에 대한 테스트 궤적 생성
- 새로운 collision margin (0.1m) 사용
- 첫 번째 pose pair만 사용
- circle_envs_10k 디렉토리에 직접 저장
- naming scheme: circle_env_XXXXXX_pair_1_traj_rb3.json
"""

import subprocess
import time
import json
from pathlib import Path
import sys

def main():
    print('🚀 Starting test trajectory generation for environments 00-10...')
    print('   Using CONSERVATIVE collision margin: 0.1m')
    print('   Using only the FIRST pose pair from each environment')
    print('   Direct storage in circle_envs_10k directory')
    
    total_envs = 11  # 00부터 10까지
    success_count = 0
    failed_count = 0
    start_time = time.time()
    
    # 출력 디렉토리 확인
    output_dir = Path('../../../data/trajectories/circle_envs_10k')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(0, 11):  # 0부터 10까지
        env_id = f'circle_env_{i:06d}'
        
        print(f'\n=== Processing {env_id} ({i+1}/11) ===')
        
        # 파일 경로 설정
        pose_pairs_file = f'../../../data/pose_pairs/circle_envs_10k/{env_id}_rb_3_pairs.json'
        pointcloud_file = f'../../../data/pointcloud/circle_envs_10k/{env_id}.ply'
        
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
                    'description': f'First SE(3) pose pair for {env_id}'
                },
                'generation_info': {
                    'source_poses': data['generation_info']['source_poses'],
                    'generated_pairs': 1,
                    'generation_method': 'first_pair_only',
                    'collision_margin': '0.1m (conservative for B-spline)'
                }
            }
            
            # 임시 첫 번째 pair 파일 저장
            temp_pair_file = f'{env_id}_first_pair_temp.json'
            with open(temp_pair_file, 'w') as f:
                json.dump(first_pair_data, f, indent=2)
            
            # RRT 궤적 생성 (새로운 collision margin 사용)
            print(f'🔧 Generating RRT trajectory with 0.1m safety margin...')
            cmd = [
                'python', 'se3_trajectory_generator.py',
                '--rigid_body_id', '3',
                '--pose_pairs_file', temp_pair_file,
                '--pointcloud_file', pointcloud_file,
                '--output_dir', str(output_dir),
                '--max_planning_time', '15.0'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # 임시 파일 정리
            Path(temp_pair_file).unlink(missing_ok=True)
            
            if result.returncode != 0:
                print(f'❌ RRT generation failed for {env_id}')
                print(f'   Error: {result.stderr.strip()[:100]}...')
                failed_count += 1
                continue
            
            # 생성된 RRT 궤적 파일 찾기 및 이름 변경
            rrt_files = list(output_dir.glob('trajectory_*.json'))
            
            if not rrt_files:
                print(f'❌ RRT trajectory file not found for {env_id}')
                failed_count += 1
                continue
            
            # 가장 최근 파일을 새로운 naming scheme으로 변경
            latest_file = sorted(rrt_files, key=lambda x: x.stat().st_mtime)[-1]
            new_filename = f'{env_id}_pair_1_traj_rb3.json'
            final_file = output_dir / new_filename
            
            # 기존 파일이 있으면 삭제하고 이름 변경
            if final_file.exists():
                final_file.unlink()
            latest_file.rename(final_file)
            
            print(f'✅ RRT trajectory saved: {new_filename}')
            success_count += 1
                
        except subprocess.TimeoutExpired:
            print(f'❌ RRT generation timeout for {env_id}')
            failed_count += 1
            # 임시 파일 정리
            Path(temp_pair_file).unlink(missing_ok=True)
            
        except Exception as e:
            print(f'❌ Unexpected error for {env_id}: {str(e)[:100]}...')
            failed_count += 1
            # 임시 파일 정리
            Path(f'{env_id}_first_pair_temp.json').unlink(missing_ok=True)
    
    # 최종 통계
    total_time = time.time() - start_time
    print(f'\n🎉 Test trajectory generation completed!')
    print(f'   Total environments: {total_envs}')
    print(f'   Successful: {success_count}')
    print(f'   Failed: {failed_count}')
    print(f'   Success rate: {success_count*100/total_envs:.1f}%')
    print(f'   Total time: {total_time:.0f}s ({total_time/60:.1f}m)')
    if success_count > 0:
        print(f'   Average time per success: {total_time/success_count:.1f}s')
    print(f'\n📁 Output directory: {output_dir}')
    
    # 생성된 파일 목록 출력
    if success_count > 0:
        print(f'\n📋 Generated files:')
        for i in range(0, 11):
            env_id = f'circle_env_{i:06d}'
            filename = f'{env_id}_pair_1_traj_rb3.json'
            file_path = output_dir / filename
            if file_path.exists():
                size_kb = file_path.stat().st_size // 1024
                print(f'   ✅ {filename} ({size_kb}KB)')

if __name__ == '__main__':
    main()