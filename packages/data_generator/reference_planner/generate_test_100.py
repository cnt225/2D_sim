#!/usr/bin/env python3
"""
100개 환경 (00-99)에 대한 RRT + B-spline + collision check
- collision margin: 0.05m (기본값)
- 올바른 geometry 설정 (1.2×0.4m)
- circle_envs_10k_bsplined 디렉토리에 저장
"""

import subprocess
import time
import json
from pathlib import Path
import sys

def main():
    print('🚀 Starting test for 100 environments (00-99)...')
    print('   Collision margin: 0.05m (default)')
    print('   Robot geometry: 1.2×0.4m (elongated_ellipse)')
    print('   Storage: circle_envs_10k_bsplined directory')
    
    total_envs = 100  # 00부터 99까지
    success_count = 0
    failed_count = 0
    collision_count = 0
    start_time = time.time()
    
    # 출력 디렉토리 생성
    output_rrt = Path('../../../data/trajectories/circle_envs_10k')
    output_bsplined = Path('../../../data/trajectories/circle_envs_10k_bsplined')
    output_rrt.mkdir(parents=True, exist_ok=True)
    output_bsplined.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for i in range(0, 100):  # 0부터 99까지
        env_id = f'circle_env_{i:06d}'
        
        print(f'\n=== Processing {env_id} ({i+1}/100) ===')
        
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
        
        result_entry = {
            'env_id': env_id,
            'rrt_success': False,
            'bspline_success': False,
            'collision_free': False,
            'collision_percentage': 0.0,
            'waypoints': 0,
            'error': None
        }
        
        try:
            # === STEP 1: RRT 생성 ===
            # 첫 번째 pose pair만 추출
            with open(pose_pairs_file, 'r') as f:
                data = json.load(f)
            
            if 'pose_pairs' not in data or 'data' not in data['pose_pairs']:
                print(f'❌ Invalid pose pairs format: {pose_pairs_file}')
                failed_count += 1
                result_entry['error'] = 'Invalid pose pairs format'
                results.append(result_entry)
                continue
                
            pairs = data['pose_pairs']['data']
            if len(pairs) == 0:
                print(f'❌ No pose pairs found: {pose_pairs_file}')
                failed_count += 1
                result_entry['error'] = 'No pose pairs found'
                results.append(result_entry)
                continue
            
            # 첫 번째 pair만 사용
            first_pair_data = {
                'source_file': data['source_file'],
                'environment': data['environment'],
                'pose_pairs': {
                    'data': [pairs[0]],
                    'count': 1,
                    'format': 'se3_pose_pairs',
                    'description': f'First SE(3) pose pair for {env_id}'
                },
                'generation_info': {
                    'source_poses': data['generation_info']['source_poses'],
                    'generated_pairs': 1,
                    'generation_method': 'first_pair_only',
                    'collision_margin': '0.05m (default)'
                }
            }
            
            # 임시 첫 번째 pair 파일 저장
            temp_pair_file = f'{env_id}_first_pair_temp.json'
            with open(temp_pair_file, 'w') as f:
                json.dump(first_pair_data, f, indent=2)
            
            # RRT 궤적 생성
            print(f'🔧 Generating RRT trajectory (margin: 0.05m)...')
            cmd = [
                'python', 'se3_trajectory_generator.py',
                '--rigid_body_id', '3',
                '--pose_pairs_file', temp_pair_file,
                '--pointcloud_file', pointcloud_file,
                '--output_dir', str(output_rrt),
                '--max_planning_time', '15.0'
            ]
            
            rrt_result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # 임시 파일 정리
            Path(temp_pair_file).unlink(missing_ok=True)
            
            if rrt_result.returncode != 0:
                print(f'❌ RRT generation failed for {env_id}')
                failed_count += 1
                result_entry['error'] = f'RRT failed: {rrt_result.stderr.strip()[:50]}...'
                results.append(result_entry)
                continue
            
            # 생성된 RRT 궤적 파일 찾기
            rrt_files = list(output_rrt.glob('trajectory_*.json'))
            if not rrt_files:
                print(f'❌ RRT trajectory file not found for {env_id}')
                failed_count += 1
                result_entry['error'] = 'RRT trajectory file not found'
                results.append(result_entry)
                continue
            
            # 가장 최근 파일을 새로운 naming으로 변경
            latest_file = sorted(rrt_files, key=lambda x: x.stat().st_mtime)[-1]
            rrt_filename = f'{env_id}_pair_1_traj_rb3.json'
            rrt_final_file = output_rrt / rrt_filename
            
            if rrt_final_file.exists():
                rrt_final_file.unlink()
            latest_file.rename(rrt_final_file)
            
            print(f'✅ RRT trajectory saved: {rrt_filename}')
            result_entry['rrt_success'] = True
            
            # === STEP 2: B-spline 스무딩 ===
            print(f'🎯 Applying B-spline smoothing...')
            try:
                from bspline_smoothing import create_bsplined_trajectory_file
                
                # B-spline 적용 (circle_envs_10k_bsplined에 저장)
                bsplined_file = create_bsplined_trajectory_file(
                    str(rrt_final_file),
                    output_dir=str(output_bsplined),
                    degree=3,
                    smoothing_factor=0.1,
                    density_multiplier=2
                )
                
                # 파일명을 환경별로 변경
                old_file = Path(bsplined_file)
                new_filename = f'{env_id}_pair_1_traj_rb3_bsplined.json'
                new_file = output_bsplined / new_filename
                
                if new_file.exists():
                    new_file.unlink()
                old_file.rename(new_file)
                
                print(f'✅ B-spline saved: {new_filename}')
                result_entry['bspline_success'] = True
                
                # === STEP 3: Collision checking ===
                print(f'🔍 Checking collisions...')
                from bspline_collision_checker import check_bsplined_trajectory_collision
                
                collision_result = check_bsplined_trajectory_collision(
                    str(new_file),
                    pointcloud_file,
                    rigid_body_id=3,
                    safety_margin=0.05,
                    check_density=1
                )
                
                result_entry['collision_free'] = collision_result['is_collision_free']
                result_entry['collision_percentage'] = collision_result['collision_percentage']
                result_entry['waypoints'] = collision_result['total_waypoints']
                
                if collision_result['is_collision_free']:
                    print(f'✅ Collision-free! ({collision_result["checked_waypoints"]} waypoints)')
                    success_count += 1
                else:
                    print(f'❌ Collision detected! {collision_result["collision_count"]}/{collision_result["checked_waypoints"]} waypoints ({collision_result["collision_percentage"]:.1f}%)')
                    collision_count += 1
                    success_count += 1  # RRT + B-spline은 성공
                
            except Exception as e:
                print(f'❌ B-spline/Collision check failed: {str(e)[:50]}...')
                result_entry['error'] = f'B-spline failed: {str(e)[:50]}...'
                
        except subprocess.TimeoutExpired:
            print(f'❌ RRT generation timeout for {env_id}')
            failed_count += 1
            result_entry['error'] = 'RRT timeout'
            Path(f'{env_id}_first_pair_temp.json').unlink(missing_ok=True)
            
        except Exception as e:
            print(f'❌ Unexpected error for {env_id}: {str(e)[:50]}...')
            failed_count += 1
            result_entry['error'] = f'Unexpected: {str(e)[:50]}...'
            Path(f'{env_id}_first_pair_temp.json').unlink(missing_ok=True)
        
        results.append(result_entry)
        
        # 중간 진행 상황 출력 (10개마다)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (total_envs - i - 1)
            
            print(f'\n📊 Progress Report ({i+1}/100):')
            print(f'   Success: {success_count}, Failed: {failed_count}, Collisions: {collision_count}')
            print(f'   Success rate: {success_count*100/(i+1):.1f}%')
            print(f'   Collision rate: {collision_count*100/max(1,success_count):.1f}%')
            print(f'   Remaining: {remaining:.0f}s ({remaining/60:.1f}m)')
    
    # 최종 통계
    total_time = time.time() - start_time
    print(f'\n🎉 Test completed for 100 environments!')
    print(f'   Total environments: {total_envs}')
    print(f'   Successful: {success_count}')
    print(f'   Failed: {failed_count}')
    print(f'   With collisions: {collision_count}')
    print(f'   Success rate: {success_count*100/total_envs:.1f}%')
    print(f'   Collision rate: {collision_count*100/max(1,success_count):.1f}%')
    print(f'   Total time: {total_time:.0f}s ({total_time/60:.1f}m)')
    
    # 상세 결과 저장
    final_report = {
        'total_environments': total_envs,
        'successful': success_count,
        'failed': failed_count,
        'with_collisions': collision_count,
        'success_rate': success_count*100/total_envs if total_envs > 0 else 0,
        'collision_rate': collision_count*100/success_count if success_count > 0 else 0,
        'total_time_seconds': total_time,
        'total_time_minutes': total_time/60,
        'settings': {
            'collision_margin': '0.05m',
            'robot_geometry': '1.2×0.4m (elongated_ellipse)',
            'bspline_degree': 3,
            'smoothing_factor': 0.1,
            'density_multiplier': 2
        },
        'detailed_results': results
    }
    
    with open('test_100_environments_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f'✅ Detailed report saved: test_100_environments_report.json')
    
    # 충돌이 있는 환경들 요약
    collision_envs = [r for r in results if r['collision_free'] == False and r['bspline_success']]
    if collision_envs:
        print(f'\n⚠️ Environments with collisions ({len(collision_envs)}):')
        for env in collision_envs[:5]:  # 처음 5개만 표시
            print(f'   {env["env_id"]}: {env["collision_percentage"]:.1f}% collision')
        if len(collision_envs) > 5:
            print(f'   ... and {len(collision_envs)-5} more')

if __name__ == '__main__':
    main()