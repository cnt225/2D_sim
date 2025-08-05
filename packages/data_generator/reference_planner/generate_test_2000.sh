#!/bin/bash

# 2,000개 환경에 대한 RRT + B-spline 궤적 생성 테스트 스크립트

echo "🚀 Starting test trajectory generation for 2,000 environments (000001-002000)..."

# 카운터 초기화
total_envs=2000
success_count=0
failed_count=0
start_time=$(date +%s)

# 출력 디렉토리 생성
mkdir -p ../../../data/trajectories/circle_envs_10k_bsplined

# 각 환경에 대해 처리 (1-2000번)
for i in $(seq -f "%06g" 1 2000); do
    env_id="circle_env_${i}"
    
    echo ""
    echo "=== Processing ${env_id} ($((i))/2000) ==="
    
    # 파일 경로 설정
    pose_pairs_file="../../../data/pose_pairs/circle_envs_10k/${env_id}_rb_3_pairs.json"
    pointcloud_file="../../../data/pointcloud/circle_envs_10k/${env_id}.ply"
    output_dir="../../../data/trajectories/circle_envs_10k_temp"
    
    # 파일 존재 확인
    if [[ ! -f "$pose_pairs_file" ]]; then
        echo "❌ Pose pairs file not found: $pose_pairs_file"
        ((failed_count++))
        continue
    fi
    
    if [[ ! -f "$pointcloud_file" ]]; then
        echo "❌ Pointcloud file not found: $pointcloud_file"
        ((failed_count++))
        continue
    fi
    
    # RRT 궤적 생성 (첫 번째 pose pair만 사용)
    echo "🔧 Generating RRT trajectory..."
    python se3_trajectory_generator.py \
        --rigid_body_id 3 \
        --pose_pairs_file "$pose_pairs_file" \
        --pointcloud_file "$pointcloud_file" \
        --output_dir "$output_dir" \
        --max_planning_time 15.0 > /dev/null 2>&1
    
    if [[ $? -ne 0 ]]; then
        echo "❌ RRT generation failed for $env_id"
        ((failed_count++))
        continue
    fi
    
    # 생성된 RRT 궤적 파일 찾기
    rrt_file=$(find "$output_dir" -name "*${env_id}*.json" -type f | head -1)
    
    if [[ -z "$rrt_file" ]] || [[ ! -f "$rrt_file" ]]; then
        echo "❌ RRT trajectory file not found for $env_id"
        ((failed_count++))
        continue
    fi
    
    echo "✅ RRT trajectory generated: $(basename "$rrt_file")"
    
    # B-spline 스무딩 적용
    echo "🎯 Applying B-spline smoothing..."
    python -c "
from bspline_smoothing import create_bsplined_trajectory_file
import sys

try:
    output_file = create_bsplined_trajectory_file(
        '$rrt_file',
        output_dir='../../../data/trajectories/circle_envs_10k_bsplined',
        degree=3,
        smoothing_factor=0.1,
        density_multiplier=2
    )
    print(f'✅ B-splined trajectory saved')
except Exception as e:
    print(f'❌ B-spline failed: {e}')
    sys.exit(1)
" > /dev/null 2>&1
    
    if [[ $? -eq 0 ]]; then
        echo "✅ B-spline smoothing completed for $env_id"
        ((success_count++))
        
        # 임시 RRT 파일 삭제
        rm -f "$rrt_file"
    else
        echo "❌ B-spline smoothing failed for $env_id"
        ((failed_count++))
    fi
    
    # 진행 상황 출력 (50개마다)
    if (( i % 50 == 0 )); then
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        avg_time=$((elapsed / i))
        remaining=$((avg_time * (total_envs - i)))
        
        echo ""
        echo "📊 Progress Report:"
        echo "   Processed: $((i))/$total_envs ($(( i * 100 / total_envs ))%)"
        echo "   Success: $success_count, Failed: $failed_count"
        echo "   Success rate: $(( success_count * 100 / i ))%"
        echo "   Elapsed: ${elapsed}s, Remaining: ${remaining}s"
        echo ""
    fi
done

# 최종 통계
end_time=$(date +%s)
total_time=$((end_time - start_time))

echo ""
echo "🎉 Mass trajectory generation completed!"
echo "   Total environments: $total_envs"
echo "   Successful: $success_count"
echo "   Failed: $failed_count"
echo "   Success rate: $(( success_count * 100 / total_envs ))%"
echo "   Total time: ${total_time}s ($(( total_time / 60 ))m $(( total_time % 60 ))s)"
echo "   Average time per env: $(( total_time / total_envs ))s"
echo ""
echo "📁 Output directory: ../../../data/trajectories/circle_envs_10k_bsplined"