#!/bin/bash
# 50개 환경만 테스트

echo "🧪 Tdot 생성 테스트 (50개 환경)"
echo "================================"

# 입력 파일 정보 확인
INPUT_FILE="/home/dhkang225/2D_sim/data/trajectory/circles_only_integrated_trajs.h5"
OUTPUT_FILE="/home/dhkang225/2D_sim/data/Tdot/circles_only_integrated_trajs_Tdot_test50.h5"

echo "입력: $INPUT_FILE"
echo "출력: $OUTPUT_FILE"
echo ""

# 기존 테스트 파일 제거
if [ -f "$OUTPUT_FILE" ]; then
    echo "기존 테스트 파일 제거..."
    rm "$OUTPUT_FILE"
fi

# 50개 환경만 처리하는 임시 파일 생성
echo "50개 환경만 추출..."
python -c "
import h5py
import sys

input_file = '$INPUT_FILE'
output_file = '${INPUT_FILE}.test50.h5'

with h5py.File(input_file, 'r') as f_in:
    with h5py.File(output_file, 'w') as f_out:
        # 메타데이터 복사
        if 'metadata' in f_in:
            f_in.copy('metadata', f_out)
        
        # 처음 50개 환경만 복사
        env_count = 0
        for key in f_in.keys():
            if key != 'metadata' and env_count < 50:
                f_in.copy(key, f_out)
                env_count += 1
                
        print(f'✅ {env_count}개 환경 추출 완료')
"

# Tdot 생성 실행
echo ""
echo "Tdot 생성 시작..."
time python /home/dhkang225/2D_sim/packages/data_generator/trajectory/generate_tdot_trajectories.py \
    --input "${INPUT_FILE}.test50.h5" \
    --dt 0.01 \
    --time-policy uniform \
    --save-format 4x4 \
    --chunk-size 10 \
    --verbose

# 임시 파일 정리
rm "${INPUT_FILE}.test50.h5"

echo ""
echo "✅ 테스트 완료!"