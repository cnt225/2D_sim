#!/bin/bash
# 전체 1000개 환경 Tdot 생성 (tmux 세션)

SESSION_NAME="tdot_generation_1000"
LOG_DIR="/home/dhkang225/2D_sim/logs/trajectory"
LOG_FILE="${LOG_DIR}/tdot_generation_1000_$(date +%Y%m%d_%H%M%S).log"

# 로그 디렉토리 생성
mkdir -p ${LOG_DIR}

echo "🚀 Tdot 생성 시작 (1000개 환경)"
echo "================================"
echo "세션명: ${SESSION_NAME}"
echo "로그 파일: ${LOG_FILE}"

# 기존 세션 확인 및 종료
tmux has-session -t ${SESSION_NAME} 2>/dev/null
if [ $? == 0 ]; then
    echo "기존 세션 종료 중..."
    tmux kill-session -t ${SESSION_NAME}
    sleep 1
fi

# 새 tmux 세션 생성 및 실행
echo "tmux 세션 시작..."
tmux new-session -d -s ${SESSION_NAME} \
    "python /home/dhkang225/2D_sim/packages/data_generator/trajectory/generate_tdot_trajectories.py \
        --input circles_only_integrated_trajs.h5 \
        --dt 0.01 \
        --time-policy uniform \
        --save-format 4x4 \
        --chunk-size 20 2>&1 | tee ${LOG_FILE}"

echo ""
echo "✅ 백그라운드 실행 시작됨!"
echo ""
echo "📊 모니터링 명령어:"
echo "   tmux attach -t ${SESSION_NAME}  # 세션 연결"
echo "   tail -f ${LOG_FILE}             # 로그 확인"
echo "   tmux ls                         # 세션 목록"
echo ""
echo "⏱️ 예상 소요 시간: ~20-30분 (1000개 환경, 2000개 궤적)"
echo ""
echo "💡 팁: Ctrl+B, D로 세션에서 나가기 (세션은 계속 실행됨)"