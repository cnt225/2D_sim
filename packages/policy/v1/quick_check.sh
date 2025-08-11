#!/bin/bash
# 빠른 학습 상태 확인 스크립트

echo "=== 학습 상태 확인: $(date) ==="

# 프로세스 확인
PROCESS=$(ps aux | grep "python.*train.py" | grep -v grep)
if [ -z "$PROCESS" ]; then
    echo "❌ 학습 프로세스가 중단되었습니다!"
    exit 1
else
    PID=$(echo $PROCESS | awk '{print $2}')
    CPU=$(echo $PROCESS | awk '{print $3}')
    MEM=$(echo $PROCESS | awk '{print $4}')
    TIME=$(echo $PROCESS | awk '{print $10}')
    echo "✅ 학습 진행 중 (PID: $PID, CPU: $CPU%, MEM: $MEM%, 시간: $TIME)"
fi

# GPU 확인
GPU_INFO=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | head -1)
echo "🎮 GPU: $GPU_INFO (사용/전체 MB, 사용률%)"

# 최근 Loss 
echo "📊 최근 진행:"
tail -50 training_production.log | grep -E "Epoch.*Loss" | tail -3

# 로그 크기와 라인 수
LOG_SIZE=$(du -h training_production.log 2>/dev/null | cut -f1)
LOG_LINES=$(wc -l training_production.log 2>/dev/null | cut -d' ' -f1)
echo "📝 로그: $LOG_SIZE ($LOG_LINES 라인)"

# wandb 링크
echo "🔗 wandb: https://wandb.ai/cnt225-seoul-national-university/motion_planning_rfm"

