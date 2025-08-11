#!/bin/bash
# 1시간마다 학습 진행 상황을 모니터링하는 스크립트

LOG_FILE="training_production.log"
MONITOR_LOG="hourly_monitor.log"

echo "=== 학습 모니터링 시작: $(date) ===" >> $MONITOR_LOG

while true; do
    echo "" >> $MONITOR_LOG
    echo "--- $(date) ---" >> $MONITOR_LOG
    
    # 프로세스 상태 확인
    PROCESS=$(ps aux | grep "python.*train.py" | grep -v grep)
    if [ -z "$PROCESS" ]; then
        echo "❌ 학습 프로세스가 중단되었습니다!" >> $MONITOR_LOG
        echo "❌ 학습 프로세스가 중단되었습니다!"
        break
    else
        echo "✅ 학습 진행 중: $(echo $PROCESS | awk '{print $2, $3"%", $4"MB"}')" >> $MONITOR_LOG
    fi
    
    # GPU 사용량 확인
    GPU_INFO=$(nvidia-smi | grep python | head -1)
    if [ ! -z "$GPU_INFO" ]; then
        echo "🎮 GPU: $(echo $GPU_INFO | awk '{print $6}')" >> $MONITOR_LOG
    fi
    
    # 최근 Loss 확인 (마지막 5줄에서 Epoch 패턴 찾기)
    RECENT_LOSS=$(tail -50 $LOG_FILE | grep -E "Epoch.*Loss" | tail -1)
    if [ ! -z "$RECENT_LOSS" ]; then
        echo "📊 $RECENT_LOSS" >> $MONITOR_LOG
    fi
    
    # 로그 파일 크기
    LOG_SIZE=$(du -h $LOG_FILE 2>/dev/null | cut -f1)
    echo "📝 로그 크기: $LOG_SIZE" >> $MONITOR_LOG
    
    # 1시간 대기 (3600초)
    sleep 3600
done

echo "=== 모니터링 종료: $(date) ===" >> $MONITOR_LOG

