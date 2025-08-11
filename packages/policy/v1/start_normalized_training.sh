#!/bin/bash

echo "🚀 정규화된 Motion RFM 학습 시작"
echo "================================"

# 설정 확인
echo "📋 학습 설정:"
echo "   - Config: motion_rcfm_normalized.yml"
echo "   - Epochs: 10"
echo "   - Save interval: 10 (매 10 에포크마다 저장)"
echo "   - Normalization: 활성화"
echo "   - Wandb project: motion_planning_rfm_normalized"
echo ""

# 이전 체크포인트 백업
if [ -d "checkpoints_old" ]; then
    echo "📦 기존 백업 폴더 정리..."
    rm -rf checkpoints_old
fi

if [ -d "checkpoints" ]; then
    echo "📦 기존 체크포인트 백업..."
    mv checkpoints checkpoints_old
fi

# 새 체크포인트 폴더 생성
mkdir -p checkpoints

# 정규화 통계 파일 확인
if [ ! -f "configs/normalization_stats.json" ]; then
    echo "❌ 정규화 통계 파일이 없습니다!"
    echo "   python utils/normalization.py 실행하여 생성하세요."
    exit 1
fi

echo "✅ 정규화 통계 파일 확인됨"

# 학습 시작
echo ""
echo "🎯 tmux 세션으로 학습 시작..."
echo "   세션명: training_normalized"
echo "   로그: training_normalized.log"
echo ""

# tmux 세션 생성 및 학습 실행
tmux new-session -d -s training_normalized "
cd /home/dhkang225/2D_sim/packages/policy/v1
echo '🚀 정규화된 학습 시작 - $(date)' | tee training_normalized.log
echo '설정 파일: configs/motion_rcfm_normalized.yml' | tee -a training_normalized.log
echo '' | tee -a training_normalized.log

python train.py --config configs/motion_rcfm_normalized.yml 2>&1 | tee -a training_normalized.log

echo '' | tee -a training_normalized.log
echo '학습 완료 - $(date)' | tee -a training_normalized.log
"

echo "✅ tmux 세션 'training_normalized' 시작됨"
echo ""
echo "📱 명령어 모음:"
echo "   tmux attach -t training_normalized    # 세션 접속"
echo "   tmux kill-session -t training_normalized  # 세션 종료"
echo "   tail -f training_normalized.log       # 실시간 로그 보기"
echo "   cat training_normalized.log           # 전체 로그 보기"
echo ""
echo "🔍 학습 진행 확인:"
echo "   ls -la checkpoints/                   # 체크포인트 파일들"
echo "   tail -20 training_normalized.log      # 최근 로그"
echo ""
echo "🌐 wandb: https://wandb.ai/ (project: motion_planning_rfm_normalized)"
