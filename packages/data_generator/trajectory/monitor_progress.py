#!/usr/bin/env python3
"""
Trajectory Generation Progress Monitor
대량 궤적 생성 진행 상황 모니터링 스크립트
"""

import time
import subprocess
from pathlib import Path

def get_tmux_status():
    """tmux 세션 상태 확인"""
    try:
        result = subprocess.run(['tmux', 'list-sessions'], 
                              capture_output=True, text=True)
        if 'trajectory_gen_1000' in result.stdout:
            return "🟢 Running"
        else:
            return "🔴 Stopped"
    except:
        return "❓ Unknown"

def get_log_progress():
    """로그 파일에서 진행 상황 파싱"""
    log_file = Path("trajectory_gen_1000.log")
    if not log_file.exists():
        return 0, 0, "로그 파일 없음"
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # 처리된 환경 수
        env_count = content.count("환경 처리 중:")
        
        # 성공한 궤적 수
        success_count = content.count("✅ RRT 성공:")
        
        # 최근 상태
        lines = content.strip().split('\n')
        recent_status = lines[-1] if lines else "상태 없음"
        
        return env_count, success_count, recent_status
        
    except Exception as e:
        return 0, 0, f"오류: {e}"

def main():
    """메인 모니터링 함수"""
    print("🚀 Trajectory Generation Progress Monitor")
    print("=" * 50)
    
    try:
        while True:
            # 상태 수집
            tmux_status = get_tmux_status()
            env_count, success_count, recent_status = get_log_progress()
            
            # 진행률 계산
            target_envs = 1000
            target_pairs = 2000  # 1000 envs × 2 pairs
            progress_env = (env_count / target_envs) * 100 if target_envs > 0 else 0
            progress_pair = (success_count / target_pairs) * 100 if target_pairs > 0 else 0
            
            # 출력
            print(f"\r🔄 {time.strftime('%H:%M:%S')} | "
                  f"tmux: {tmux_status} | "
                  f"환경: {env_count}/{target_envs} ({progress_env:.1f}%) | "
                  f"궤적: {success_count}/{target_pairs} ({progress_pair:.1f}%)", end="")
            
            # 완료 체크
            if env_count >= target_envs or "배치 생성 완료" in recent_status:
                print(f"\n✅ 생성 완료! 총 {env_count}개 환경, {success_count}개 궤적")
                break
            
            # tmux가 중지되었으면 종료
            if "Stopped" in tmux_status:
                print(f"\n⚠️ tmux 세션이 중지되었습니다.")
                print(f"마지막 상태: {recent_status}")
                break
            
            time.sleep(5)  # 5초마다 업데이트
            
    except KeyboardInterrupt:
        print(f"\n🛑 모니터링 중단")
        env_count, success_count, _ = get_log_progress()
        print(f"현재 진행: {env_count}개 환경, {success_count}개 궤적")

if __name__ == "__main__":
    main()
