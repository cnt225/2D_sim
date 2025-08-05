#!/usr/bin/env python3
"""
Quick Circle Environment Visualizer
circle_envs_10k 디렉토리의 특정 환경을 빠르게 시각화

사용법:
    python quick_visualize.py 65        # 65번째 환경 시각화
    python quick_visualize.py 123       # 123번째 환경 시각화
    python quick_visualize.py 65 --save # 65번째 환경 시각화 + 이미지 저장
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json


def read_ply(filename):
    """PLY 파일에서 포인트 데이터 읽기"""
    points = []
    with open(filename, 'r') as f:
        # 헤더 스킵
        line = f.readline()
        while not line.startswith('end_header'):
            line = f.readline()
        
        # 포인트 데이터 읽기
        for line in f:
            if line.strip():
                x, y, z = map(float, line.strip().split())
                points.append([x, y])
    
    return np.array(points)


def read_metadata(filename):
    """메타데이터 JSON 파일 읽기"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except:
        return {}


def visualize_environment(env_number, save_image=False, data_dir="data/pointcloud/circle_envs_10k"):
    """특정 환경 번호의 포인트클라우드 시각화"""
    
    # 파일 경로 생성
    env_id = f"{env_number:06d}"  # 000065 형태로 포맷
    ply_file = f"{data_dir}/circle_env_{env_id}.ply"
    meta_file = f"{data_dir}/circle_env_{env_id}_meta.json"
    
    # 파일 존재 확인
    if not os.path.exists(ply_file):
        print(f"Error: PLY 파일을 찾을 수 없습니다: {ply_file}")
        return
    
    # PLY 파일 로드
    print(f"Loading environment #{env_number}...")
    points = read_ply(ply_file)
    
    # 메타데이터 로드
    metadata = read_metadata(meta_file)
    
    # 환경 정보 출력
    print(f"Environment #{env_number} 정보:")
    print(f"  - 포인트 수: {len(points):,}개")
    if metadata:
        print(f"  - 난이도: {metadata.get('difficulty', 'N/A')}")
        if 'environment_details' in metadata:
            env_details = metadata['environment_details']
            print(f"  - 장애물 수: {env_details.get('num_obstacles', 'N/A')}개")
            if 'config' in env_details:
                config = env_details['config']
                print(f"  - 분포 타입: {config.get('spatial_distribution', 'N/A')}")
                print(f"  - 밀도: {config.get('density_level', 'N/A')}")
                print(f"  - 복잡도: {config.get('complexity_level', 'N/A')}")
    
    # 시각화
    plt.figure(figsize=(12, 9))
    plt.scatter(points[:, 0], points[:, 1], s=0.5, c='blue', alpha=0.7)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # 제목 설정
    title = f'Circle Environment #{env_number}'
    if metadata:
        difficulty = metadata.get('difficulty', 'N/A')
        num_obstacles = metadata.get('environment_details', {}).get('num_obstacles', 'N/A')
        title += f' ({difficulty} - {num_obstacles} obstacles, {len(points):,} points)'
    
    plt.title(title, fontsize=14)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    
    # 저장 또는 표시
    if save_image:
        output_file = f'circle_env_{env_id}_visualization.jpg'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"이미지 저장됨: {output_file}")
    else:
        plt.show()


def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("사용법: python quick_visualize.py <환경번호> [--save]")
        print("예시:")
        print("  python quick_visualize.py 65        # 65번째 환경 화면에 표시")
        print("  python quick_visualize.py 123 --save # 123번째 환경 이미지로 저장")
        return
    
    try:
        env_number = int(sys.argv[1])
    except ValueError:
        print("Error: 환경 번호는 숫자여야 합니다.")
        return
    
    save_image = '--save' in sys.argv
    
    visualize_environment(env_number, save_image)


if __name__ == "__main__":
    main() 