#!/usr/bin/env python3
"""
PLY Pointcloud Visualizer
기존 PLY 파일을 읽고 시각화하거나 이미지로 저장하는 도구

사용법:
    python visualize_pointcloud.py my_env.ply
    python visualize_pointcloud.py my_env --save-image
    python visualize_pointcloud.py --list
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

# 상위 디렉토리를 path에 추가하여 모듈 import 가능하도록 설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pointcloud import PointcloudLoader


class PointcloudVisualizer:
    """PLY 포인트클라우드 시각화 클래스"""
    
    def __init__(self, data_dir: str = "data/pointcloud"):
        self.data_dir = data_dir
        self.loader = PointcloudLoader(data_dir)
    
    def list_available_files(self) -> List[str]:
        """사용 가능한 PLY 파일 목록"""
        return self.loader.list_available_pointclouds()
    
    def load_and_visualize(self, filename: str, save_image: bool = False, 
                          show_plot: bool = True, output_name: str = None) -> Optional[str]:
        """
        PLY 파일을 로드하고 시각화
        
        Args:
            filename: PLY 파일명 (확장자 포함 또는 제외)
            save_image: 이미지로 저장할지 여부
            show_plot: 화면에 표시할지 여부
            output_name: 저장할 이미지 파일명 (None이면 자동 생성)
            
        Returns:
            저장된 이미지 파일 경로 (저장하지 않으면 None)
        """
        try:
            # PLY 파일 로드
            points, metadata = self.loader.load_pointcloud(filename)
            print(f"Loaded {len(points)} points from {filename}")
            
            if metadata:
                print("Metadata:")
                for key, value in metadata.items():
                    if key != 'environment_details':  # 너무 긴 내용은 제외
                        print(f"  {key}: {value}")
            
            # 시각화 생성 - 단일 그래프로 변경
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            # 포인트클라우드 시각화
            self._plot_pointcloud(ax, points, metadata, filename)
            
            plt.tight_layout()
            
            # 이미지 저장
            saved_path = None
            if save_image:
                if output_name is None:
                    # 파일명에서 폴더 구분자 제거하고 순수 파일명만 추출
                    clean_filename = filename.split('/')[-1] if '/' in filename else filename
                    base_name = clean_filename.replace('.ply', '') if clean_filename.endswith('.ply') else clean_filename
                    output_name = f"{base_name}_scene.jpg"
                
                # 환경별 폴더 구조 확인
                if '/' in filename:
                    # folder/filename 형태인 경우
                    folder_name = filename.split('/')[0]
                    image_path = os.path.join(self.data_dir, folder_name, output_name)
                else:
                    image_path = os.path.join(self.data_dir, output_name)
                
                plt.savefig(image_path, dpi=150, bbox_inches='tight', facecolor='white')
                saved_path = image_path
                print(f"Visualization saved: {image_path}")
            
            # 화면 표시
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return saved_path
            
        except Exception as e:
            print(f"Error visualizing {filename}: {e}")
            return None
    
    def _plot_pointcloud(self, ax, points: np.ndarray, metadata: Dict, filename: str):
        """포인트클라우드 플롯"""
        # 포인트 색상을 더 선명하게
        ax.scatter(points[:, 0], points[:, 1], s=2, alpha=0.8, c='darkblue', edgecolors='none')
        
        # 제목과 레이블
        ax.set_title(f'Pointcloud: {filename}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 로봇 베이스 표시
        ax.plot(0, 0, 'ro', markersize=10, label='Robot Base', zorder=5)
        
        # 작업공간 경계 설정
        if metadata and 'workspace_bounds' in metadata:
            min_x, max_x, min_y, max_y = metadata['workspace_bounds']
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
        else:
            # 포인트클라우드 범위에서 추정
            min_x, max_x = points[:, 0].min() - 0.5, points[:, 0].max() + 0.5
            min_y, max_y = points[:, 1].min() - 0.5, points[:, 1].max() + 0.5
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
        
        # 정보 텍스트 - 더 상세하게
        info_lines = [f'Total Points: {len(points):,}']
        if metadata:
            if 'resolution' in metadata:
                info_lines.append(f'Resolution: {metadata["resolution"]:.3f}m')
            if 'noise_level' in metadata:
                info_lines.append(f'Noise Level: {metadata["noise_level"]:.3f}')
            if 'env_type' in metadata:
                info_lines.append(f'Environment: {metadata["env_type"]}')
            if 'difficulty' in metadata:
                info_lines.append(f'Difficulty: {metadata["difficulty"]}')
            if 'num_obstacles' in metadata:
                info_lines.append(f'Obstacles: {metadata["num_obstacles"]}')
            if 'clustering_eps' in metadata:
                info_lines.append(f'Clustering ε: {metadata["clustering_eps"]}')
        
        info_text = '\n'.join(info_lines)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
                facecolor='lightblue', alpha=0.9, edgecolor='darkblue'))
        
        # 범례 추가
        ax.legend(loc='upper right', fontsize=10)
    
    def batch_visualize(self, filenames: List[str], save_images: bool = True) -> List[str]:
        """여러 PLY 파일을 일괄 시각화"""
        saved_paths = []
        
        for filename in filenames:
            print(f"\nProcessing {filename}...")
            try:
                saved_path = self.load_and_visualize(
                    filename, 
                    save_image=save_images, 
                    show_plot=False  # 일괄 처리시 화면 표시 안함
                )
                if saved_path:
                    saved_paths.append(saved_path)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
        
        return saved_paths


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize PLY pointcloud files')
    
    # 파일 관련
    parser.add_argument('filename', nargs='?', type=str,
                        help='PLY filename to visualize (with or without .ply extension)')
    parser.add_argument('--list', action='store_true',
                        help='List all available PLY files')
    parser.add_argument('--all', action='store_true',
                        help='Visualize all available PLY files')
    
    # 출력 관련
    parser.add_argument('--save-image', action='store_true',
                        help='Save visualization as JPG image')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image filename (auto-generated if not specified)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not show plot on screen (useful with --save-image)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    visualizer = PointcloudVisualizer()
    
    # 파일 목록 표시
    if args.list or (not args.filename and not args.all):
        available_files = visualizer.list_available_files()
        print("Available PLY files:")
        if available_files:
            for i, filename in enumerate(available_files, 1):
                print(f"  {i}. {filename}.ply")
        else:
            print("  (No PLY files found)")
        
        if not args.filename and not args.all:
            print("\nUsage:")
            print("  python visualize_pointcloud.py <filename>")
            print("  python visualize_pointcloud.py --all --save-image")
            return
    
    # 모든 파일 처리
    if args.all:
        available_files = visualizer.list_available_files()
        if not available_files:
            print("No PLY files found to visualize")
            return
        
        print(f"Visualizing {len(available_files)} files...")
        saved_paths = visualizer.batch_visualize(
            available_files, 
            save_images=args.save_image
        )
        
        if saved_paths:
            print(f"\nSaved {len(saved_paths)} visualizations:")
            for path in saved_paths:
                print(f"  {path}")
        
        return
    
    # 개별 파일 처리
    if args.filename:
        # 파일 존재 확인
        if not args.filename.endswith('.ply'):
            check_filename = args.filename + '.ply'
        else:
            check_filename = args.filename
        
        # 폴더 구조 확인
        if '/' in args.filename:
            file_path = os.path.join("pointcloud", "data", check_filename)
        else:
            file_path = os.path.join("pointcloud", "data", check_filename)
        
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            available_files = visualizer.list_available_files()
            if available_files:
                print("Available files:")
                for filename in available_files:
                    print(f"  {filename}.ply")
            return
        
        # 시각화 실행
        saved_path = visualizer.load_and_visualize(
            args.filename,
            save_image=args.save_image,
            show_plot=not args.no_show,
            output_name=args.output
        )
        
        if saved_path:
            print(f"Visualization complete. Image saved: {saved_path}")
        else:
            print("Visualization complete.")


if __name__ == "__main__":
    main() 