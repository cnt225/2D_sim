#!/usr/bin/env python3
"""
정규화 유틸리티 클래스
학습 <-> 추론 양방향 정규화 지원
"""

import torch
import numpy as np
import json
from pathlib import Path

class TwistNormalizer:
    """Twist vector 정규화/역정규화 클래스"""
    
    def __init__(self, stats_path=None, stats_dict=None):
        """
        Args:
            stats_path: 정규화 통계 JSON 파일 경로
            stats_dict: 직접 전달된 통계 딕셔너리
        """
        self.stats = None
        
        if stats_path and Path(stats_path).exists():
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
            print(f"✅ 정규화 통계 로드: {stats_path}")
        elif stats_dict:
            self.stats = stats_dict
            print("✅ 정규화 통계 직접 로드")
        else:
            print("⚠️ 정규화 통계 없음 - 정규화 비활성화")
            self.stats = None
    
    def normalize_twist(self, twist):
        """
        Twist vector 정규화 (학습 시 사용)
        
        Args:
            twist: [B, 6] 또는 [6] torch.Tensor 또는 numpy.ndarray
            
        Returns:
            normalized_twist: 정규화된 twist
        """
        if self.stats is None:
            return twist
        
        # numpy/torch 처리
        is_torch = isinstance(twist, torch.Tensor)
        if is_torch:
            device = twist.device
            twist_np = twist.cpu().numpy()
        else:
            twist_np = twist
        
        # 통계 추출
        mean = np.array(self.stats['twist_normalization']['total']['mean'])
        std = np.array(self.stats['twist_normalization']['total']['std'])
        
        # 0으로 나누기 방지
        std = np.where(std < 1e-8, 1.0, std)
        
        # 정규화
        normalized = (twist_np - mean) / std
        
        # 결과 반환
        if is_torch:
            return torch.tensor(normalized, dtype=torch.float32, device=device)
        else:
            return normalized.astype(np.float32)
    
    def denormalize_twist(self, normalized_twist):
        """
        정규화된 twist vector 역정규화 (추론 시 사용)
        
        Args:
            normalized_twist: [B, 6] 또는 [6] 정규화된 twist
            
        Returns:
            twist: 원본 스케일의 twist
        """
        if self.stats is None:
            return normalized_twist
        
        # numpy/torch 처리
        is_torch = isinstance(normalized_twist, torch.Tensor)
        if is_torch:
            device = normalized_twist.device
            norm_np = normalized_twist.cpu().numpy()
        else:
            norm_np = normalized_twist
        
        # 통계 추출
        mean = np.array(self.stats['twist_normalization']['total']['mean'])
        std = np.array(self.stats['twist_normalization']['total']['std'])
        
        # 0으로 나누기 방지
        std = np.where(std < 1e-8, 1.0, std)
        
        # 역정규화
        denormalized = norm_np * std + mean
        
        # 결과 반환
        if is_torch:
            return torch.tensor(denormalized, dtype=torch.float32, device=device)
        else:
            return denormalized.astype(np.float32)
    
    def get_stats_summary(self):
        """정규화 통계 요약 출력"""
        if self.stats is None:
            return "정규화 통계 없음"
        
        angular_stats = self.stats['twist_normalization']['angular']
        linear_stats = self.stats['twist_normalization']['linear']
        
        summary = f"""
🔍 정규화 통계 요약:
   Angular velocity:
      평균 크기: {angular_stats['overall_mean']:.4f} ± {angular_stats['overall_std']:.4f} rad/s
   Linear velocity:
      평균 크기: {linear_stats['overall_mean']:.4f} ± {linear_stats['overall_std']:.4f} m/s
   
   정규화 방식: (twist - mean) / std
   """
        return summary

def create_normalization_stats():
    """정규화 통계 생성 (한 번만 실행)"""
    print("📊 정규화 통계 생성 중...")
    
    import sys
    sys.path.append('.')
    from loaders.trajectory_dataset import TrajectoryDataset
    
    # 데이터셋 로드
    trajectory_root = "../../../data/trajectories/circle_envs_10k_bsplined"
    pointcloud_root = "../../../data/pointcloud/circle_envs_10k/circle_envs_10k"
    
    dataset = TrajectoryDataset(
        trajectory_root=trajectory_root,
        pointcloud_root=pointcloud_root,
        split='train',
        max_trajectories=500,  # 통계용으로 충분
        use_bsplined=True,
        augment_data=False,
        num_points=300
    )
    
    # Twist vector 수집
    all_twists = []
    
    print(f"📈 {len(dataset)} 샘플에서 통계 수집 중...")
    sample_count = min(len(dataset), 5000)  # 5000개 샘플로 충분
    
    for i in range(sample_count):
        if i % 1000 == 0:
            print(f"   진행률: {i}/{sample_count}")
        
        try:
            sample = dataset[i]
            T_dot = sample['T_dot'].numpy()
            all_twists.append(T_dot)
        except Exception as e:
            continue
    
    all_twists = np.array(all_twists)  # [N, 6]
    
    # 통계 계산
    stats = {
        'twist_normalization': {
            'method': 'standardization',
            'angular': {
                'mean': np.mean(all_twists[:, :3], axis=0).tolist(),
                'std': np.std(all_twists[:, :3], axis=0).tolist(),
                'overall_mean': float(np.mean(np.linalg.norm(all_twists[:, :3], axis=1))),
                'overall_std': float(np.std(np.linalg.norm(all_twists[:, :3], axis=1)))
            },
            'linear': {
                'mean': np.mean(all_twists[:, 3:], axis=0).tolist(),
                'std': np.std(all_twists[:, 3:], axis=0).tolist(),
                'overall_mean': float(np.mean(np.linalg.norm(all_twists[:, 3:], axis=1))),
                'overall_std': float(np.std(np.linalg.norm(all_twists[:, 3:], axis=1)))
            },
            'total': {
                'mean': np.mean(all_twists, axis=0).tolist(),
                'std': np.std(all_twists, axis=0).tolist(),
                'overall_mean': float(np.mean(np.linalg.norm(all_twists, axis=1))),
                'overall_std': float(np.std(np.linalg.norm(all_twists, axis=1)))
            }
        },
        'data_info': {
            'samples_used': len(all_twists),
            'dataset_size': len(dataset)
        }
    }
    
    # 저장
    output_path = "configs/normalization_stats.json"
    Path("configs").mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ 정규화 통계 저장: {output_path}")
    print(f"📊 사용된 샘플: {len(all_twists)}")
    
    # 통계 출력
    normalizer = TwistNormalizer(stats_dict=stats)
    print(normalizer.get_stats_summary())
    
    return stats

if __name__ == "__main__":
    # 정규화 통계 생성
    stats = create_normalization_stats()
    
    # 테스트
    print("\n🧪 정규화 테스트")
    print("-" * 40)
    
    normalizer = TwistNormalizer(stats_dict=stats)
    
    # 테스트 twist
    test_twist = np.array([0.5, -0.2, 0.1, 3.2, 1.8, -0.5])
    print(f"원본 twist: {test_twist}")
    print(f"원본 크기: {np.linalg.norm(test_twist):.4f}")
    
    # 정규화
    normalized = normalizer.normalize_twist(test_twist)
    print(f"정규화된 twist: {normalized}")
    print(f"정규화된 크기: {np.linalg.norm(normalized):.4f}")
    
    # 역정규화
    denormalized = normalizer.denormalize_twist(normalized)
    print(f"역정규화된 twist: {denormalized}")
    print(f"역정규화된 크기: {np.linalg.norm(denormalized):.4f}")
    
    # 정확성 확인
    error = np.linalg.norm(test_twist - denormalized)
    print(f"복원 오차: {error:.8f}")
    
    if error < 1e-6:
        print("✅ 정규화/역정규화 정확!")
    else:
        print("❌ 정규화/역정규화 오차 발생!")

