#!/usr/bin/env python3
"""
정규화된 Motion RFM 추론 엔진
완전한 정규화 양방향 파이프라인 적용
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import time

from models import get_model
from utils.Lie import *
from omegaconf import OmegaConf
from utils.normalization import TwistNormalizer

class NormalizedMotionRFMInference:
    """정규화된 Motion RFM 추론 엔진"""
    
    def __init__(self, model_path: str, config_path: str, 
                 normalize_twist: bool = True,
                 normalization_stats_path: str = "configs/normalization_stats.json"):
        """추론 엔진 초기화"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 설정 로드
        self.config = OmegaConf.load(config_path)
        
        # 정규화 설정
        self.normalize_twist = normalize_twist
        if normalize_twist:
            self.twist_normalizer = TwistNormalizer(stats_path=normalization_stats_path)
            print(f"✅ Twist 역정규화 활성화: {normalization_stats_path}")
            print(self.twist_normalizer.get_stats_summary())
        else:
            self.twist_normalizer = None
            print("⚠️ Twist 역정규화 비활성화")
        
        # 모델 로드
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"✅ Normalized Motion RFM Inference Engine loaded on {self.device}")
        print(f"📊 Model config: {self.config.model.arch}")
    
    def _load_model(self, model_path: str):
        """모델 로드"""
        model = get_model(self.config.model)
        
        if Path(model_path).exists():
            # PyTorch 2.6 호환성을 위해 weights_only=False 사용
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state'])
            print(f"✅ Model loaded from {model_path}")
        else:
            print(f"⚠️ Checkpoint not found: {model_path}, using random weights")
        
        return model.to(self.device)
    
    def generate_trajectory(self, 
                          start_pose: Union[torch.Tensor, np.ndarray],
                          target_pose: Union[torch.Tensor, np.ndarray], 
                          pointcloud: Union[torch.Tensor, np.ndarray],
                          config: Optional[Dict] = None) -> Dict:
        """궤적 생성 (정규화 적용)"""
        
        # 기본 설정
        cfg = {
            'dt': 0.05,                    # 적분 타임스텝 (개선됨)
            'max_steps': 200,              # 최대 반복 수
            'pos_tolerance': 0.05,         # 위치 허용 오차 (m)
            'rot_tolerance': 0.1,          # 회전 허용 오차 (rad)
            'early_stop': True,            # 조기 종료 사용
            'safety_check': True,          # 안전 체크 사용
            'safety_distance_threshold': 10.0,  # 안전 거리 임계값
        }
        
        if config:
            cfg.update(config)
        
        # 입력 텐서 변환 및 디바이스 이동
        if isinstance(start_pose, np.ndarray):
            start_pose = torch.tensor(start_pose, dtype=torch.float32).to(self.device)
        else:
            start_pose = start_pose.to(self.device)
            
        if isinstance(target_pose, np.ndarray):
            target_pose = torch.tensor(target_pose, dtype=torch.float32).to(self.device)
        else:
            target_pose = target_pose.to(self.device)
            
        if isinstance(pointcloud, np.ndarray):
            pointcloud = torch.tensor(pointcloud, dtype=torch.float32).to(self.device)
        else:
            pointcloud = pointcloud.to(self.device)
        
        # 초기화
        current_pose = start_pose.clone()
        trajectory = [current_pose.clone().cpu()]
        step = 0
        initial_distance = self._pose_distance(start_pose, target_pose).item()
        start_time = time.time()
        
        print(f"🚀 궤적 생성 시작 (정규화 모드: {'ON' if self.normalize_twist else 'OFF'})")
        print(f"   시작 → 목표 거리: {initial_distance:.3f}m")
        print(f"   설정: dt={cfg['dt']}, max_steps={cfg['max_steps']}")
        
        with torch.no_grad():
            while step < cfg['max_steps']:
                # 1. Progress 계산
                progress = self._calculate_progress(current_pose, start_pose, target_pose)
                
                # 2. 모델 추론: 6D twist vector 예측 (정규화 적용)
                twist_6d = self._predict_twist(current_pose, target_pose, progress, pointcloud)
                
                # 3. 목표 도달 확인 (조기 종료)
                if cfg['early_stop'] and self._reached_target(current_pose, target_pose, cfg):
                    print(f"✅ 목표 도달! (스텝: {step})")
                    break
                
                # 4. 안전 체크
                if cfg['safety_check'] and self._safety_check(current_pose, target_pose, 
                                                            initial_distance, cfg):
                    print(f"⚠️ Safety check failed at step {step}")
                    break
                
                # 5. SE(3) 적분: twist → next pose
                current_pose = self._integrate_se3(current_pose, twist_6d, cfg['dt'])
                
                # 6. 궤적에 추가
                trajectory.append(current_pose.clone().cpu())
                step += 1
                
                # 진행 상황 출력
                if step % 50 == 0:
                    current_dist = self._pose_distance(current_pose, target_pose).item()
                    twist_norm = torch.norm(twist_6d).item()
                    print(f"   스텝 {step}: 거리 {current_dist:.3f}m, twist크기 {twist_norm:.4f}")
        
        generation_time = time.time() - start_time
        final_error = self._compute_final_error(current_pose, target_pose)
        success = self._reached_target(current_pose, target_pose, cfg)
        
        result = {
            'trajectory': trajectory,
            'steps': step,
            'success': success,
            'generation_time': generation_time,
            'final_error': final_error,
            'config_used': cfg,
            'normalization_used': self.normalize_twist,
            'stats': {
                'initial_distance': initial_distance,
                'final_distance': self._pose_distance(current_pose, target_pose).item(),
                'avg_step_time': generation_time / max(step, 1),
            }
        }
        
        print(f"🎯 생성 완료: {'성공' if success else '실패'}, {step}스텝, {generation_time:.3f}초")
        return result
    
    def _predict_twist(self, current_pose: torch.Tensor, target_pose: torch.Tensor,
                      progress: torch.Tensor, pointcloud: torch.Tensor) -> torch.Tensor:
        """모델을 사용하여 6D twist vector 예측 (정규화 적용)"""
        # 입력 준비 (배치 차원 추가)
        current_T = current_pose.unsqueeze(0)  # (1, 4, 4)
        target_T = target_pose.unsqueeze(0)    # (1, 4, 4)
        time_t = progress.unsqueeze(0).unsqueeze(0)  # (1, 1)
        pc = pointcloud.unsqueeze(0)           # (1, N, 3)
        
        # 포인트클라우드 특징 추출 (DGCNN 입력 형식: [B, 3, N])
        pc_dgcnn = pc.transpose(1, 2)  # (1, 3, N)
        v = self.model.latent_feature(pc_dgcnn)  # (1, latent_dim)
        
        # 모델의 velocity field 직접 호출 (정규화된 출력)
        normalized_twist = self.model.velocity_field(current_T, target_T, time_t, v)  # (1, 6)
        normalized_twist = normalized_twist.squeeze(0)  # (6,)
        
        # 역정규화 (정규화된 출력 → 실제 twist)
        if self.twist_normalizer is not None:
            real_twist = self.twist_normalizer.denormalize_twist(normalized_twist)
            return real_twist
        else:
            return normalized_twist
    
    def _calculate_progress(self, current: torch.Tensor, start: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """진행도 계산 (거리 기반)"""
        total_distance = self._pose_distance(start, target)
        current_distance = self._pose_distance(current, target)
        
        if total_distance < 1e-6:
            return torch.tensor(1.0, device=self.device)
        
        progress = 1.0 - (current_distance / total_distance)
        return torch.clamp(progress, 0.0, 1.0)
    
    def _pose_distance(self, pose1: torch.Tensor, pose2: torch.Tensor) -> torch.Tensor:
        """두 SE(3) pose 간의 거리"""
        # 위치 거리
        pos_dist = torch.norm(pose1[:3, 3] - pose2[:3, 3])
        
        # 회전 거리 (Frobenius norm of rotation difference)
        rot_diff = pose1[:3, :3] - pose2[:3, :3]
        rot_dist = torch.norm(rot_diff, 'fro')
        
        # 가중 합 (위치와 회전 스케일 조정)
        return pos_dist + 0.1 * rot_dist
    
    def _reached_target(self, current: torch.Tensor, target: torch.Tensor, config: Dict) -> bool:
        """목표 도달 여부 판별"""
        pos_error = torch.norm(current[:3, 3] - target[:3, 3])
        rot_error = self._rotation_angle_between(current[:3, :3], target[:3, :3])
        
        return (pos_error < config['pos_tolerance']) and (rot_error < config['rot_tolerance'])
    
    def _rotation_angle_between(self, R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
        """두 회전 행렬 간의 각도 차이"""
        R_rel = R1.T @ R2
        trace = torch.trace(R_rel)
        cos_angle = (trace - 1) / 2
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        return torch.acos(cos_angle)
    
    def _safety_check(self, current: torch.Tensor, target: torch.Tensor, 
                     initial_distance: float, config: Dict) -> bool:
        """안전 체크 (발산 방지)"""
        current_distance = self._pose_distance(current, target).item()
        
        # 초기 거리의 N배 이상 멀어지면 실패
        if current_distance > config['safety_distance_threshold'] * initial_distance:
            return True
        
        return False
    
    def _integrate_se3(self, pose: torch.Tensor, twist: torch.Tensor, dt: float) -> torch.Tensor:
        """SE(3) 적분 (Euler method)"""
        # Twist vector: [ωx, ωy, ωz, vx, vy, vz] (body frame)
        omega = twist[:3] * dt  # 각속도 * dt
        v = twist[3:] * dt      # 선속도 * dt
        
        # Current pose
        R = pose[:3, :3]
        p = pose[:3, 3]
        
        # 회전 업데이트 (rodrigues formula)
        omega_norm = torch.norm(omega)
        if omega_norm > 1e-6:
            omega_hat = omega / omega_norm
            omega_skew = torch.tensor([
                [0, -omega_hat[2], omega_hat[1]],
                [omega_hat[2], 0, -omega_hat[0]],
                [-omega_hat[1], omega_hat[0], 0]
            ], device=pose.device)
            
            R_new = R @ (torch.eye(3, device=pose.device) + 
                        torch.sin(omega_norm) * omega_skew + 
                        (1 - torch.cos(omega_norm)) * (omega_skew @ omega_skew))
        else:
            R_new = R
        
        # 위치 업데이트 (body frame velocity를 world frame으로 변환)
        p_new = p + R @ v
        
        # 새로운 SE(3) pose
        new_pose = torch.eye(4, device=pose.device)
        new_pose[:3, :3] = R_new
        new_pose[:3, 3] = p_new
        
        return new_pose
    
    def _compute_final_error(self, current: torch.Tensor, target: torch.Tensor) -> Dict:
        """최종 오차 계산"""
        pos_error = torch.norm(current[:3, 3] - target[:3, 3])
        rot_error = self._rotation_angle_between(current[:3, :3], target[:3, :3])
        
        return {
            'position_error_m': pos_error.item(),
            'rotation_error_rad': rot_error.item(),
            'rotation_error_deg': torch.rad2deg(rot_error).item(),
        }

# 편의 함수들
class NormalizedInferenceConfigs:
    """정규화된 추론 설정 프리셋"""
    
    @staticmethod
    def default():
        return {
            'dt': 0.05,
            'max_steps': 200,
            'pos_tolerance': 0.05,
            'rot_tolerance': 0.1,
            'early_stop': True,
            'safety_check': True,
            'safety_distance_threshold': 5.0,
        }
    
    @staticmethod
    def fast():
        return {
            'dt': 0.1,
            'max_steps': 100,
            'pos_tolerance': 0.1,
            'rot_tolerance': 0.2,
            'early_stop': True,
            'safety_check': True,
            'safety_distance_threshold': 3.0,
        }
    
    @staticmethod
    def precise():
        return {
            'dt': 0.02,
            'max_steps': 500,
            'pos_tolerance': 0.01,
            'rot_tolerance': 0.05,
            'early_stop': True,
            'safety_check': True,
            'safety_distance_threshold': 10.0,
        }

if __name__ == "__main__":
    print("🧪 정규화된 추론 엔진 테스트")
    
    # 테스트 실행
    engine = NormalizedMotionRFMInference(
        'checkpoints/motion_rcfm_final_epoch10.pth',
        'configs/motion_rcfm.yml',
        normalize_twist=True
    )
    
    # 더미 데이터로 테스트
    start_pose = torch.eye(4, dtype=torch.float32)
    target_pose = torch.eye(4, dtype=torch.float32)
    target_pose[:3, 3] = torch.tensor([2.0, 2.0, 0.0])
    pointcloud = torch.randn(300, 3)
    
    result = engine.generate_trajectory(
        start_pose, target_pose, pointcloud, 
        NormalizedInferenceConfigs.default()
    )
    
    print(f"✅ 테스트 완료: {result['steps']}스텝, {'성공' if result['success'] else '실패'}")

