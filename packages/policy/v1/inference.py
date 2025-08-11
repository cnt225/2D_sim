#!/usr/bin/env python3
"""
Motion RFM 추론 엔진
SE(3) Rigid Body Motion Planning을 위한 Riemannian Flow Matching 기반 궤적 생성기
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import time

from models import get_model
from utils.Lie import *
# from utils.pointcloud import load_pointcloud
from omegaconf import OmegaConf
from utils.normalization import TwistNormalizer


class MotionRFMInference:
    """Motion RFM 추론 엔진"""
    
    def __init__(self, model_path: str, config_path: str, device: str = 'cuda'):
        """
        Args:
            model_path: 학습된 모델 체크포인트 경로
            config_path: 모델 설정 파일 경로  
            device: 연산 디바이스 ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        self.config = OmegaConf.load(config_path)
        
        # 모델 로드
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 기본 설정
        self.default_config = {
            'dt': 0.02,                    # 적분 타임스텝
            'max_steps': 100,              # 최대 반복 수
            'pos_tolerance': 0.02,         # 위치 허용 오차 (m)
            'rot_tolerance': 0.1,          # 회전 허용 오차 (rad)
            'early_stop': True,            # 조기 종료 활성화
            'safety_check': True,          # 안전 체크 활성화
            'max_divergence': 2.0,         # 발산 감지 배수
        }
        
        print(f"✅ Motion RFM Inference Engine loaded on {device}")
        print(f"📊 Model config: {self.config.model.arch}")
    
    def _load_model(self, model_path: str):
        """모델 로드"""
        model = get_model(self.config.model)
        
        if Path(model_path).exists():
            # PyTorch 2.6 호환성을 위해 weights_only=False 사용
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded from {model_path}")
        else:
            print(f"⚠️ Checkpoint not found: {model_path}, using random weights")
        
        return model.to(self.device)
    
    def generate_trajectory(self, 
                          start_pose: Union[torch.Tensor, np.ndarray],
                          target_pose: Union[torch.Tensor, np.ndarray], 
                          pointcloud: Union[torch.Tensor, np.ndarray],
                          config: Optional[Dict] = None) -> Dict:
        """
        궤적 생성 메인 함수
        
        Args:
            start_pose: 시작 SE(3) 포즈 (4x4 matrix)
            target_pose: 목표 SE(3) 포즈 (4x4 matrix)
            pointcloud: 환경 포인트클라우드 (Nx3)
            config: 생성 설정 (None이면 기본값 사용)
            
        Returns:
            Dict containing:
                - trajectory: List[torch.Tensor] - SE(3) 포즈 시퀀스
                - success: bool - 목표 도달 성공 여부
                - steps: int - 사용된 스텝 수
                - final_error: Dict - 최종 오차 정보
                - generation_time: float - 생성 시간 (초)
                - info: Dict - 추가 디버깅 정보
        """
        start_time = time.time()
        
        # 설정 병합
        cfg = {**self.default_config, **(config or {})}
        
        # 입력 전처리
        start_pose = self._to_tensor(start_pose)
        target_pose = self._to_tensor(target_pose)
        pointcloud = self._to_tensor(pointcloud)
        
        # 초기화
        current_pose = start_pose.clone()
        trajectory = [current_pose.clone().cpu()]
        step = 0
        
        # 초기 거리 계산 (발산 감지용)
        initial_distance = self._pose_distance(start_pose, target_pose)
        
        # 메인 생성 루프
        with torch.no_grad():
            while step < cfg['max_steps']:
                # 1. Progress 계산
                progress = self._calculate_progress(current_pose, start_pose, target_pose)
                
                # 2. 모델 추론: 6D twist vector 예측
                twist_6d = self._predict_twist(current_pose, target_pose, progress, pointcloud)
                
                # 3. 목표 도달 확인 (조기 종료)
                if cfg['early_stop'] and self._reached_target(current_pose, target_pose, cfg):
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
        
        # 결과 정리
        generation_time = time.time() - start_time
        success = self._reached_target(current_pose, target_pose, cfg)
        final_error = self._calculate_final_error(current_pose, target_pose)
        
        return {
            'trajectory': trajectory,
            'success': success,
            'steps': step,
            'final_error': final_error,
            'generation_time': generation_time,
            'info': {
                'config': cfg,
                'initial_distance': initial_distance.item(),
                'final_distance': self._pose_distance(current_pose, target_pose).item(),
                'avg_step_time': generation_time / max(step, 1),
            }
        }
    
    def _predict_twist(self, current_pose: torch.Tensor, target_pose: torch.Tensor,
                      progress: torch.Tensor, pointcloud: torch.Tensor) -> torch.Tensor:
        """모델을 사용하여 6D twist vector 예측"""
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
    
    def _se3_to_12d(self, se3_matrix: torch.Tensor) -> torch.Tensor:
        """SE(3) 4x4 matrix → 12D vector 변환"""
        batch_size = se3_matrix.shape[0]
        result = torch.zeros(batch_size, 12, device=se3_matrix.device)
        
        # Rotation matrix (3x3) → 9D vector
        result[:, :9] = se3_matrix[:, :3, :3].reshape(batch_size, 9)
        
        # Translation vector (3D)
        result[:, 9:12] = se3_matrix[:, :3, 3]
        
        return result
    
    def _calculate_progress(self, current: torch.Tensor, start: torch.Tensor, 
                          target: torch.Tensor) -> torch.Tensor:
        """거리 기반 progress 계산"""
        total_dist = self._pose_distance(start, target)
        current_dist = self._pose_distance(current, target)
        
        # progress = 1 - (current_distance / total_distance)
        progress = 1.0 - (current_dist / (total_dist + 1e-8))
        return torch.clamp(progress, 0.0, 1.0)
    
    def _pose_distance(self, pose1: torch.Tensor, pose2: torch.Tensor) -> torch.Tensor:
        """SE(3) 포즈 간 거리 계산"""
        # 위치 거리
        pos_dist = torch.norm(pose1[:3, 3] - pose2[:3, 3])
        
        # 회전 거리 (Frobenius norm of rotation difference)
        rot_diff = pose1[:3, :3] - pose2[:3, :3]
        rot_dist = torch.norm(rot_diff, 'fro')
        
        # 가중 합 (위치와 회전 스케일 조정)
        return pos_dist + 0.1 * rot_dist
    
    def _reached_target(self, current: torch.Tensor, target: torch.Tensor, 
                       config: Dict) -> bool:
        """목표 도달 여부 판별"""
        pos_error = torch.norm(current[:3, 3] - target[:3, 3])
        rot_error = self._rotation_angle_between(current[:3, :3], target[:3, :3])
        
        return (pos_error < config['pos_tolerance']) and (rot_error < config['rot_tolerance'])
    
    def _rotation_angle_between(self, R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
        """두 회전 행렬 간의 각도 차이 계산"""
        # R_diff = R2^T @ R1
        R_diff = R2.T @ R1
        
        # trace를 이용한 각도 계산
        trace = torch.trace(R_diff)
        angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
        
        return angle
    
    def _safety_check(self, current: torch.Tensor, target: torch.Tensor,
                     initial_distance: torch.Tensor, config: Dict) -> bool:
        """안전 체크 (발산 감지)"""
        current_distance = self._pose_distance(current, target)
        
        # 발산 감지: 초기 거리의 배수 이상 멀어지면 위험
        is_diverging = current_distance > initial_distance * config['max_divergence']
        
        return is_diverging
    
    def _integrate_se3(self, pose: torch.Tensor, twist_6d: torch.Tensor, 
                      dt: float) -> torch.Tensor:
        """SE(3) exponential map을 이용한 적분"""
        # twist_6d = [w_x, w_y, w_z, v_x, v_y, v_z] (body frame)
        w = twist_6d[:3] * dt  # angular displacement
        v = twist_6d[3:] * dt  # linear displacement
        
        # Skew-symmetric matrix for rotation
        w_skew = self._skew_symmetric(w)
        
        # Matrix exponential for rotation
        R_delta = torch.matrix_exp(w_skew)
        
        # 새로운 pose 계산
        new_pose = pose.clone()
        new_pose[:3, :3] = pose[:3, :3] @ R_delta  # rotation 업데이트
        new_pose[:3, 3] = pose[:3, 3] + pose[:3, :3] @ v  # position 업데이트 (body frame)
        
        return new_pose
    
    def _skew_symmetric(self, w: torch.Tensor) -> torch.Tensor:
        """3D 벡터를 skew-symmetric matrix로 변환"""
        return torch.tensor([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ], device=w.device, dtype=w.dtype)
    
    def _calculate_final_error(self, current: torch.Tensor, target: torch.Tensor) -> Dict:
        """최종 오차 계산"""
        pos_error = torch.norm(current[:3, 3] - target[:3, 3])
        rot_error = self._rotation_angle_between(current[:3, :3], target[:3, :3])
        
        return {
            'position_error_m': pos_error.item(),
            'rotation_error_rad': rot_error.item(),
            'rotation_error_deg': torch.rad2deg(rot_error).item(),
            'total_distance': self._pose_distance(current, target).item(),
        }
    
    def _to_tensor(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """numpy array를 torch tensor로 변환"""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        return data.to(self.device)
    
    def generate_multiple_trajectories(self, 
                                     starts: List[Union[torch.Tensor, np.ndarray]],
                                     targets: List[Union[torch.Tensor, np.ndarray]],
                                     pointclouds: List[Union[torch.Tensor, np.ndarray]],
                                     config: Optional[Dict] = None) -> List[Dict]:
        """여러 궤적을 배치로 생성"""
        results = []
        for start, target, pc in zip(starts, targets, pointclouds):
            result = self.generate_trajectory(start, target, pc, config)
            results.append(result)
        return results


# 설정 프리셋들
class InferenceConfigs:
    """추론 설정 프리셋"""
    
    @staticmethod
    def default():
        """기본 설정 (균형형)"""
        return {
            'dt': 0.02,
            'max_steps': 100,
            'pos_tolerance': 0.02,
            'rot_tolerance': 0.1,
            'early_stop': True,
            'safety_check': True,
            'max_divergence': 2.0,
        }
    
    @staticmethod
    def high_quality():
        """고품질 설정 (정밀형)"""
        return {
            'dt': 0.005,
            'max_steps': 400,
            'pos_tolerance': 0.005,
            'rot_tolerance': 0.05,
            'early_stop': True,
            'safety_check': True,
            'max_divergence': 2.0,
        }
    
    @staticmethod
    def fast():
        """고속 생성 설정 (효율형)"""
        return {
            'dt': 0.05,
            'max_steps': 50,
            'pos_tolerance': 0.05,
            'rot_tolerance': 0.2,
            'early_stop': True,
            'safety_check': False,
            'max_divergence': 3.0,
        }


def load_inference_engine(checkpoint_path: str, config_path: str = None, 
                         device: str = 'cuda') -> MotionRFMInference:
    """추론 엔진 로드 헬퍼 함수"""
    if config_path is None:
        config_path = Path(checkpoint_path).parent / 'motion_rcfm.yml'
    
    return MotionRFMInference(checkpoint_path, config_path, device)


# 사용 예시
if __name__ == "__main__":
    # 예시 사용법
    print("🚀 Motion RFM Inference Example")
    
    # 추론 엔진 초기화
    engine = MotionRFMInference(
        model_path="checkpoints/best_model.pth",
        config_path="configs/motion_rcfm.yml"
    )
    
    # 더미 데이터 생성
    start_pose = torch.eye(4)
    target_pose = torch.eye(4)
    target_pose[:3, 3] = torch.tensor([1.0, 1.0, 0.0])  # 1m, 1m 이동
    pointcloud = torch.randn(1000, 3)  # 랜덤 포인트클라우드
    
    # 궤적 생성
    result = engine.generate_trajectory(
        start_pose=start_pose,
        target_pose=target_pose,
        pointcloud=pointcloud,
        config=InferenceConfigs.default()
    )
    
    # 결과 출력
    print(f"✅ Success: {result['success']}")
    print(f"📊 Steps: {result['steps']}")
    print(f"⏱️ Time: {result['generation_time']:.3f}s")
    print(f"📏 Final position error: {result['final_error']['position_error_m']:.3f}m")
    print(f"🔄 Final rotation error: {result['final_error']['rotation_error_deg']:.1f}°")


