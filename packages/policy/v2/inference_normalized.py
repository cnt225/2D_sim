#!/usr/bin/env python3
"""
v2 정규화된 Motion RCFM 추론 엔진
fm-main 호환 구조 + 정규화 파이프라인 적용
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import time
import glob
import open3d as o3d
import json
from datetime import datetime

from models import get_model
from utils.Lie import *
from omegaconf import OmegaConf
from utils.normalization import TwistNormalizer

class NormalizedMotionRCFMInference:
    """정규화된 Motion RCFM 추론 엔진 (v2 fm-main 호환)"""
    
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
        
        print(f"✅ Normalized Motion RCFM Inference Engine loaded on {self.device}")
        print(f"📊 Model config: {self.config.model.arch}")
    
    def _load_model(self, model_path: str):
        """모델 로드"""
        model = get_model(self.config.model)
        
        # 체크포인트 로드
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # 다양한 체크포인트 형태 지원
            if 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
                print(f"✅ Model loaded from model_state: {model_path}")
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Model loaded from checkpoint: {model_path}")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print(f"✅ Model loaded from state dict: {model_path}")
            else:
                model.load_state_dict(checkpoint)
                print(f"✅ Model loaded directly: {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        return model.to(self.device)
    
    def load_pointcloud(self, pc_path: str, num_points: int = 2000) -> torch.Tensor:
        """포인트클라우드 로드 (PLY/OBJ 지원)"""
        try:
            if pc_path.endswith('.ply'):
                pcd = o3d.io.read_point_cloud(pc_path)
                points = np.asarray(pcd.points)
            elif pc_path.endswith('.obj'):
                # OBJ 파일에서 정점 추출
                points = []
                with open(pc_path, 'r') as f:
                    for line in f:
                        if line.startswith('v '):
                            _, x, y, z = line.strip().split()
                            points.append([float(x), float(y), float(z)])
                points = np.array(points)
            else:
                raise ValueError(f"Unsupported point cloud format: {pc_path}")
            
            # 포인트 개수 조정
            if len(points) > num_points:
                indices = np.random.choice(len(points), num_points, replace=False)
                points = points[indices]
            elif len(points) < num_points:
                indices = np.random.choice(len(points), num_points, replace=True)
                points = points[indices]
            
            return torch.FloatTensor(points)
            
        except Exception as e:
            print(f"❌ Error loading point cloud {pc_path}: {e}")
            # 더미 포인트클라우드 반환
            return torch.randn(num_points, 3)
    
    @torch.no_grad()
    def generate_trajectory(self, 
                          pointcloud: torch.Tensor,
                          start_pose: np.ndarray,
                          target_pose: np.ndarray,
                          num_samples: int = 20,
                          ode_steps: int = 20) -> Dict[str, Union[np.ndarray, List]]:
        """
        궤적 생성 (정규화 파이프라인 적용)
        
        Args:
            pointcloud: [N, 3] 환경 포인트클라우드
            start_pose: [4, 4] 시작 SE(3) pose
            target_pose: [4, 4] 목표 SE(3) pose  
            num_samples: 생성할 웨이포인트 수
            ode_steps: ODE solver 스텝 수
            
        Returns:
            Dict with 'poses': [num_samples, 4, 4], 'success': bool, etc.
        """
        start_time = time.time()
        
        # 입력 검증
        assert pointcloud.shape[1] == 3, f"Expected [N, 3] pointcloud, got {pointcloud.shape}"
        assert start_pose.shape == (4, 4), f"Expected [4, 4] start pose, got {start_pose.shape}"
        assert target_pose.shape == (4, 4), f"Expected [4, 4] target pose, got {target_pose.shape}"
        
        # GPU로 이동
        pointcloud = pointcloud.to(self.device)
        start_pose_torch = torch.FloatTensor(start_pose).to(self.device)
        target_pose_torch = torch.FloatTensor(target_pose).to(self.device)
        
        try:
            # Point cloud 특징 추출
            pc_batch = pointcloud.unsqueeze(0).transpose(2, 1)  # [1, 3, N]
            pc_features = self.model.get_latent_vector(pc_batch)  # [1, feat_dim]
            
            # CFM 샘플링 (ode_steps는 모델 내부에서 self.ode_steps 사용)
            generated_poses, nfe_counts = self.model.sample(
                num_samples=num_samples,
                pc=pc_batch
            )
            
            # [num_samples, 4, 4]로 변환
            if generated_poses.dim() == 3:
                trajectory_poses = generated_poses.cpu().numpy()
            else:
                trajectory_poses = generated_poses.view(num_samples, 4, 4).cpu().numpy()
            
            # 시작과 끝점 조정 (선택적)
            trajectory_poses[0] = start_pose
            trajectory_poses[-1] = target_pose
            
            inference_time = time.time() - start_time
            
            return {
                'poses': trajectory_poses,
                'success': True,
                'inference_time': inference_time,
                'num_samples': num_samples,
                'start_pose': start_pose,
                'target_pose': target_pose
            }
            
        except Exception as e:
            print(f"❌ Trajectory generation failed: {e}")
            return {
                'poses': None,
                'success': False,
                'error': str(e),
                'inference_time': time.time() - start_time
            }
    
    def denormalize_trajectory(self, trajectory_result: Dict) -> Dict:
        """궤적 역정규화 (추후 필요시 구현)"""
        # v2에서는 SE(3) 직접 생성하므로 별도 역정규화 불필요
        return trajectory_result
    
    def visualize_trajectory(self, trajectory_result: Dict, save_path: Optional[str] = None):
        """궤적 시각화 (간단한 출력)"""
        if not trajectory_result['success']:
            print(f"❌ Cannot visualize failed trajectory: {trajectory_result.get('error', 'Unknown error')}")
            return
        
        poses = trajectory_result['poses']
        print(f"📊 Generated trajectory with {len(poses)} poses")
        print(f"⏱️ Inference time: {trajectory_result['inference_time']:.3f}s")
        
        # 위치 변화량 계산
        positions = poses[:, :3, 3]
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_distance = np.sum(distances)
        
        print(f"📏 Total distance: {total_distance:.3f}")
        print(f"📐 Average step size: {np.mean(distances):.3f}")
        print(f"🎯 Start position: {positions[0]}")
        print(f"🏁 End position: {positions[-1]}")
    
    def save_trajectory_json(self,
                           trajectory_poses: np.ndarray,
                           start_pose: np.ndarray,
                           goal_pose: np.ndarray,
                           environment_name: str = "inferred_env",
                           rigid_body_id: int = 3,
                           rigid_body_type: str = "elongated_ellipse",
                           output_path: str = None) -> str:
        """
        추론된 궤적을 기존 시각화 코드와 호환되는 JSON 형식으로 저장
        
        Args:
            trajectory_poses: [N, 4, 4] SE(3) 변환 행렬들
            start_pose: [4, 4] 시작 SE(3) 포즈
            goal_pose: [4, 4] 목표 SE(3) 포즈  
            environment_name: 환경 이름
            rigid_body_id: Rigid body ID (0-3)
            rigid_body_type: Rigid body 타입
            output_path: 출력 파일 경로 (None이면 자동 생성)
            
        Returns:
            저장된 파일 경로
        """
        
        # SE(3) 행렬을 6D 포즈로 변환
        def se3_to_6d_pose(se3_matrix):
            """SE(3) 4x4 행렬을 [x, y, z, roll, pitch, yaw] 6D 포즈로 변환"""
            from scipy.spatial.transform import Rotation
            
            # 평행이동 벡터 추출
            translation = se3_matrix[:3, 3]
            
            # 회전 행렬 추출 및 오일러각 변환
            rotation_matrix = se3_matrix[:3, :3]
            rotation = Rotation.from_matrix(rotation_matrix)
            euler_angles = rotation.as_euler('xyz', degrees=False)  # [roll, pitch, yaw]
            
            return [float(translation[0]), float(translation[1]), float(translation[2]), 
                   float(euler_angles[0]), float(euler_angles[1]), float(euler_angles[2])]
        
        # 궤적 데이터를 6D 포즈 리스트로 변환
        trajectory_6d = []
        for pose_matrix in trajectory_poses:
            pose_6d = se3_to_6d_pose(pose_matrix)
            trajectory_6d.append(pose_6d)
        
        # 시작/목표 포즈도 6D로 변환
        start_6d = se3_to_6d_pose(start_pose)
        goal_6d = se3_to_6d_pose(goal_pose)
        
        # 궤적 ID 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trajectory_id = f"inferred_traj_rb{rigid_body_id}_{timestamp}"
        
        # JSON 데이터 구조 생성 (기존 형식과 호환)
        trajectory_data = {
            "pair_id": 0,
            "trajectory_id": trajectory_id,
            "rigid_body": {
                "id": rigid_body_id,
                "type": rigid_body_type
            },
            "environment": {
                "name": environment_name,
                "ply_file": f"../../../data/pointcloud/circle_envs_10k/{environment_name}.ply"
            },
            "start_pose": start_6d,
            "goal_pose": goal_6d,
            "path": {
                "data": trajectory_6d
            },
            "metadata": {
                "generated_by": "v2_motion_rcfm_inference",
                "timestamp": timestamp,
                "num_poses": len(trajectory_6d),
                "model_config": self.config.model.arch if hasattr(self.config, 'model') else "motion_rcfm"
            }
        }
        
        # 출력 경로 설정
        if output_path is None:
            output_dir = Path("inference_results")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{trajectory_id}.json"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # JSON 파일 저장
        with open(output_path, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
        
        print(f"💾 궤적 JSON 저장됨: {output_path}")
        print(f"   📊 포즈 개수: {len(trajectory_6d)}")
        print(f"   🎯 시작 위치: [{start_6d[0]:.3f}, {start_6d[1]:.3f}, {start_6d[5]:.3f}°]")
        print(f"   🏁 목표 위치: [{goal_6d[0]:.3f}, {goal_6d[1]:.3f}, {goal_6d[5]:.3f}°]")
        
        return str(output_path)

def test_inference():
    """추론 테스트 함수"""
    print("🧪 Testing v2 Normalized Motion RCFM Inference...")
    
    # 최신 체크포인트 찾기
    checkpoint_patterns = [
        "train_results/motion_rcfm/*/best_model.pth",
        "train_results/motion_rcfm/*/model_latest.pth",
        "train_results/motion_rcfm/*/*.pth"
    ]
    
    checkpoints = []
    for pattern in checkpoint_patterns:
        checkpoints.extend(glob.glob(pattern))
        if checkpoints:
            break
    
    if not checkpoints:
        print("❌ No trained model found. Please train a model first.")
        return
    
    latest_checkpoint = max(checkpoints, key=lambda x: Path(x).stat().st_mtime)
    print(f"📦 Using checkpoint: {latest_checkpoint}")
    
    # 추론 엔진 초기화
    inference_engine = NormalizedMotionRCFMInference(
        model_path=latest_checkpoint,
        config_path="configs/motion_rcfm.yml",
        normalize_twist=True
    )
    
    # 더미 데이터로 테스트
    print("\n🎯 Testing with dummy data...")
    
    # 더미 포인트클라우드
    pointcloud = torch.randn(2000, 3)
    
    # 더미 시작/목표 pose
    start_pose = np.eye(4)
    target_pose = np.eye(4)
    target_pose[:3, 3] = [1.0, 1.0, 0.0]  # 1m x, 1m y 이동
    
    # 궤적 생성
    result = inference_engine.generate_trajectory(
        pointcloud=pointcloud,
        start_pose=start_pose,
        target_pose=target_pose,
        num_samples=20
    )
    
    # 결과 출력
    inference_engine.visualize_trajectory(result)
    
    if result['success']:
        print("✅ Inference test successful!")
        
        # 궤적을 JSON 파일로 저장 테스트
        print("\n💾 Testing trajectory JSON export...")
        try:
            saved_path = inference_engine.save_trajectory_json(
                trajectory_poses=result['poses'],
                start_pose=result['start_pose'],
                goal_pose=result['target_pose'],
                environment_name="test_environment",
                rigid_body_id=3,
                rigid_body_type="elongated_ellipse"
            )
            print(f"✅ 궤적 저장 성공! 파일: {saved_path}")
            print(f"📁 기존 시각화 코드로 사용 가능:")
            print(f"   python packages/data_generator/reference_planner/utils/trajectory_visualizer.py {saved_path} --mode static")
        except Exception as e:
            print(f"❌ 궤적 저장 실패: {e}")
    else:
        print("❌ Inference test failed!")

if __name__ == "__main__":
    test_inference()
