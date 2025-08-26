#!/usr/bin/env python3
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.append('/home/dhkang225/2D_sim/packages/policy/v2')

from models import get_model
from loaders import get_dataloader
from omegaconf import OmegaConf

def test_simple_inference():
    """간단한 추론 테스트"""
    print("🧪 Simple Inference Test...")
    
    # Config 로드
    cfg = OmegaConf.load('configs/motion_rcfm.yml')
    
    # 모델 생성
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(cfg['model']).to(device)
    
    # 체크포인트 로드 (속도 기반 모델)
    checkpoint_path = 'train_results/motion_rcfm/20250820-0025/model_best_normalized.pth'
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # 체크포인트 구조 확인
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    print(f"🔧 Checkpoint keys: {list(checkpoint.keys())}")
    
    # 모델 상태 로드
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✅ Model loaded successfully!")
    
    # 테스트 데이터 로드
    test_cfg = cfg['data']['test']
    dataloader = get_dataloader(test_cfg)
    batch = next(iter(dataloader))
    
    # GPU로 이동
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    
    print(f"📊 Test batch loaded: {batch['current_T'].shape[0]} samples")
    
    # 추론 실행
    with torch.no_grad():
        # 포인트클라우드 특징 추출
        pc_features = model.get_latent_vector(batch['pc'])
        
        # T_dot 예측
        T_dot_pred = model.forward(
            batch['current_T'], 
            batch['target_T'], 
            batch['time_t'], 
            pc_features, 
            batch['g']
        )
        
        print(f"🎯 Prediction successful!")
        print(f"   Input shapes:")
        print(f"     current_T: {batch['current_T'].shape}")
        print(f"     target_T: {batch['target_T'].shape}")
        print(f"     time_t: {batch['time_t'].shape}")
        print(f"     pc: {batch['pc'].shape}")
        print(f"     g: {batch['g'].shape}")
        print(f"   Output shape: {T_dot_pred.shape}")
        
        # 예측 결과 분석
        T_dot_gt = batch['T_dot']
        mse_loss = torch.nn.functional.mse_loss(T_dot_pred, T_dot_gt)
        
        print(f"📈 Results:")
        print(f"   MSE Loss: {mse_loss.item():.6f}")
        print(f"   Predicted T_dot range: [{T_dot_pred.min().item():.4f}, {T_dot_pred.max().item():.4f}]")
        print(f"   Ground truth T_dot range: [{T_dot_gt.min().item():.4f}, {T_dot_gt.max().item():.4f}]")
        
        # 샘플별 비교
        for i in range(min(2, T_dot_pred.shape[0])):
            print(f"\n   Sample {i}:")
            print(f"     Predicted: {T_dot_pred[i].cpu().numpy()}")
            print(f"     Actual:    {T_dot_gt[i].cpu().numpy()}")
            print(f"     Error:     {torch.abs(T_dot_pred[i] - T_dot_gt[i]).cpu().numpy()}")

if __name__ == "__main__":
    test_simple_inference()
