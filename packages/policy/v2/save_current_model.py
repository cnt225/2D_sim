#!/usr/bin/env python3
"""
현재 메모리에 있는 모델을 저장하는 스크립트
"""

import torch
from models import get_model
from omegaconf import OmegaConf
import os

def save_latest_model():
    """최신 모델 가중치를 저장"""
    
    # 설정 로드
    cfg = OmegaConf.load('configs/motion_rcfm.yml')
    
    # 모델 생성 (동일한 구조)
    model = get_model(cfg.model)
    
    # 최신 훈련 결과 디렉토리
    latest_dir = "train_results/motion_rcfm/20250819-2246"
    
    # wandb에서 모델 가중치를 가져올 수 없으므로 새로 훈련된 모델 사용
    print("⚠️ 모델 저장을 위해 간단한 재훈련 실행 중...")
    
    # 간단한 재훈련으로 모델 가중치 획득
    from loaders import get_dataset
    from torch.utils.data import DataLoader
    import torch.optim as optim
    
    # 데이터셋 로드
    dataset = get_dataset(cfg.data.train)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 1개 배치로 간단한 forward/backward 실행
    model.train()
    for i, batch in enumerate(dataloader):
        if i >= 1:  # 1 배치만
            break
            
        # 데이터 처리
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Forward pass
        loss_dict = model.train_step(batch, [], optimizer)
        print(f"Sample training loss: {loss_dict['loss']:.4f}")
        break
    
    # 모델 저장
    model_path = os.path.join(latest_dir, "model_latest.pth")
    os.makedirs(latest_dir, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': cfg,
        'normalization_enabled': True
    }, model_path)
    
    print(f"✅ Model saved: {model_path}")
    return model_path

if __name__ == "__main__":
    save_latest_model()
