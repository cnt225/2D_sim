"""
Data Interface for loading external data into simulation

This module provides a unified interface for loading data from 
data_generator and rfm_policy packages.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class DataInterface:
    """통합 데이터 로딩 인터페이스"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize data interface
        
        Args:
            config_path: Path to data_paths.yaml config file
        """
        if config_path is None:
            # Default to config in same package
            config_path = Path(__file__).parent.parent / "config" / "data_paths.yaml"
            
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load data paths configuration"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file not found at {self.config_path}")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file not found"""
        return {
            'data_sources': {
                'data_generator_base': '../data_generator',
                'environments': 'data/pointcloud/',
                'poses': 'data/pose/',
                'trajectories': 'data/trajectories/'
            },
            'models': {
                'rfm_policy_base': '../rfm_policy',
                'pretrained': 'models/pretrained/',
                'checkpoints': 'experiments/checkpoints/'
            }
        }
    
    def get_environment_path(self, env_name: str) -> Path:
        """Get path to environment PLY file"""
        base_path = Path(self.config['data_sources']['data_generator_base'])
        env_path = base_path / self.config['data_sources']['environments']
        return env_path / f"{env_name}.ply"
        
    def get_pose_data_path(self, env_name: str, robot_id: int) -> Path:
        """Get path to pose data JSON file"""
        base_path = Path(self.config['data_sources']['data_generator_base'])
        pose_path = base_path / self.config['data_sources']['poses']
        return pose_path / f"{env_name}_geo_{robot_id}_poses.json"
        
    def get_model_path(self, model_name: str) -> Path:
        """Get path to trained RFM model"""
        base_path = Path(self.config['models']['rfm_policy_base'])
        model_path = base_path / self.config['models']['pretrained']
        return model_path / f"{model_name}.pt"
        
    def list_available_environments(self) -> list:
        """List all available environment files"""
        base_path = Path(self.config['data_sources']['data_generator_base'])
        env_path = base_path / self.config['data_sources']['environments']
        
        if not env_path.exists():
            return []
            
        ply_files = list(env_path.glob("*.ply"))
        return [f.stem for f in ply_files]
        
    def check_data_availability(self) -> Dict[str, bool]:
        """Check if data sources are available"""
        results = {}
        
        # Check data generator
        data_gen_path = Path(self.config['data_sources']['data_generator_base'])
        results['data_generator'] = data_gen_path.exists()
        
        # Check RFM policy  
        rfm_path = Path(self.config['models']['rfm_policy_base'])
        results['rfm_policy'] = rfm_path.exists()
        
        return results


# Convenience function for easy import
def get_data_interface(config_path: Optional[str] = None) -> DataInterface:
    """Get a configured data interface instance"""
    return DataInterface(config_path) 