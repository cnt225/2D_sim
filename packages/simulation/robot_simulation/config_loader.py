"""
Configuration Loader
SE(3) Rigid Body 시뮬레이션을 위한 설정 파일 로딩 및 관리 유틸리티
"""
import yaml
import os
from typing import Dict, Any, Optional

class ConfigLoader:
    """SE(3) Rigid Body 설정 파일 로더"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = config_path
        self._config = None
        self._load_config()
    
    def _load_config(self) -> None:
        """설정 파일 로드"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML config: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load config file: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        점 표기법으로 설정값 가져오기
        
        Args:
            key_path: 설정 키 경로 (예: "se3_simulation.control.position_gain")
            default: 기본값
            
        Returns:
            설정값
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_rigid_bodies(self) -> Dict[int, Dict[str, Any]]:
        """모든 SE(3) rigid body 설정 반환"""
        return self.get("rigid_bodies", {})
    
    def get_rigid_body_config(self, rigid_body_id: int) -> Optional[Dict[str, Any]]:
        """특정 SE(3) rigid body 설정 반환"""
        rigid_bodies = self.get_rigid_bodies()
        return rigid_bodies.get(rigid_body_id)
    
    def get_se3_simulation_config(self) -> Dict[str, Any]:
        """SE(3) 시뮬레이션 설정 반환"""
        return self.get("se3_simulation", {})
    
    def get_data_generation_config(self) -> Dict[str, Any]:
        """데이터 생성 설정 반환"""
        return self.get("data_generation", {})
    
    def get_default_geometry_id(self) -> int:
        """기본 rigid body ID 반환"""
        return self.get("default_geometry", 0)
    
    def reload(self) -> None:
        """설정 파일 다시 로드"""
        self._load_config()

# 전역 설정 인스턴스 (싱글톤 패턴)
_config_instance = None

def get_config(config_path: str = None) -> ConfigLoader:
    """전역 설정 인스턴스 반환"""
    global _config_instance
    if _config_instance is None:
        # Default config path: try to find robot_geometries.yaml
        if config_path is None:
            import os
            # Try different possible paths
            possible_paths = [
                "config/robot_geometries.yaml",
                "../config/robot_geometries.yaml", 
                "robot_simulation/config/robot_geometries.yaml",
                "packages/simulation/config/robot_geometries.yaml",
                "config.yaml"  # fallback
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            else:
                # If no config found, use the most likely path
                config_path = "config/robot_geometries.yaml"
        
        _config_instance = ConfigLoader(config_path)
    return _config_instance

def reload_config() -> None:
    """전역 설정 다시 로드"""
    global _config_instance
    if _config_instance is not None:
        _config_instance.reload()

# SE(3) Rigid Body 편의 함수들
def get_rigid_body_config(rigid_body_id: int) -> Optional[Dict[str, Any]]:
    """특정 SE(3) rigid body 설정 반환"""
    config = get_config()
    return config.get_rigid_body_config(rigid_body_id)

def list_rigid_bodies() -> Dict[int, Dict[str, Any]]:
    """SE(3) rigid body 설정 목록 반환"""
    config = get_config()
    return config.get_rigid_bodies()

def get_se3_simulation_config() -> Dict[str, Any]:
    """SE(3) 시뮬레이션 설정 반환"""
    config = get_config()
    return config.get_se3_simulation_config()

def get_default_geometry_id() -> int:
    """기본 rigid body ID 반환"""
    config = get_config()
    return config.get_default_geometry_id()

def is_valid_rigid_body_id(geometry_id: int) -> bool:
    """해당 ID가 유효한 rigid body인지 확인 (0-3)"""
    return 0 <= geometry_id <= 3

# 호환성을 위한 별칭 함수들
def get_geometry_config(geometry_id: int) -> Optional[Dict[str, Any]]:
    """get_rigid_body_config의 별칭 (호환성)"""
    return get_rigid_body_config(geometry_id)
