"""
Configuration Loader
설정 파일 로딩 및 관리 유틸리티
"""
import yaml
import os
from typing import Dict, Any, Optional

class ConfigLoader:
    """설정 파일 로더"""
    
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
            key_path: 설정 키 경로 (예: "simulation.fps")
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
    
    def get_robot_geometries(self) -> Dict[int, Dict[str, Any]]:
        """로봇 기하학 구성 전체 반환"""
        return self.get("robot_geometries", {})
    
    def get_robot_geometry(self, geometry_id: int) -> Optional[Dict[str, Any]]:
        """특정 로봇 기하학 구성 반환"""
        geometries = self.get_robot_geometries()
        return geometries.get(geometry_id)
    
    def get_simulation_config(self) -> Dict[str, Any]:
        """시뮬레이션 설정 반환"""
        return self.get("simulation", {})
    
    def get_pointcloud_config(self) -> Dict[str, Any]:
        """Pointcloud 설정 반환"""
        return self.get("pointcloud", {})
    
    def get_policy_config(self, policy_name: str) -> Dict[str, Any]:
        """특정 정책 설정 반환"""
        return self.get(f"policies.{policy_name}", {})
    
    def get_environment_config(self) -> Dict[str, Any]:
        """환경 설정 반환"""
        return self.get("environment", {})
    
    def get_video_config(self) -> Dict[str, Any]:
        """비디오 설정 반환"""
        return self.get("video", {})
    
    def reload(self) -> None:
        """설정 파일 다시 로드"""
        self._load_config()

# 전역 설정 인스턴스 (싱글톤 패턴)
_config_instance = None

def get_config(config_path: str = "config.yaml") -> ConfigLoader:
    """전역 설정 인스턴스 반환"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)
    return _config_instance

def reload_config() -> None:
    """전역 설정 다시 로드"""
    global _config_instance
    if _config_instance is not None:
        _config_instance.reload()

# 편의 함수들
def get_robot_geometry(geometry_id: int) -> Optional[Dict[str, Any]]:
    """특정 로봇 기하학 구성 반환"""
    config = get_config()
    return config.get_robot_geometry(geometry_id)

def list_robot_geometries() -> Dict[int, Dict[str, Any]]:
    """로봇 기하학 구성 목록 반환"""
    config = get_config()
    return config.get_robot_geometries()

def get_default_geometry_id() -> int:
    """기본 로봇 기하학 ID 반환"""
    config = get_config()
    return config.get("default_geometry", 0)

# 하위 호환성을 위한 별칭 함수들
def get_geometry_config(geometry_id: int) -> Optional[Dict[str, Any]]:
    """get_robot_geometry의 별칭 (하위 호환성)"""
    return get_robot_geometry(geometry_id)
