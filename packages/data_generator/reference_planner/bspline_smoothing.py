"""
B-spline 궤적 스무딩 구현
SE(2) 매니폴드 상에서 RRT 궤적을 부드럽게 스무딩
SE(3) 쿼터니언 기반 스무딩 지원 (신규)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.spatial.distance import cdist
from pathlib import Path
import sys

# SE3 functions import for quaternion operations
sys.path.append(str(Path(__file__).parent.parent.parent / 'utils'))
from SE3_functions import (
    bspline_quaternion_smoothing,
    trajectory_euler_to_quaternion,
    trajectory_quaternion_to_euler,
    quaternion_slerp_interpolation
)


def normalize_angle(angle):
    """각도를 [-π, π] 범위로 정규화"""
    return np.arctan2(np.sin(angle), np.cos(angle))


def unwrap_angles(angles):
    """각도 연속성 보장 (unwrapping)"""
    unwrapped = [angles[0]]
    for i in range(1, len(angles)):
        diff = angles[i] - unwrapped[-1]
        # 최단 경로 선택
        if diff > np.pi:
            unwrapped.append(angles[i] - 2*np.pi)
        elif diff < -np.pi:
            unwrapped.append(angles[i] + 2*np.pi)
        else:
            unwrapped.append(angles[i])
    return np.array(unwrapped)


def compute_arc_length_parameterization(waypoints):
    """아크 길이 기반 파라미터화"""
    if len(waypoints) < 2:
        return np.array([0.0])
    
    # 위치만 고려해서 아크 길이 계산 (SE(2)에서 각도는 별도 처리)
    positions = waypoints[:, :2]  # [x, y]
    distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
    
    # [0, 1] 범위로 정규화
    total_length = cumulative_distances[-1]
    if total_length == 0:
        return np.linspace(0, 1, len(waypoints))
    
    return cumulative_distances / total_length


def bspline_trajectory_smoother(rrt_waypoints, num_points=None, degree=3, smoothing_factor=0):
    """
    RRT 궤적을 B-spline으로 스무딩 (SE(2) 매니폴드 고려)
    
    Args:
        rrt_waypoints: [N, 3] RRT 웨이포인트 [x, y, theta]
        num_points: 출력 궤적 포인트 수 (None이면 원본의 2배)
        degree: B-spline 차수 (3=cubic)
        smoothing_factor: 스무딩 강도 (0=보간, >0=근사)
    
    Returns:
        smooth_trajectory: [num_points, 3] 스무딩된 궤적
    """
    
    if len(rrt_waypoints) < degree + 1:
        print(f"Warning: 웨이포인트 수({len(rrt_waypoints)})가 차수+1({degree+1})보다 작습니다.")
        return rrt_waypoints
    
    if num_points is None:
        num_points = len(rrt_waypoints) * 2
    
    # 1. 각도 unwrapping
    angles = unwrap_angles(rrt_waypoints[:, 2])
    
    # 2. 아크 길이 기반 파라미터화
    t_original = compute_arc_length_parameterization(rrt_waypoints)
    t_smooth = np.linspace(0, 1, num_points)
    
    # 3. 각 차원별 B-spline fitting
    try:
        # 위치 (x, y)
        x_spline = interpolate.splrep(t_original, rrt_waypoints[:, 0], k=degree, s=smoothing_factor)
        y_spline = interpolate.splrep(t_original, rrt_waypoints[:, 1], k=degree, s=smoothing_factor)
        
        # 각도 (theta) - unwrapped 상태에서 처리
        theta_spline = interpolate.splrep(t_original, angles, k=degree, s=smoothing_factor)
        
        # 4. 스무딩된 궤적 생성
        x_smooth = interpolate.splev(t_smooth, x_spline)
        y_smooth = interpolate.splev(t_smooth, y_spline)
        theta_smooth = interpolate.splev(t_smooth, theta_spline)
        
        # 5. 각도 정규화
        theta_smooth = [normalize_angle(a) for a in theta_smooth]
        
        return np.column_stack([x_smooth, y_smooth, theta_smooth])
        
    except Exception as e:
        print(f"B-spline fitting 오류: {e}")
        return rrt_waypoints


def calculate_smoothness_metrics(trajectory):
    """궤적의 부드러움 메트릭 계산"""
    if len(trajectory) < 3:
        return {"curvature_variance": 0.0, "acceleration_norm": 0.0, "jerk_norm": 0.0}
    
    # 1차 미분 (속도)
    velocities = np.diff(trajectory, axis=0)
    
    # 2차 미분 (가속도)
    accelerations = np.diff(velocities, axis=0)
    
    # 3차 미분 (저크)
    jerks = np.diff(accelerations, axis=0)
    
    # 곡률 분산 계산 (위치만)
    positions = trajectory[:, :2]
    curvatures = []
    for i in range(1, len(positions) - 1):
        p1, p2, p3 = positions[i-1], positions[i], positions[i+1]
        
        # 벡터
        v1 = p2 - p1
        v2 = p3 - p2
        
        # 곡률 계산 (외적 기반)
        if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
            cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            curvature = abs(cross_product) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            curvatures.append(curvature)
    
    metrics = {
        "curvature_variance": np.var(curvatures) if curvatures else 0.0,
        "acceleration_norm": np.mean(np.linalg.norm(accelerations, axis=1)) if len(accelerations) > 0 else 0.0,
        "jerk_norm": np.mean(np.linalg.norm(jerks, axis=1)) if len(jerks) > 0 else 0.0,
        "path_length": np.sum(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)),
        "num_waypoints": len(trajectory)
    }
    
    return metrics


def load_rrt_trajectory(trajectory_file):
    """RRT 궤적 파일 로드"""
    with open(trajectory_file, 'r') as f:
        data = json.load(f)
    
    # SE(3) → SE(2) 변환 [x, y, z, roll, pitch, yaw] → [x, y, yaw]
    path_data = np.array(data['path']['data'])
    se2_trajectory = path_data[:, [0, 1, 5]]  # x, y, yaw 추출
    
    return se2_trajectory, data


def visualize_trajectory_comparison(original, smoothed, title="궤적 비교"):
    """원본 vs 스무딩된 궤적 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 전체 궤적 비교
    ax1 = axes[0, 0]
    ax1.plot(original[:, 0], original[:, 1], 'r.-', label='Original RRT', alpha=0.7, markersize=4)
    ax1.plot(smoothed[:, 0], smoothed[:, 1], 'b-', label='B-spline Smoothed', linewidth=2)
    ax1.scatter(original[0, 0], original[0, 1], color='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(original[-1, 0], original[-1, 1], color='red', s=100, marker='s', label='Goal', zorder=5)
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title('궤적 경로 비교')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. 각도 변화
    ax2 = axes[0, 1]
    t_orig = np.arange(len(original))
    t_smooth = np.arange(len(smoothed))
    ax2.plot(t_orig, original[:, 2], 'r.-', label='Original', alpha=0.7, markersize=4)
    ax2.plot(t_smooth, smoothed[:, 2], 'b-', label='Smoothed', linewidth=2)
    ax2.set_xlabel('Waypoint Index')
    ax2.set_ylabel('Theta [rad]')
    ax2.set_title('각도 변화')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 속도 프로파일 비교
    ax3 = axes[1, 0]
    if len(original) > 1:
        orig_vel = np.linalg.norm(np.diff(original[:, :2], axis=0), axis=1)
        ax3.plot(orig_vel, 'r.-', label='Original', alpha=0.7, markersize=4)
    
    if len(smoothed) > 1:
        smooth_vel = np.linalg.norm(np.diff(smoothed[:, :2], axis=0), axis=1)
        ax3.plot(smooth_vel, 'b-', label='Smoothed', linewidth=2)
    
    ax3.set_xlabel('Segment Index')
    ax3.set_ylabel('Linear Velocity [m/step]')
    ax3.set_title('속도 프로파일')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 방향 화살표 시각화 (서브샘플링)
    ax4 = axes[1, 1]
    ax4.plot(original[:, 0], original[:, 1], 'r.-', alpha=0.3, markersize=2)
    ax4.plot(smoothed[:, 0], smoothed[:, 1], 'b-', linewidth=2)
    
    # 방향 화살표 (서브샘플링)
    step = max(1, len(smoothed) // 10)
    for i in range(0, len(smoothed), step):
        x, y, theta = smoothed[i]
        dx = 0.3 * np.cos(theta)
        dy = 0.3 * np.sin(theta)
        ax4.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    ax4.set_xlabel('X [m]')
    ax4.set_ylabel('Y [m]')
    ax4.set_title('방향 화살표 (스무딩됨)')
    ax4.axis('equal')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    return fig


def main():
    """메인 실행 함수"""
    # 궤적 파일 로드
    trajectory_file = "data/trajectories/circle_envs_10k/circle_env_000000/trajectory_00.json"
    
    print("RRT 궤적 로딩 중...")
    original_trajectory, trajectory_data = load_rrt_trajectory(trajectory_file)
    
    print(f"원본 궤적: {len(original_trajectory)} 웨이포인트")
    print(f"로봇: {trajectory_data['rigid_body']['type']} (ID: {trajectory_data['rigid_body']['id']})")
    print(f"환경: {trajectory_data['environment']['name']}")
    
    # B-spline 스무딩 적용
    print("\nB-spline 스무딩 적용 중...")
    smoothed_trajectory = bspline_trajectory_smoother(
        original_trajectory, 
        num_points=len(original_trajectory) * 2,  # 2배 밀도
        degree=3,
        smoothing_factor=0  # 보간 모드
    )
    
    print(f"스무딩된 궤적: {len(smoothed_trajectory)} 웨이포인트")
    
    # 부드러움 메트릭 계산
    print("\n부드러움 메트릭 계산 중...")
    original_metrics = calculate_smoothness_metrics(original_trajectory)
    smoothed_metrics = calculate_smoothness_metrics(smoothed_trajectory)
    
    print("\n=== 부드러움 메트릭 비교 ===")
    print(f"{'메트릭':<20} {'원본':<15} {'스무딩됨':<15} {'개선율':<10}")
    print("-" * 65)
    
    for key in original_metrics:
        orig_val = original_metrics[key]
        smooth_val = smoothed_metrics[key]
        
        if orig_val > 0:
            improvement = (orig_val - smooth_val) / orig_val * 100
            improvement_str = f"{improvement:.1f}%"
        else:
            improvement_str = "N/A"
        
        print(f"{key:<20} {orig_val:<15.4f} {smooth_val:<15.4f} {improvement_str:<10}")
    
    # 시각화
    print("\n궤적 시각화 중...")
    fig = visualize_trajectory_comparison(
        original_trajectory, 
        smoothed_trajectory, 
        title=f"B-spline 스무딩 비교 - {trajectory_data['environment']['name']}"
    )
    
    # 저장
    output_file = "data/visualizations/bspline_smoothing_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"시각화 결과 저장: {output_file}")
    
    plt.show()
    
    return original_trajectory, smoothed_trajectory, original_metrics, smoothed_metrics


def create_bsplined_trajectory_file(input_trajectory_file, output_dir=None, 
                                   degree=3, smoothing_factor=0.1, density_multiplier=2):
    """
    기존 궤적 파일에 B-spline 스무딩을 적용하고 동일한 형식으로 저장
    
    Args:
        input_trajectory_file: 원본 궤적 JSON 파일 경로
        output_dir: 출력 디렉토리 (None이면 원본과 동일한 디렉토리)
        degree: B-spline 차수
        smoothing_factor: 스무딩 강도
        density_multiplier: 웨이포인트 밀도 배수
    
    Returns:
        output_file_path: 생성된 B-spline 궤적 파일 경로
    """
    
    # 1. 원본 궤적 데이터 로드
    with open(input_trajectory_file, 'r') as f:
        original_data = json.load(f)
    
    print(f"원본 궤적 로드: {input_trajectory_file}")
    print(f"- 궤적 ID: {original_data['trajectory_id']}")
    print(f"- 웨이포인트 수: {len(original_data['path']['data'])}")
    
    # 2. SE(3) → SE(2) 변환
    se3_path = np.array(original_data['path']['data'])
    se2_trajectory = se3_path[:, [0, 1, 5]]  # x, y, rz 추출
    
    # 3. B-spline 스무딩 적용
    num_points = int(len(se2_trajectory) * density_multiplier)
    smoothed_se2 = bspline_trajectory_smoother(
        se2_trajectory,
        degree=degree,
        num_points=num_points,
        smoothing_factor=smoothing_factor
    )
    
    print(f"B-spline 스무딩 적용:")
    print(f"- 원본 → 스무딩: {len(se2_trajectory)} → {len(smoothed_se2)} waypoints")
    print(f"- 설정: degree={degree}, smoothing={smoothing_factor}")
    
    # 4. SE(2) → SE(3) 변환 (원본 형식 유지)
    smoothed_se3_path = []
    for point in smoothed_se2:
        x, y, theta = point
        # 원본과 동일한 SE(3) 형식: [x, y, z, rx, ry, rz]
        se3_pose = [x, y, 0.0, 0.0, 0.0, theta]
        smoothed_se3_path.append(se3_pose)
    
    # 5. 새로운 궤적 데이터 생성 (원본 형식 완전 유지)
    bsplined_data = original_data.copy()
    
    # 궤적 ID에 "_bsplined" 추가
    original_id = original_data['trajectory_id']
    bsplined_data['trajectory_id'] = f"{original_id}_bsplined"
    
    # 궤적 경로 업데이트
    bsplined_data['path']['data'] = smoothed_se3_path
    
    # start_pose, goal_pose 업데이트
    bsplined_data['start_pose'] = smoothed_se3_path[0]
    bsplined_data['goal_pose'] = smoothed_se3_path[-1]
    
    # 메타데이터 추가
    if 'bspline_metadata' not in bsplined_data:
        bsplined_data['bspline_metadata'] = {}
    
    bsplined_data['bspline_metadata'].update({
        'original_trajectory_file': str(input_trajectory_file),
        'original_waypoints': len(se2_trajectory),
        'smoothed_waypoints': len(smoothed_se2),
        'bspline_degree': degree,
        'smoothing_factor': smoothing_factor,
        'density_multiplier': density_multiplier
    })
    
    # 부드러움 메트릭 계산 및 추가
    original_metrics = calculate_smoothness_metrics(se2_trajectory)
    smoothed_metrics = calculate_smoothness_metrics(smoothed_se2)
    
    bsplined_data['bspline_metadata']['metrics'] = {
        'original': original_metrics,
        'smoothed': smoothed_metrics,
        'improvements': {
            'curvature_variance': f"{(1-smoothed_metrics['curvature_variance']/original_metrics['curvature_variance'])*100:.1f}%" if original_metrics['curvature_variance'] > 0 else "N/A",
            'acceleration_norm': f"{(1-smoothed_metrics['acceleration_norm']/original_metrics['acceleration_norm'])*100:.1f}%" if original_metrics['acceleration_norm'] > 0 else "N/A",
            'jerk_norm': f"{(1-smoothed_metrics['jerk_norm']/original_metrics['jerk_norm'])*100:.1f}%" if original_metrics['jerk_norm'] > 0 else "N/A"
        }
    }
    
    # 6. 출력 파일 경로 설정
    input_path = Path(input_trajectory_file)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일명에 "_bsplined" 추가
    output_filename = input_path.stem + "_bsplined" + input_path.suffix
    output_file_path = output_dir / output_filename
    
    # 7. 저장
    with open(output_file_path, 'w') as f:
        json.dump(bsplined_data, f, indent=2)
    
    print(f"\\n✅ B-spline 궤적 저장 완료:")
    print(f"   파일: {output_file_path}")
    print(f"   개선사항:")
    print(f"   - 곡률 분산: {bsplined_data['bspline_metadata']['metrics']['improvements']['curvature_variance']}")
    print(f"   - 가속도 노름: {bsplined_data['bspline_metadata']['metrics']['improvements']['acceleration_norm']}")
    print(f"   - 저크 노름: {bsplined_data['bspline_metadata']['metrics']['improvements']['jerk_norm']}")
    
    return str(output_file_path)


# === 쿼터니언 기반 B-spline 스무딩 함수들 (신규 추가) ===

def bspline_quaternion_trajectory_smoother(trajectory_7d, num_points=None, degree=3):
    """
    쿼터니언 기반 B-spline 스무딩
    - 위치: 기존 B-spline 방식
    - 회전: SLERP 기반 스무딩
    
    Args:
        trajectory_7d: [N, 7] 궤적 - [[x,y,z,qw,qx,qy,qz], ...]
        num_points: 출력 포인트 수 (None이면 입력 * 2)
        degree: B-spline 차수
    
    Returns:
        [M, 7] 스무딩된 쿼터니언 궤적
    """
    if isinstance(trajectory_7d, list):
        trajectory_7d = np.array(trajectory_7d)
    
    if len(trajectory_7d.shape) != 2 or trajectory_7d.shape[1] != 7:
        raise ValueError(f"Expected [N, 7] trajectory, got shape {trajectory_7d.shape}")
    
    N = trajectory_7d.shape[0]
    if N < 2:
        return trajectory_7d
    
    if num_points is None:
        num_points = N * 2
    
    print(f"🔄 Quaternion B-spline smoothing: {N} → {num_points} waypoints")
    
    # bspline_quaternion_smoothing 함수 사용 (SE3_functions에서 구현됨)
    smoothed_trajectory = bspline_quaternion_smoothing(trajectory_7d, num_points, degree)
    
    return smoothed_trajectory


def create_bsplined_trajectory_hdf5(hdf5_file, env_id, rb_id, input_trajectory_type='raw',
                                   degree=3, density_multiplier=2):
    """
    HDF5 내에서 직접 B-spline 스무딩 적용
    
    Args:
        hdf5_file: h5py.File 객체
        env_id: 환경 ID
        rb_id: 로봇 ID
        input_trajectory_type: 입력 궤적 타입 ('raw')
        degree: B-spline 차수
        density_multiplier: 밀도 배수
    
    Returns:
        int: 스무딩 처리된 궤적 수
    """
    import h5py
    
    print(f"🚀 HDF5 B-spline smoothing for {env_id}/rb_{rb_id}")
    
    # 입력 그룹 경로
    input_path = f"trajectories/{input_trajectory_type}/{env_id}/rb_{rb_id}"
    output_path = f"trajectories/bsplined/{env_id}/rb_{rb_id}"
    
    if input_path not in hdf5_file:
        print(f"⚠️ No input trajectories found: {input_path}")
        return 0
    
    # 출력 그룹 생성
    if output_path not in hdf5_file:
        hdf5_file.create_group(output_path)
    
    input_group = hdf5_file[input_path]
    output_group = hdf5_file[output_path]
    
    processed_count = 0
    
    for traj_key in input_group.keys():
        if not traj_key.startswith('traj_'):
            continue
        
        try:
            # 원본 궤적 로드 (7D)
            trajectory_7d = input_group[traj_key][...]
            
            if trajectory_7d.shape[1] != 7:
                print(f"⚠️ Skipping {traj_key}: expected 7D data, got {trajectory_7d.shape}")
                continue
            
            # B-spline 스무딩 적용
            smoothed_7d = bspline_quaternion_trajectory_smoother(
                trajectory_7d,
                num_points=len(trajectory_7d) * density_multiplier,
                degree=degree
            )
            
            # HDF5에 저장
            if traj_key in output_group:
                del output_group[traj_key]  # 기존 데이터 교체
            
            smoothed_dataset = output_group.create_dataset(
                traj_key,
                data=smoothed_7d,
                compression='gzip',
                compression_opts=6
            )
            
            # 메타데이터 복사 및 추가
            original_attrs = dict(input_group[traj_key].attrs)
            for key, value in original_attrs.items():
                smoothed_dataset.attrs[key] = value
            
            # B-spline 메타데이터 추가
            smoothed_dataset.attrs['smoothing_method'] = 'quaternion_bspline'
            smoothed_dataset.attrs['bspline_degree'] = degree
            smoothed_dataset.attrs['density_multiplier'] = density_multiplier
            smoothed_dataset.attrs['original_waypoints'] = len(trajectory_7d)
            smoothed_dataset.attrs['smoothed_waypoints'] = len(smoothed_7d)
            
            processed_count += 1
            print(f"✅ Processed {traj_key}: {len(trajectory_7d)} → {len(smoothed_7d)} waypoints")
            
        except Exception as e:
            print(f"❌ Error processing {traj_key}: {e}")
    
    print(f"🎯 B-spline smoothing complete: {processed_count} trajectories processed")
    return processed_count


def calculate_quaternion_trajectory_metrics(trajectory_7d):
    """
    쿼터니언 궤적의 부드러움 메트릭 계산
    
    Args:
        trajectory_7d: [N, 7] 쿼터니언 궤적
    
    Returns:
        dict: 부드러움 메트릭
    """
    if len(trajectory_7d) < 3:
        return {"position_smoothness": 0.0, "rotation_smoothness": 0.0, "total_rotation": 0.0}
    
    # 위치 부분 [x, y, z]
    positions = trajectory_7d[:, :3]
    
    # 쿼터니언 부분 [qw, qx, qy, qz]
    quaternions = trajectory_7d[:, 3:7]
    
    # 위치 스무딩 메트릭 (기존 방식)
    pos_velocities = np.diff(positions, axis=0)
    pos_accelerations = np.diff(pos_velocities, axis=0)
    pos_jerks = np.diff(pos_accelerations, axis=0)
    
    position_smoothness = np.mean(np.linalg.norm(pos_jerks, axis=1)) if len(pos_jerks) > 0 else 0.0
    
    # 회전 스무딩 메트릭 (쿼터니언 각속도 변화)
    rotation_changes = []
    total_rotation = 0.0
    
    for i in range(1, len(quaternions)):
        q1 = quaternions[i-1]
        q2 = quaternions[i]
        
        # 쿼터니언 내적으로 회전 각도 계산
        dot = np.abs(np.dot(q1, q2))
        dot = np.clip(dot, 0.0, 1.0)  # 수치 안정성
        
        angle_change = 2 * np.arccos(dot)
        rotation_changes.append(angle_change)
        total_rotation += angle_change
    
    # 회전 가속도 (각속도 변화율)
    rotation_accelerations = np.diff(rotation_changes) if len(rotation_changes) > 1 else []
    rotation_smoothness = np.std(rotation_accelerations) if len(rotation_accelerations) > 0 else 0.0
    
    metrics = {
        "position_smoothness": position_smoothness,
        "rotation_smoothness": rotation_smoothness,
        "total_rotation": total_rotation,
        "path_length": np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)),
        "num_waypoints": len(trajectory_7d),
        "avg_rotation_per_step": total_rotation / len(rotation_changes) if rotation_changes else 0.0
    }
    
    return metrics


def compare_smoothing_methods(trajectory_6d, num_points=None):
    """
    기존 6D 오일러각 vs 새로운 7D 쿼터니언 스무딩 비교
    
    Args:
        trajectory_6d: [N, 6] 오일러각 궤적 [x,y,z,rx,ry,rz]
        num_points: 출력 포인트 수
    
    Returns:
        dict: 비교 결과
    """
    if len(trajectory_6d) < 2:
        return {}
    
    if num_points is None:
        num_points = len(trajectory_6d) * 2
    
    print(f"🔄 Comparing smoothing methods: {len(trajectory_6d)} → {num_points} waypoints")
    
    # 1. 기존 방식: SE(3) → SE(2) → B-spline → SE(2) → SE(3)
    se2_trajectory = trajectory_6d[:, [0, 1, 5]]  # x, y, rz만 사용
    se2_smoothed = bspline_trajectory_smoother(se2_trajectory, num_points)
    
    # SE(2) → SE(3) 변환
    euler_smoothed = np.zeros((len(se2_smoothed), 6))
    euler_smoothed[:, [0, 1, 5]] = se2_smoothed  # x, y, rz
    # z, rx, ry는 0으로 유지
    
    # 2. 새로운 방식: SE(3) → 쿼터니언 → SLERP B-spline → 쿼터니언 → SE(3)
    trajectory_7d = trajectory_euler_to_quaternion(trajectory_6d)
    quaternion_smoothed_7d = bspline_quaternion_trajectory_smoother(trajectory_7d, num_points)
    quaternion_smoothed_6d = trajectory_quaternion_to_euler(quaternion_smoothed_7d)
    
    # 3. 메트릭 계산
    original_metrics = calculate_smoothness_metrics(trajectory_6d[:, [0, 1, 5]])  # SE(2)
    euler_metrics = calculate_smoothness_metrics(euler_smoothed[:, [0, 1, 5]])
    quat_metrics = calculate_quaternion_trajectory_metrics(quaternion_smoothed_7d)
    
    results = {
        'original_6d': trajectory_6d,
        'euler_smoothed_6d': euler_smoothed,
        'quaternion_smoothed_7d': quaternion_smoothed_7d,
        'quaternion_smoothed_6d': quaternion_smoothed_6d,
        'metrics': {
            'original': original_metrics,
            'euler_smoothed': euler_metrics,
            'quaternion_smoothed': quat_metrics
        },
        'comparison': {
            'euler_improvement': {
                'curvature_variance': ((original_metrics['curvature_variance'] - euler_metrics['curvature_variance']) / original_metrics['curvature_variance'] * 100) if original_metrics['curvature_variance'] > 0 else 0,
                'jerk_norm': ((original_metrics['jerk_norm'] - euler_metrics['jerk_norm']) / original_metrics['jerk_norm'] * 100) if original_metrics['jerk_norm'] > 0 else 0
            },
            'quaternion_advantage': {
                'rotation_smoothness': quat_metrics['rotation_smoothness'],
                'total_rotation': quat_metrics['total_rotation']
            }
        }
    }
    
    return results


def main_quaternion_example():
    """쿼터니언 기반 스무딩 예제"""
    print("🧪 Quaternion B-spline Smoothing Example")
    
    # 테스트 궤적 생성 (간단한 SE(3) 궤적)
    N = 20
    t = np.linspace(0, 2*np.pi, N)
    
    test_trajectory_6d = np.zeros((N, 6))
    test_trajectory_6d[:, 0] = np.cos(t)  # x
    test_trajectory_6d[:, 1] = np.sin(t)  # y
    test_trajectory_6d[:, 2] = t * 0.1    # z (상승)
    test_trajectory_6d[:, 5] = t          # yaw (회전)
    
    print(f"Test trajectory: {N} waypoints")
    
    # 스무딩 방법 비교
    results = compare_smoothing_methods(test_trajectory_6d, num_points=40)
    
    if results:
        print("\n📊 Smoothing Comparison Results:")
        print(f"Original jerk norm: {results['metrics']['original']['jerk_norm']:.4f}")
        print(f"Euler smoothed jerk norm: {results['metrics']['euler_smoothed']['jerk_norm']:.4f}")
        print(f"Quaternion rotation smoothness: {results['metrics']['quaternion_smoothed']['rotation_smoothness']:.4f}")
        
        print(f"\nEuler method improvements:")
        for key, value in results['comparison']['euler_improvement'].items():
            print(f"  {key}: {value:.1f}%")
        
        print(f"\nQuaternion advantages:")
        for key, value in results['comparison']['quaternion_advantage'].items():
            print(f"  {key}: {value:.4f}")
    
    return results


if __name__ == "__main__":
    # 기존 main 또는 새로운 쿼터니언 예제 선택
    print("Choose example:")
    print("1. Traditional SE(2) B-spline smoothing")
    print("2. New quaternion-based SE(3) smoothing")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        main_quaternion_example()
    else:
        main()