"""
B-spline 궤적 스무딩 구현
SE(2) 매니폴드 상에서 RRT 궤적을 부드럽게 스무딩
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.spatial.distance import cdist
from pathlib import Path


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


if __name__ == "__main__":
    main()