# Track 2 Motion Planning Data Requirements

## 🎯 현재 구현 상태

### ✅ 완성된 부분 (Track 2 완료!)
- **JSON + H5 trajectory format** 완전 지원 ✨
- **PLY + OBJ pointcloud format** 완전 지원 ✨  
- **Auto-format detection** 구현 완료 ✨
- **Motion-specific RCFM model** 구현 완료
- **SE(3) ↔ Twist conversion utilities** 구현 완료
- **CFM 호환 듀얼 데이터 로더** 구현 완료
- **fm-main 호환성** 완전 보장

### ✅ 추가 완성된 부분
- **H5 trajectory format** ✅ 구현완료
- **OBJ pointcloud format** ✅ 구현완료 
- **Dual format auto-detection** ✅ 구현완료

---

## 📁 데이터 형식 요구사항

### 1. 궤적 데이터 (Trajectory Data)

#### A. JSON Format ✅ (구현완료)
**파일 경로**: `trajectory_root/*.json`

**내부 구조**:
```json
{
  "environment": {
    "name": "circle_env_001"
  },
  "pair_id": 12345,
  "path": {
    "timestamps": [0.0, 0.1, 0.2, 0.3, ...],
    "data": [
      [x1, y1, z1, rx1, ry1, rz1],  // 6DOF pose at t=0.0
      [x2, y2, z2, rx2, ry2, rz2],  // 6DOF pose at t=0.1
      ...
    ]
  }
}
```

**지원 파일명 패턴**:
- `*_bsplined.json` (B-spline 보간된 궤적)
- `*_traj_rb3.json` (원본 궤적)

#### B. H5 Format ✅ (구현완료)
**지원 구조** (fm-main 호환):
```python
# H5 파일 내부 구조
{
  'poses': np.array([[4x4_matrix1], [4x4_matrix2], ...]),  # SE(3) matrices
  'timestamps': np.array([t1, t2, t3, ...]),               # timestamps
}
# 또는 그룹 구조
{
  'trajectory': {
    'poses': np.array(...),
    'timestamps': np.array(...)
  }
}
```

**자동 감지**: JSON 파일 우선, H5 파일 보조로 로딩

### 2. 포인트클라우드 데이터 (Point Cloud Data)

#### A. PLY Format ✅ (구현완료)
**파일 경로**: `pointcloud_root/{env_name}.ply`

**내부 구조**:
- Vertices: 3D 점들 (x, y, z)
- Open3D로 로딩: `o3d.io.read_triangle_mesh()`
- 자동 리샘플링: `num_point_cloud` 개수로 조정
- Fallback: 파일 손상시 원형 환경 생성

#### B. OBJ Format ✅ (구현완료)  
**파일 경로**: `pointcloud_root/{env_name}.obj`

**지원 구조** (fm-main 호환):
```obj
v 1.0 2.0 3.0
v 4.0 5.0 6.0  
v 7.0 8.0 9.0
# faces, materials 등은 무시
```

**파싱 방식**: 
- `v x y z` 라인만 추출하여 vertices 배열 생성
- Face, texture 정보 무시
- **자동 감지**: PLY 우선, OBJ 보조로 로딩

---

## 🔧 현재 구현된 시스템

### 듀얼 형식 데이터 로더 (`Motion_dataset_rcfm_ply_used.py`)

```python
MotionDataset4RCFM(
    trajectory_root="/path/to/trajectories",     # JSON + H5 파일 혼재
    pointcloud_root="/path/to/pointclouds",     # PLY + OBJ 파일 혼재
    split='train',
    max_trajectories=3241,
    use_bsplined=True,                          # JSON 파일 선택
    num_point_cloud=2000,                       # PC 포인트 수
    num_twists=1000,                            # CFM 샘플 수  
    scale=1.0,
    augmentation=True
)
```

**자동 형식 감지**:
- Trajectory: `*.json` → `*.h5` 순서로 탐색  
- Pointcloud: `{env_id}.ply` → `{env_id}.obj` 순서로 탐색
- 형식 정보 메타데이터에 저장

### 출력 데이터 형식
```python
{
    'pc': torch.FloatTensor([2000, 3]),         # 포인트클라우드
    'Ts_grasp': torch.FloatTensor([1000, 4, 4]), # Twist→SE(3) 변환
    'target_poses': torch.FloatTensor([1000, 4, 4]), # 목표 포즈들
    'env_id': str                                # 환경 ID
}
```

### 궤적 파서 (`Trajectory_parser_rcfm.py`)

```python
class TrajectoryParser:
    def __init__(self, trajectory_file)
    
    # 주요 메서드
    def _parse_waypoints()          # JSON → SE(3) waypoints
    def _compute_twist_vectors()    # 연속 waypoint → twist vectors
    def _compute_se3_twist()        # SE(3) 간 twist 계산 (body frame)
    def validate_data()             # 데이터 유효성 검증
    def get_statistics()           # 궤적 통계 정보
```

---

## 🏗️ 아키텍처 비교

### fm-main 대비 변경사항

#### 유지된 부분 ✅
- **DGCNN**: 포인트클라우드 특징 추출 (`emb_dims: 1024 → 2048 after pooling`)
- **CFM 구조**: `x_t`, `u_t`, `x_1` 보간 방식
- **ODE Solver**: RK4 통합 (`n_steps: 20`)
- **Lie Group Utils**: SE(3)/so(3) 수학적 기반
- **Training Loop**: CFM loss, optimizer 구조

#### 변경된 부분 🔄
- **Data Format**: 
  - fm-main: OBJ + H5 → Track2: PLY + JSON
  - Key명: `Ts_grasp` (grasp 호환성 유지)
- **Velocity Field**: 
  - fm-main: `vf_FC_vec_grasp(in=13, out=6)` 
  - Track2: `vf_FC_vec_motion(in=25, out=6)`
  - Input: `current(12) + target(12) + time(1) = 25D`
- **Target Conditioning**:
  - fm-main: grasp type + shape features
  - Track2: target pose + pointcloud features

#### 새로 추가된 부분 ➕
- **Motion Utils** (`utils/motion_utils.py`):
  - Twist ↔ SE(3) 변환 함수들
  - 궤적 적분/미분 유틸리티
  - CFM 특화 통합 함수
- **MotionRCFM** (`models/motion_rcfm.py`):
  - 궤적 생성 특화 CFM 모델
  - 목표 조건부 속도 필드
  - 모션 특화 초기화 분포

---

## 📊 설정 파일 (`configs/motion_rcfm.yml`)

### 데이터 설정
```yaml
data:
  train:
    dataset: Motion_dataset_rcfm_ply_used    # 전용 데이터로더
    trajectory_root: "/path/to/trajectories"  # JSON 파일들
    pointcloud_root: "/path/to/pointclouds"   # PLY 파일들
    use_bsplined: true                        # JSON 파일 선택
    num_twists: 1000                         # CFM 샘플 개수
```

### 모델 설정  
```yaml
model:
  arch: motion_rcfm                          # 전용 RCFM 모델
  velocity_field:
    arch: vf_fc_vec_motion                   # 모션 속도 필드
    in_dim: 25                               # current+target+time
    out_dim: 6                               # SE(3) twist
```

---

## 🎉 Track 2 완전 구현 완료!

### ✅ 구현된 기능들

#### 1. H5 Trajectory Support ✅
```python
def _load_h5_trajectory(self, h5_file):
    """H5 궤적 파일 로딩 (fm-main 호환)"""
    with h5py.File(h5_file, 'r') as f:
        if 'poses' in f and 'timestamps' in f:
            poses = f['poses'][:]           # [N, 4, 4] SE(3) matrices
            timestamps = f['timestamps'][:] # [N] timestamps
        elif 'trajectory' in f:
            poses = f['trajectory']['poses'][:]
            timestamps = f['trajectory']['timestamps'][:]
        return {'poses': poses, 'timestamps': timestamps, 'env_id': env_id}
```

#### 2. OBJ Pointcloud Support ✅
```python
def _load_obj_pointcloud(self, obj_file):
    """OBJ 파일에서 포인트클라우드 로딩 (fm-main 호환)"""
    vertices = []
    with open(obj_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
    return np.array(vertices, dtype=np.float32)
```

#### 3. Auto-Format Detection ✅
```python
def _detect_pointcloud_file(self, env_id):
    """포인트클라우드 파일 자동 감지 (PLY -> OBJ 순서)"""
    ply_file = os.path.join(self.pointcloud_root, f"{env_id}.ply")
    if os.path.exists(ply_file):
        return ply_file, 'ply'
    
    obj_file = os.path.join(self.pointcloud_root, f"{env_id}.obj")
    if os.path.exists(obj_file):
        return obj_file, 'obj'
    
    return None, None
```

#### 4. 통합 로딩 시스템 ✅
- JSON + H5 trajectory 동시 지원
- PLY + OBJ pointcloud 동시 지원  
- 자동 형식 감지 및 fallback 메커니즘
- 메타데이터에 형식 정보 저장

---

## 💡 사용 방법

### 듀얼 형식으로 학습
```bash
# JSON + H5 궤적, PLY + OBJ 포인트클라우드 혼재 환경에서 학습
python train.py --config configs/motion_rcfm.yml
```

### 지원되는 데이터 구조
```
data/
├── trajectories/                    # 궤적 데이터 (혼재 가능)
│   ├── circle_env_001_bsplined.json # JSON 형식
│   ├── circle_env_002.h5           # H5 형식 (fm-main 호환)
│   ├── circle_env_003_traj_rb3.json
│   └── ...
└── pointclouds/                     # 포인트클라우드 데이터 (혼재 가능)
    ├── circle_env_001.ply          # PLY 형식 
    ├── circle_env_002.obj          # OBJ 형식 (fm-main 호환)
    ├── circle_env_003.ply
    └── ...
```

**자동 매칭**: `env_id` 기반으로 trajectory-pointcloud 파일 자동 매칭

---

## ⚠️ 주의사항

1. **형식 호환성**: JSON+H5 궤적, PLY+OBJ 포인트클라우드 완전 지원 ✅
2. **데이터 매칭**: env_name 기반 trajectory-pointcloud 자동 매칭 필수
3. **우선순위**: JSON > H5, PLY > OBJ 순서로 탐색 및 로딩
4. **메모리 사용**: 큰 H5 궤적 데이터시 배치 크기 조절 필요
5. **좌표계**: Body frame twist 계산 (fm-main 호환성 보장)
6. **Fallback**: 모든 로딩 실패시 원형 환경 자동 생성
7. **H5 구조**: `poses` + `timestamps` 또는 `trajectory` 그룹 모두 지원
8. **OBJ 파싱**: vertex 라인(`v x y z`)만 추출, face/texture 무시

## 🎊 Track 2 구현 상태: **완료** 

fm-main 베이스에서 motion planning용으로 완전히 변환된 Track 2가 구현 완료되었습니다!