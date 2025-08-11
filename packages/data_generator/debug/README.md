# Debug Tools for Data Generator

이 폴더에는 데이터 생성 및 검증을 위한 디버그 도구들이 포함되어 있습니다.

## 📋 도구 목록

### 1. Point Cloud Metadata Validator (`validate_pointcloud.py`)

포인트 클라우드 파일(.ply)과 메타데이터 파일(.json)의 일관성을 검증하는 도구입니다.

#### 🎯 주요 기능

- **메타데이터 검증**: 메타데이터의 `num_points`와 실제 PLY 파일의 포인트 개수 비교
- **불일치 분석**: 차이가 있는 파일들의 상세 분석 제공
- **자동 수정**: 불일치하는 메타데이터 자동 수정 (백업 생성)
- **보고서 생성**: JSON 형식의 상세 검증 보고서 생성

#### 📖 사용법

##### 기본 사용법
```bash
# 도움말 보기
python validate_pointcloud.py -h

# 기본 검증 (결과는 result/default 폴더에 저장)
python validate_pointcloud.py --data-dir /path/to/pointcloud/data

# 특정 프로젝트 폴더로 결과 저장
python validate_pointcloud.py --data-dir /path/to/data --folder v2

# 조용한 모드로 검증
python validate_pointcloud.py --data-dir /path/to/data --folder test --quiet
```

##### 보고서 생성 및 프로젝트 관리
```bash
# 특정 프로젝트(v1)로 검증 후 보고서 저장
python validate_pointcloud.py --data-dir /path/to/data --folder v1 --output my_validation.json

# 여러 프로젝트로 결과 분리 관리
python validate_pointcloud.py --data-dir /path/to/data --folder experiment_1
python validate_pointcloud.py --data-dir /path/to/data --folder experiment_2

# 짧은 옵션 사용
python validate_pointcloud.py --data-dir /path/to/data --folder prod -o report.json
```

##### 메타데이터 수정
```bash
# 수정 미리보기 (실제로 수정하지 않음)
python validate_pointcloud.py --data-dir /path/to/data --folder test --dry-run

# 실제 메타데이터 수정 (백업 파일 자동 생성)
python validate_pointcloud.py --data-dir /path/to/data --folder v1 --fix

# 수정과 동시에 보고서 저장
python validate_pointcloud.py --data-dir /path/to/data --folder v1 --fix -o validation.json
```

##### 고급 옵션
```bash
# 상세 출력할 불일치 파일 수 조정
python validate_pointcloud.py --data-dir /path/to/data --folder analysis --max-details 50

# 조용한 모드로 수정
python validate_pointcloud.py --data-dir /path/to/data --folder v1 --fix --quiet
```

#### 📝 명령행 옵션

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--data-dir` | 포인트 클라우드 파일이 있는 디렉토리 경로 (필수) | `--data-dir /path/to/data` |
| `--folder` | 결과 저장용 프로젝트 폴더명 (기본값: default) | `--folder v1` |
| `--output`, `-o` | 검증 보고서 출력 파일명 (기본값: validation_report.json) | `-o my_report.json` |
| `--fix` | 불일치하는 메타데이터 수정 (백업 생성) | `--fix` |
| `--dry-run` | 수정 미리보기 (실제 수정하지 않음) | `--dry-run` |
| `--quiet`, `-q` | 최소한의 출력만 표시 | `--quiet` |
| `--max-details` | 상세 출력할 최대 불일치 파일 수 | `--max-details 50` |
| `--help`, `-h` | 도움말 표시 | `-h` |

#### 📊 출력 형식

##### 검증 결과 예시
```
🔍 Point Cloud Metadata Validator
==================================================
� Results will be saved to: /path/to/debug/result/v1
�🔍 Validating point cloud files in: /path/to/data
📁 Found 3241 environment files to validate
⚙️  Processing 1/3241: circle_env_000000
...

✅ Validation completed:
   • Total files: 3241
   • Consistent: 2961
   • Inconsistent: 280

🚨 INCONSISTENCY ANALYSIS
Found 280 inconsistent files:

📊 Difference statistics:
   • Difference -20618: 1 files
   • Difference -13866: 1 files
   ...

📝 Detailed inconsistent files (top 20):
   • circle_env_000001: PLY=9698, Meta=11120, Diff=-1422
   • circle_env_000012: PLY=9683, Meta=15670, Diff=-5987
   ...

📄 Validation report saved to: /path/to/debug/result/v1/validation_report.json
```

##### 수정 결과 예시
```
🛠️  FIXING METADATA
⚠️  FIXING MODE - Files will be modified
💾 Creating backups for all modified files...
   • circle_env_000001: 11120 -> 9698
   • circle_env_000012: 15670 -> 9683
   ...

✅ 280 files fixed
💾 Backup files created with .json.backup extension
📄 Validation report saved to: /path/to/debug/result/v1/final_validation_report.json
```

#### 📄 보고서 형식

생성되는 JSON 보고서는 다음과 같은 구조를 가집니다:

```json
{
  "validation_timestamp": "2025-08-11T09:42:54.297195",
  "data_directory": "/path/to/data",
  "total_files": 3241,
  "consistent_files": 3241,
  "inconsistent_files": 0,
  "error_files": 0,
  "results": [
    {
      "env_index": "000000",
      "ply_path": "/path/to/circle_env_000000.ply",
      "meta_path": "/path/to/circle_env_000000_meta.json",
      "ply_exists": true,
      "meta_exists": true,
      "actual_points": 1310,
      "metadata_points": 1310,
      "is_consistent": true,
      "difference": 0,
      "error": null
    }
  ]
}
```

#### 🔄 백업에서 복구

메타데이터를 수정한 후 원본으로 되돌리고 싶다면:

```bash
# 모든 백업 파일을 원본으로 복구
find /path/to/data -name "*.json.backup" -exec sh -c 'mv "$1" "${1%.backup}"' _ {} \;

# 특정 파일만 복구
mv /path/to/circle_env_000001_meta.json.backup /path/to/circle_env_000001_meta.json
```

#### ⚠️ 주의사항

1. **백업**: `--fix` 옵션 사용 시 자동으로 `.json.backup` 파일이 생성됩니다.
2. **권한**: 메타데이터 파일을 수정할 수 있는 권한이 필요합니다.
3. **대용량 데이터**: 수천 개의 파일을 처리할 때 시간이 걸릴 수 있습니다.
4. **경로**: 절대 경로 사용을 권장합니다.

#### 🐛 문제 해결

**Q: "Data directory does not exist" 오류가 발생합니다.**
A: `--data-dir` 옵션에 올바른 경로를 지정했는지 확인하세요. 절대 경로 사용을 권장합니다.

**Q: PLY 파일을 읽을 수 없다는 오류가 발생합니다.**
A: PLY 파일이 올바른 형식인지, 읽기 권한이 있는지 확인하세요.

**Q: 메타데이터 수정이 실패합니다.**
A: 해당 디렉토리에 쓰기 권한이 있는지 확인하세요.

---

### 📚 추가 정보

### 📁 결과 디렉토리 구조

검증 결과는 다음과 같은 구조로 저장됩니다:

```
packages/data_generator/debug/result/
├── v1/                                    # 첫 번째 프로젝트
│   ├── validation_report.json             # 초기 검증 결과
│   ├── final_validation_report.json       # 수정 후 최종 결과 (--fix 사용 시)
│   ├── validation_summary.py              # 요약 스크립트 (이동된 파일)
│   └── validate_pointcloud_metadata.py    # 원본 스크립트 (이동된 파일)
├── v2/                                    # 두 번째 프로젝트
│   ├── validation_report.json
│   └── custom_report.json
├── experiment_1/                          # 실험용 프로젝트
│   └── validation_report.json
└── default/                               # 기본 폴더 (--folder 미지정 시)
    └── validation_report.json
```

### 🔧 프로젝트 관리 방법

#### 여러 실험 결과 관리
```bash
# 실험 1: 원본 데이터 검증
python validate_pointcloud.py --data-dir /path/to/data --folder experiment_1

# 실험 2: 수정된 데이터 검증  
python validate_pointcloud.py --data-dir /path/to/data --folder experiment_2 --fix

# 실험 3: 다른 데이터셋 검증
python validate_pointcloud.py --data-dir /path/to/other_data --folder dataset_2
```

#### 버전 관리
```bash
# 버전별 결과 저장
python validate_pointcloud.py --data-dir /path/to/data --folder v1.0
python validate_pointcloud.py --data-dir /path/to/data --folder v1.1 --fix
python validate_pointcloud.py --data-dir /path/to/data --folder v2.0 --fix
```

#### 결과 비교
```bash
# 수정 전후 비교를 위한 별도 저장
python validate_pointcloud.py --data-dir /path/to/data --folder before_fix
python validate_pointcloud.py --data-dir /path/to/data --folder after_fix --fix
```

### 개발 환경 설정

```bash
# 프로젝트 루트에서
cd packages/data_generator/debug

# Python 가상환경에서 실행
python validate_pointcloud.py --help
```

### 기여하기

버그 발견이나 개선 사항이 있다면 이슈를 생성하거나 풀 리퀘스트를 제출해주세요.

---

*Last updated: 2025-08-11*
