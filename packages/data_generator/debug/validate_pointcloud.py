#!/usr/bin/env python3
"""
Point Cloud Metadata Validator

포인트 클라우드 파일과 메타데이터의 일관성을 검증하는 도구입니다.
- 메타데이터의 num_points와 실제 PLY 파일의 포인트 개수 비교
- 불일치하는 케이스들의 인덱스와 차이 분석  
- 메타데이터 자동 수정 기능 제공

Author: GitHub Copilot & dhkang225
Date: 2025-08-11
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import datetime


class PointCloudValidator:
    """포인트 클라우드 메타데이터 검증 클래스"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.validation_results: List[Dict] = []
        
    def read_ply_point_count(self, ply_path: Path) -> Optional[int]:
        """PLY 파일에서 실제 포인트 개수를 읽어옴
        
        Args:
            ply_path: PLY 파일 경로
            
        Returns:
            포인트 개수 또는 None (오류 시)
        """
        try:
            with open(ply_path, 'r') as f:
                for line in f:
                    if line.startswith('element vertex'):
                        return int(line.split()[2])
            return None
        except Exception as e:
            print(f"Error reading PLY file {ply_path}: {e}")
            return None
    
    def read_metadata_point_count(self, meta_path: Path) -> Optional[int]:
        """메타데이터 JSON 파일에서 num_points를 읽어옴
        
        Args:
            meta_path: 메타데이터 JSON 파일 경로
            
        Returns:
            포인트 개수 또는 None (오류 시)
        """
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                return metadata.get('num_points')
        except Exception as e:
            print(f"Error reading metadata file {meta_path}: {e}")
            return None
    
    def validate_single_file(self, env_index: str) -> Dict:
        """단일 환경 파일의 메타데이터와 실제 포인트 수를 검증
        
        Args:
            env_index: 환경 인덱스 (예: "000571")
            
        Returns:
            검증 결과 딕셔너리
        """
        ply_path = self.data_dir / f"circle_env_{env_index}.ply"
        meta_path = self.data_dir / f"circle_env_{env_index}_meta.json"
        
        result = {
            'env_index': env_index,
            'ply_path': str(ply_path),
            'meta_path': str(meta_path),
            'ply_exists': ply_path.exists(),
            'meta_exists': meta_path.exists(),
            'actual_points': None,
            'metadata_points': None,
            'is_consistent': False,
            'difference': 0,
            'error': None
        }
        
        if not result['ply_exists']:
            result['error'] = "PLY file not found"
            return result
            
        if not result['meta_exists']:
            result['error'] = "Metadata file not found"
            return result
        
        # PLY 파일에서 실제 포인트 개수 읽기
        actual_points = self.read_ply_point_count(ply_path)
        if actual_points is None:
            result['error'] = "Could not read point count from PLY file"
            return result
            
        # 메타데이터에서 포인트 개수 읽기
        metadata_points = self.read_metadata_point_count(meta_path)
        if metadata_points is None:
            result['error'] = "Could not read num_points from metadata"
            return result
        
        result['actual_points'] = actual_points
        result['metadata_points'] = metadata_points
        result['difference'] = actual_points - metadata_points
        result['is_consistent'] = (actual_points == metadata_points)
        
        return result
    
    def validate_all_files(self, verbose: bool = True) -> List[Dict]:
        """모든 포인트 클라우드 파일들을 검증
        
        Args:
            verbose: 상세 출력 여부
            
        Returns:
            검증 결과 리스트
        """
        if verbose:
            print(f"🔍 Validating point cloud files in: {self.data_dir}")
        
        # 모든 PLY 파일 찾기
        ply_files = list(self.data_dir.glob("circle_env_*.ply"))
        ply_files = [f for f in ply_files if not f.name.endswith('.backup')]  # 백업 파일 제외
        
        env_indices = []
        for ply_file in ply_files:
            # 파일명에서 인덱스 추출 (예: circle_env_000571.ply -> 000571)
            match = re.search(r'circle_env_(\d+)\.ply$', ply_file.name)
            if match:
                env_indices.append(match.group(1))
        
        env_indices.sort()
        if verbose:
            print(f"📁 Found {len(env_indices)} environment files to validate")
        
        results = []
        inconsistent_count = 0
        
        for i, env_index in enumerate(env_indices):
            if verbose and i % 100 == 0:
                print(f"⚙️  Processing {i+1}/{len(env_indices)}: circle_env_{env_index}")
            
            result = self.validate_single_file(env_index)
            results.append(result)
            
            if not result['is_consistent'] and result['error'] is None:
                inconsistent_count += 1
        
        self.validation_results = results
        
        if verbose:
            print(f"\n✅ Validation completed:")
            print(f"   • Total files: {len(results)}")
            print(f"   • Consistent: {len(results) - inconsistent_count}")
            print(f"   • Inconsistent: {inconsistent_count}")
        
        return results
    
    def analyze_inconsistencies(self, max_details: int = 20) -> None:
        """불일치 케이스들을 분석하고 출력
        
        Args:
            max_details: 상세 출력할 최대 파일 수
        """
        inconsistent = [r for r in self.validation_results if not r['is_consistent'] and r['error'] is None]
        
        if not inconsistent:
            print("✅ No inconsistencies found!")
            return
        
        print(f"\n🚨 INCONSISTENCY ANALYSIS")
        print(f"Found {len(inconsistent)} inconsistent files:")
        
        # 차이별로 그룹핑
        diff_groups = {}
        for result in inconsistent:
            diff = result['difference']
            if diff not in diff_groups:
                diff_groups[diff] = []
            diff_groups[diff].append(result)
        
        print(f"\n📊 Difference statistics:")
        for diff, items in sorted(diff_groups.items())[:10]:  # 상위 10개만
            print(f"   • Difference {diff:+d}: {len(items)} files")
        if len(diff_groups) > 10:
            print(f"   • ... and {len(diff_groups) - 10} more difference values")
        
        print(f"\n📝 Detailed inconsistent files (top {max_details}):")
        for result in inconsistent[:max_details]:
            print(f"   • circle_env_{result['env_index']}: "
                  f"PLY={result['actual_points']}, "
                  f"Meta={result['metadata_points']}, "
                  f"Diff={result['difference']:+d}")
        
        if len(inconsistent) > max_details:
            print(f"   • ... and {len(inconsistent) - max_details} more")
    
    def save_validation_report(self, output_path: str) -> None:
        """검증 결과를 JSON 파일로 저장
        
        Args:
            output_path: 출력 파일 경로
        """
        # 출력 디렉토리가 없으면 생성
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'validation_timestamp': datetime.datetime.now().isoformat(),
            'data_directory': str(self.data_dir),
            'total_files': len(self.validation_results),
            'consistent_files': len([r for r in self.validation_results if r['is_consistent']]),
            'inconsistent_files': len([r for r in self.validation_results if not r['is_consistent'] and r['error'] is None]),
            'error_files': len([r for r in self.validation_results if r['error'] is not None]),
            'results': self.validation_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Validation report saved to: {output_path}")
    
    def fix_metadata(self, dry_run: bool = True) -> None:
        """불일치하는 메타데이터를 수정
        
        Args:
            dry_run: True면 실제 수정하지 않고 미리보기만
        """
        inconsistent = [r for r in self.validation_results if not r['is_consistent'] and r['error'] is None]
        
        if not inconsistent:
            print("✅ No inconsistencies to fix!")
            return
        
        print(f"\n🛠️  FIXING METADATA")
        if dry_run:
            print("🔍 DRY RUN MODE - No files will be modified")
        else:
            print("⚠️  FIXING MODE - Files will be modified")
            print("💾 Creating backups for all modified files...")
        
        fixed_count = 0
        for result in inconsistent:
            meta_path = Path(result['meta_path'])
            
            try:
                # 메타데이터 읽기
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                
                # num_points 업데이트
                old_value = metadata['num_points']
                new_value = result['actual_points']
                metadata['num_points'] = new_value
                
                print(f"   • circle_env_{result['env_index']}: {old_value} -> {new_value}")
                
                if not dry_run:
                    # 백업 생성
                    backup_path = meta_path.with_suffix('.json.backup')
                    if not backup_path.exists():
                        with open(backup_path, 'w') as f:
                            json.dump(json.load(open(meta_path)), f, indent=2)
                    
                    # 수정된 메타데이터 저장
                    with open(meta_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                
                fixed_count += 1
                
            except Exception as e:
                print(f"❌ Error fixing {meta_path}: {e}")
        
        action = "would be fixed" if dry_run else "fixed"
        print(f"\n✅ {fixed_count} files {action}")
        
        if not dry_run:
            print("💾 Backup files created with .json.backup extension")
            print("🔄 To restore from backup: find /path -name '*.json.backup' -exec sh -c 'mv \"$1\" \"${1%.backup}\"' _ {} \\;")


def main():
    parser = argparse.ArgumentParser(
        description='Point Cloud Metadata Validator - 포인트 클라우드 메타데이터 일관성 검증 도구',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
사용 예시:
  # 기본 검증 (결과는 result/default 폴더에 저장)
  python validate_pointcloud.py --data-dir /path/to/data

  # 특정 프로젝트 폴더로 결과 저장
  python validate_pointcloud.py --data-dir /path/to/data --folder v2

  # 검증 후 보고서 저장 (custom 폴더)
  python validate_pointcloud.py --data-dir /path/to/data --folder custom -o my_report.json

  # 수정 미리보기 (실제 수정하지 않음)
  python validate_pointcloud.py --data-dir /path/to/data --folder test --dry-run

  # 실제 메타데이터 수정 (백업 생성됨)
  python validate_pointcloud.py --data-dir /path/to/data --folder v1 --fix

자세한 사용법은 README.md를 참조하세요.
        ''')
    
    parser.add_argument('--data-dir', 
                       required=True,
                       help='포인트 클라우드 파일이 있는 디렉토리 경로')
    parser.add_argument('--folder', 
                       default='default',
                       help='결과 저장용 프로젝트 폴더명 (기본값: default)')
    parser.add_argument('--output', '-o', 
                       default='validation_report.json',
                       help='검증 보고서 출력 파일명 (기본값: validation_report.json)')
    parser.add_argument('--fix', action='store_true',
                       help='불일치하는 메타데이터 수정 (백업 파일 생성)')
    parser.add_argument('--dry-run', action='store_true',
                       help='수정 미리보기 (실제로 수정하지 않음)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='최소한의 출력만 표시')
    parser.add_argument('--max-details', type=int, default=20,
                       help='상세 출력할 최대 불일치 파일 수 (기본값: 20)')
    
    args = parser.parse_args()
    
    # 데이터 디렉토리 확인
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Error: Data directory does not exist: {data_dir}")
        return 1
    
    # 결과 저장 경로 설정
    script_dir = Path(__file__).parent
    result_dir = script_dir / "result" / args.folder
    output_path = result_dir / args.output
    
    if not args.quiet:
        print("🔍 Point Cloud Metadata Validator")
        print("=" * 50)
        print(f"📁 Results will be saved to: {result_dir}")
    
    # 검증기 초기화
    validator = PointCloudValidator(data_dir)
    
    # 검증 실행
    results = validator.validate_all_files(verbose=not args.quiet)
    
    # 분석 결과 출력
    validator.analyze_inconsistencies(max_details=args.max_details)
    
    # 보고서 저장
    validator.save_validation_report(str(output_path))
    
    # 수정 옵션 처리
    if args.fix or args.dry_run:
        validator.fix_metadata(dry_run=args.dry_run)
        
        # 수정 후 다시 검증한 결과도 저장
        if not args.dry_run:
            final_output_path = result_dir / f"final_{args.output}"
            validator.validation_results = []  # 결과 초기화
            validator.validate_all_files(verbose=False)  # 다시 검증
            validator.save_validation_report(str(final_output_path))
    
    if not args.quiet:
        print("\n" + "=" * 50)
        print("🎉 Validation completed successfully!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
