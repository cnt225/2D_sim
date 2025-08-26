#!/usr/bin/env python3
"""
기존 생성된 PLY 파일들을 검증하는 스크립트
개선된 검증 로직을 사용하여 파일 무결성 확인
"""

import os
import json
import sys
from typing import Dict, List, Any
from pathlib import Path


def validate_ply_file(ply_path: str, expected_points: int = None) -> Dict[str, Any]:
    """PLY 파일 검증 (개선된 버전)"""
    result = {
        'file_path': ply_path,
        'valid': False,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        if not os.path.exists(ply_path):
            result['errors'].append('File does not exist')
            return result
            
        with open(ply_path, 'r') as f:
            lines = f.readlines()
        
        result['stats']['total_lines'] = len(lines)
        
        # 기본 구조 검증
        if len(lines) < 8:  # 최소 헤더 라인 수
            result['errors'].append(f'Too few lines: {len(lines)} < 8')
            return result
            
        if not lines[0].strip() == 'ply':
            result['errors'].append(f'Invalid PLY header: {lines[0].strip()}')
            return result
        
        if not lines[1].strip() == 'format ascii 1.0':
            result['errors'].append(f'Invalid format line: {lines[1].strip()}')
            return result
        
        # 포인트 수 검증 (element vertex 라인과 end_header 라인 찾기)
        element_vertex_line = None
        element_vertex_line_idx = None
        header_end_line_idx = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('element vertex'):
                element_vertex_line = stripped
                element_vertex_line_idx = i
            elif stripped == 'end_header':
                header_end_line_idx = i
                break
        
        if element_vertex_line is None:
            result['errors'].append('No "element vertex" line found in header')
            return result
            
        if header_end_line_idx is None:
            result['errors'].append('No "end_header" line found')
            return result
            
        try:
            header_points = int(element_vertex_line.split()[-1])
            result['stats']['header_points'] = header_points
        except (IndexError, ValueError) as e:
            result['errors'].append(f'Cannot parse vertex count: {e}')
            return result
        
        # 실제 데이터 라인 수 확인 (헤더 라인 제외)
        header_lines = header_end_line_idx + 1  # end_header 다음부터 데이터
        actual_data_lines = len(lines) - header_lines
        result['stats']['actual_data_lines'] = actual_data_lines
        result['stats']['header_lines'] = header_lines
        
        # 포인트 수 일치 확인
        if expected_points is not None:
            result['stats']['expected_points'] = expected_points
            if header_points != expected_points:
                result['errors'].append(f'Header mismatch: header={header_points}, expected={expected_points}')
            
            if actual_data_lines != expected_points:
                result['errors'].append(f'Data mismatch: data_lines={actual_data_lines}, expected={expected_points}')
        
        if header_points != actual_data_lines:
            result['errors'].append(f'Header vs data mismatch: header={header_points}, data={actual_data_lines}')
        
        # 마지막 라인이 완전한지 확인
        if actual_data_lines > 0:
            last_line = lines[-1].strip()
            if last_line:
                parts = last_line.split()
                result['stats']['last_line_parts'] = len(parts)
                result['stats']['last_line'] = last_line
                
                if len(parts) != 3:  # x, y, z 좌표
                    result['errors'].append(f'Incomplete last line: {len(parts)} parts instead of 3')
                else:
                    # 좌표가 숫자인지 확인
                    try:
                        for i, part in enumerate(parts):
                            float(part)
                    except ValueError:
                        result['errors'].append(f'Invalid coordinates in last line: {last_line}')
            else:
                result['warnings'].append('Last line is empty')
        
        # 파일 크기 확인
        file_size = os.path.getsize(ply_path)
        result['stats']['file_size'] = file_size
        
        # 256KB 제한 확인
        if file_size == 262143:  # 256KB - 1
            result['warnings'].append('File size exactly 262143 bytes (256KB-1) - likely truncated')
        elif file_size == 262144:  # 256KB
            result['warnings'].append('File size exactly 262144 bytes (256KB) - boundary case')
        
        # 모든 검증 통과하면 valid = True
        if not result['errors']:
            result['valid'] = True
        
        return result
        
    except Exception as e:
        result['errors'].append(f'Validation error: {e}')
        return result


def load_metadata(meta_path: str) -> Dict[str, Any]:
    """메타데이터 JSON 파일 로드"""
    try:
        with open(meta_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {'error': str(e)}


def validate_directory(data_dir: str, max_files: int = None) -> Dict[str, Any]:
    """디렉토리 내 모든 PLY 파일 검증"""
    
    print(f"🔍 Validating PLY files in: {data_dir}")
    
    ply_files = []
    for file in os.listdir(data_dir):
        if file.endswith('.ply'):
            ply_files.append(file)
    
    if max_files:
        ply_files = ply_files[:max_files]
    
    print(f"📁 Found {len(ply_files)} PLY files to validate")
    
    results = {
        'summary': {
            'total_files': len(ply_files),
            'valid_files': 0,
            'invalid_files': 0,
            'truncated_files': 0,
            'size_262143_files': 0
        },
        'files': []
    }
    
    for i, ply_file in enumerate(ply_files):
        ply_path = os.path.join(data_dir, ply_file)
        meta_path = ply_path.replace('.ply', '_meta.json')
        
        # 메타데이터에서 예상 포인트 수 가져오기
        expected_points = None
        if os.path.exists(meta_path):
            metadata = load_metadata(meta_path)
            expected_points = metadata.get('num_points')
        
        # PLY 파일 검증
        validation_result = validate_ply_file(ply_path, expected_points)
        validation_result['metadata_available'] = os.path.exists(meta_path)
        validation_result['metadata'] = metadata if 'metadata' in locals() else None
        
        results['files'].append(validation_result)
        
        # 통계 업데이트
        if validation_result['valid']:
            results['summary']['valid_files'] += 1
        else:
            results['summary']['invalid_files'] += 1
        
        # 크기 통계
        file_size = validation_result['stats'].get('file_size', 0)
        if file_size == 262143:
            results['summary']['size_262143_files'] += 1
        
        # 잘림 파일 감지
        if any('truncated' in warning.lower() for warning in validation_result['warnings']):
            results['summary']['truncated_files'] += 1
        
        # 진행률 표시
        if (i + 1) % 100 == 0 or (i + 1) == len(ply_files):
            print(f"Progress: {i + 1}/{len(ply_files)} ({(i + 1)/len(ply_files)*100:.1f}%)")
    
    return results


def main():
    """메인 실행 함수"""
    
    if len(sys.argv) < 2:
        data_dir = "/home/dhkang225/2D_sim/data/pointcloud/circle_envs_10k/circle_envs_10k"
        max_files = 1000  # 기본값: 1000개만 검증
    else:
        data_dir = sys.argv[1]
        max_files = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    print(f"🚀 Starting PLY file validation")
    print(f"📂 Directory: {data_dir}")
    print(f"📊 Max files: {max_files or 'All'}")
    print()
    
    # 검증 실행
    results = validate_directory(data_dir, max_files)
    
    # 결과 출력
    summary = results['summary']
    print(f"\n📋 === Validation Results ===")
    print(f"Total files: {summary['total_files']}")
    print(f"✅ Valid files: {summary['valid_files']} ({summary['valid_files']/summary['total_files']*100:.1f}%)")
    print(f"❌ Invalid files: {summary['invalid_files']} ({summary['invalid_files']/summary['total_files']*100:.1f}%)")
    print(f"⚠️  256KB-1 files: {summary['size_262143_files']} ({summary['size_262143_files']/summary['total_files']*100:.1f}%)")
    
    # 에러 파일들 상세 출력
    error_files = [f for f in results['files'] if not f['valid']]
    if error_files:
        print(f"\n❌ Files with errors ({len(error_files)}):")
        for i, error_file in enumerate(error_files[:10]):  # 처음 10개만
            file_name = os.path.basename(error_file['file_path'])
            errors = "; ".join(error_file['errors'])
            print(f"  {i+1}. {file_name}: {errors}")
        
        if len(error_files) > 10:
            print(f"  ... and {len(error_files) - 10} more")
    
    # 256KB 제한 파일들
    truncated_files = [f for f in results['files'] 
                      if f['stats'].get('file_size') == 262143]
    if truncated_files:
        print(f"\n⚠️  Files exactly 262143 bytes (likely truncated):")
        for i, trunc_file in enumerate(truncated_files[:5]):  # 처음 5개만
            file_name = os.path.basename(trunc_file['file_path'])
            stats = trunc_file['stats']
            print(f"  {i+1}. {file_name}: {stats.get('header_points', 'N/A')} points expected, "
                  f"{stats.get('actual_data_lines', 'N/A')} lines found")
        
        if len(truncated_files) > 5:
            print(f"  ... and {len(truncated_files) - 5} more")
    
    print(f"\n✨ Validation complete!")


if __name__ == "__main__":
    main()
