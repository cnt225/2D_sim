#!/usr/bin/env python3
"""
통합 Pose 생성 CLI 인터페이스

사용법:
    # 환경별 pose + pose_pair 동시 생성
    python unified_pose_cli.py generate circle_env_000000.ply --rigid-bodies 0,1,2 --num-poses 100 --num-pairs 50

    # 결과 확인  
    python unified_pose_cli.py summary circle_env_000000

    # 전체 HDF5 파일 상태 확인
    python unified_pose_cli.py status

    # 특정 환경-RB 데이터 조회
    python unified_pose_cli.py show circle_env_000000 --rigid-body 0

    # HDF5 파일 초기화
    python unified_pose_cli.py reset
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List

try:
    from .unified_pose_generator import UnifiedPoseGenerator
    from .unified_pose_manager import UnifiedPoseManager
except ImportError:
    from unified_pose_generator import UnifiedPoseGenerator
    from unified_pose_manager import UnifiedPoseManager


def parse_rigid_body_list(rb_string: str) -> List[int]:
    """
    문자열에서 rigid body ID 리스트 파싱
    
    Args:
        rb_string: "0,1,2" 또는 "0" 형태
        
    Returns:
        [0, 1, 2] 형태의 리스트
    """
    try:
        if ',' in rb_string:
            return [int(rb.strip()) for rb in rb_string.split(',')]
        else:
            return [int(rb_string)]
    except ValueError:
        raise ValueError(f"Invalid rigid body list format: {rb_string}. Use format like '0,1,2' or '0'")


def cmd_generate(args):
    """pose + pose_pair 생성 명령"""
    print(f"🚀 Generating poses and pose pairs for {args.env_path}")
    
    try:
        # rigid body 리스트 파싱
        rb_ids = parse_rigid_body_list(args.rigid_bodies)
        
        # 생성기 초기화
        generator = UnifiedPoseGenerator(
            config_file=args.config,
            h5_path=args.output,
            seed=args.seed
        )
        
        # 데이터셋 생성
        result = generator.generate_complete_dataset(
            env_path=args.env_path,
            rb_ids=rb_ids,
            num_poses=args.num_poses,
            num_pairs=args.num_pairs,
            safety_margin=args.safety_margin,
            max_attempts=args.max_attempts
        )
        
        if result['success']:
            print(f"🎉 Successfully generated dataset for {result['env_name']}")
            return 0
        else:
            print(f"❌ Generation completed with errors")
            return 1
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return 1


def cmd_summary(args):
    """특정 환경 요약 정보 출력"""
    try:
        manager = UnifiedPoseManager(args.output)
        
        print(f"📊 Summary for {args.env_name}")
        
        # 환경 존재 확인
        environments = manager.list_environments()
        if args.env_name not in environments:
            print(f"❌ Environment '{args.env_name}' not found")
            print(f"Available environments: {environments}")
            return 1
        
        # rigid body 목록
        rb_ids = manager.list_rigid_bodies(args.env_name)
        print(f"   Rigid bodies: {rb_ids}")
        
        # 각 RB별 상세 정보
        for rb_id in rb_ids:
            poses, pose_meta = manager.get_poses(args.env_name, rb_id)
            pairs, pair_meta = manager.get_pose_pairs(args.env_name, rb_id)
            
            pose_count = len(poses) if poses is not None else 0
            pair_count = len(pairs) if pairs is not None else 0
            
            print(f"   rb_{rb_id}:")
            print(f"     Poses: {pose_count}")
            print(f"     Pose pairs: {pair_count}")
            
            if pose_meta:
                success_rate = pose_meta.get('success_rate', 'N/A')
                creation_time = pose_meta.get('creation_time', 'N/A')
                print(f"     Success rate: {success_rate}%")
                print(f"     Created: {creation_time}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to get summary: {e}")
        return 1


def cmd_status(args):
    """전체 HDF5 파일 상태 확인"""
    try:
        manager = UnifiedPoseManager(args.output)
        summary = manager.get_summary()
        
        print(f"📋 HDF5 File Status: {args.output}")
        print(f"   Creation time: {summary.get('creation_time', 'N/A')}")
        print(f"   Last updated: {summary.get('last_updated', 'N/A')}")
        print(f"   Total environments: {summary.get('environment_count', 0)}")
        
        if 'environment_stats' in summary:
            print(f"\n   Environment breakdown:")
            for env_name, stats in summary['environment_stats'].items():
                print(f"     {env_name}:")
                print(f"       Rigid bodies: {stats['rigid_bodies']}")
                print(f"       Total poses: {stats['total_poses']}")
                print(f"       Total pairs: {stats['total_pairs']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to get status: {e}")
        return 1


def cmd_show(args):
    """특정 환경-RB 데이터 조회"""
    try:
        manager = UnifiedPoseManager(args.output)
        
        print(f"🔍 Showing data for {args.env_name}/rb_{args.rigid_body}")
        
        # pose 데이터 조회
        poses, pose_meta = manager.get_poses(args.env_name, args.rigid_body)
        pairs, pair_meta = manager.get_pose_pairs(args.env_name, args.rigid_body)
        
        if poses is None:
            print(f"❌ No pose data found for {args.env_name}/rb_{args.rigid_body}")
            return 1
        
        print(f"   Poses: {len(poses)} entries")
        print(f"   Pose pairs: {len(pairs) if pairs is not None else 0} entries")
        
        # 샘플 pose 출력
        if len(poses) > 0:
            print(f"   Sample poses:")
            for i, pose in enumerate(poses[:3]):  # 처음 3개만
                print(f"     [{i}]: [x={pose[0]:.2f}, y={pose[1]:.2f}, yaw={pose[5]:.2f}]")
        
        # 메타데이터 출력
        if pose_meta:
            print(f"   Metadata:")
            for key, value in pose_meta.items():
                if key not in ['creation_time']:  # 너무 긴 것들 제외
                    print(f"     {key}: {value}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to show data: {e}")
        return 1


def cmd_reset(args):
    """HDF5 파일 초기화"""
    if os.path.exists(args.output):
        if not args.force:
            confirm = input(f"Are you sure you want to delete {args.output}? (y/N): ")
            if confirm.lower() != 'y':
                print("Cancelled.")
                return 0
        
        os.remove(args.output)
        print(f"✅ Removed {args.output}")
    else:
        print(f"⚠️ File {args.output} does not exist")
    
    return 0


def cmd_visualize(args):
    """pose 시각화 명령"""
    try:
        from utils.pose_vis import HDF5PoseVisualizer
        
        print(f"🎨 Visualizing poses for {args.env_name}/rb_{args.rigid_body}")
        
        visualizer = HDF5PoseVisualizer(args.output)
        
        success = visualizer.visualize_poses_from_hdf5(
            env_name=args.env_name,
            rb_id=args.rigid_body,
            save_image=args.save_image,
            output_file=args.output_file,
            show_plot=not args.no_show,
            max_poses_to_show=args.max_poses,
            show_pairs=args.show_pairs
        )
        
        if success:
            print(f"🎉 Visualization completed")
            return 0
        else:
            print(f"❌ Visualization failed")
            return 1
            
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return 1


def cmd_batch(args):
    """배치 생성 명령"""
    import os
    import time
    from pathlib import Path
    
    try:
        # 환경 파일 목록 가져오기
        pointcloud_root = Path("/home/dhkang225/2D_sim/data/pointcloud")
        env_dir = pointcloud_root / args.env_directory.rstrip('/')
        
        if not env_dir.exists():
            print(f"❌ Environment directory not found: {env_dir}")
            return 1
        
        ply_files = sorted(list(env_dir.glob("*.ply")))
        if not ply_files:
            print(f"❌ No PLY files found in {env_dir}")
            return 1
        
        # 시작점과 최대 파일 수 적용
        if args.start_from > 0:
            ply_files = ply_files[args.start_from:]
            print(f"Starting from index {args.start_from}")
        
        if args.max_files:
            ply_files = ply_files[:args.max_files]
        
        print(f"🚀 Starting batch generation")
        print(f"   Directory: {env_dir}")
        print(f"   Files to process: {len(ply_files)}")
        print(f"   Rigid bodies: {args.rigid_bodies}")
        print(f"   Poses per RB: {args.num_poses}")
        print(f"   Pairs per RB: {args.num_pairs}")
        
        # rigid body 리스트 파싱
        rb_ids = parse_rigid_body_list(args.rigid_bodies)
        
        # 생성기 초기화
        generator = UnifiedPoseGenerator(
            h5_path=args.output,
            seed=args.seed
        )
        
        # 배치 생성 실행
        start_time = time.time()
        success_count = 0
        failed_files = []
        
        for i, ply_file in enumerate(ply_files):
            print(f"\n[{i+1}/{len(ply_files)}] Processing {ply_file.name}...")
            
            try:
                # 환경 경로는 pointcloud 루트 기준 상대경로로 만들기
                relative_path = f"{args.env_directory.rstrip('/')}/{ply_file.name}"
                
                # 파일별로 다른 시드 사용
                file_seed = args.seed + i if args.seed else None
                
                result = generator.generate_complete_dataset(
                    env_path=relative_path,
                    rb_ids=rb_ids,
                    num_poses=args.num_poses,
                    num_pairs=args.num_pairs
                )
                
                if result['success']:
                    success_count += 1
                    print(f"✅ {ply_file.name}: Success")
                else:
                    failed_files.append(ply_file.name)
                    print(f"❌ {ply_file.name}: Failed")
                    
            except Exception as e:
                failed_files.append(ply_file.name)
                print(f"❌ {ply_file.name}: Exception - {e}")
            
            # 진행상황 출력
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = len(ply_files) - (i + 1)
                eta = remaining / rate if rate > 0 else 0
                
                print(f"📊 Progress: {i+1}/{len(ply_files)} ({success_count} success)")
                print(f"   Rate: {rate:.1f} files/sec, ETA: {eta/60:.1f} min")
        
        # 최종 결과
        total_time = time.time() - start_time
        print(f"\n🎉 Batch generation completed!")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Success: {success_count}/{len(ply_files)} ({success_count/len(ply_files)*100:.1f}%)")
        
        if failed_files:
            print(f"   Failed files ({len(failed_files)}):")
            for failed_file in failed_files[:10]:  # 처음 10개만 출력
                print(f"     - {failed_file}")
            if len(failed_files) > 10:
                print(f"     ... and {len(failed_files)-10} more")
        
        return 0 if success_count == len(ply_files) else 1
        
    except Exception as e:
        print(f"❌ Batch generation failed: {e}")
        return 1


def main():
    """메인 CLI 함수"""
    parser = argparse.ArgumentParser(
        description="Unified Pose Generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate poses and pairs for environment
  python unified_pose_cli.py generate circle_env_000000.ply --rigid-bodies 0,1,2 --num-poses 100

  # Check specific environment
  python unified_pose_cli.py summary circle_env_000000

  # View overall status
  python unified_pose_cli.py status

  # Show specific data
  python unified_pose_cli.py show circle_env_000000 --rigid-body 0
        """
    )
    
    # 전역 옵션
    parser.add_argument('--output', type=str, 
                       default="/home/dhkang225/2D_sim/data/pose/unified_poses.h5",
                       help='Output HDF5 file path (default: root/data/pose/unified_poses.h5)')
    
    # 서브커맨드
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # generate 커맨드
    generate_parser = subparsers.add_parser('generate', help='Generate poses and pose pairs')
    generate_parser.add_argument('env_path', type=str, 
                                help='Environment file path (relative to data/pointcloud)')
    generate_parser.add_argument('--rigid-bodies', type=str, required=True,
                                help='Rigid body IDs (e.g., "0,1,2" or "0")')
    generate_parser.add_argument('--num-poses', type=int, default=100,
                                help='Number of poses to generate per rigid body (default: 100)')
    generate_parser.add_argument('--num-pairs', type=int, default=50,
                                help='Number of pose pairs to generate per rigid body (default: 50)')
    generate_parser.add_argument('--safety-margin', type=float, default=0.05,
                                help='Safety margin for collision detection (default: 0.05)')
    generate_parser.add_argument('--max-attempts', type=int, default=1000,
                                help='Maximum attempts per pose (default: 1000)')
    generate_parser.add_argument('--config', type=str, 
                                default="config/rigid_body_configs.yaml",
                                help='Rigid body config file (default: config/rigid_body_configs.yaml)')
    generate_parser.add_argument('--seed', type=int, default=None,
                                help='Random seed for reproducible results')
    
    # summary 커맨드
    summary_parser = subparsers.add_parser('summary', help='Show environment summary')
    summary_parser.add_argument('env_name', type=str, help='Environment name')
    
    # status 커맨드
    status_parser = subparsers.add_parser('status', help='Show overall HDF5 file status')
    
    # show 커맨드
    show_parser = subparsers.add_parser('show', help='Show specific environment-RB data')
    show_parser.add_argument('env_name', type=str, help='Environment name')
    show_parser.add_argument('--rigid-body', type=int, required=True,
                            help='Rigid body ID')
    
    # visualize 커맨드
    visualize_parser = subparsers.add_parser('visualize', help='Visualize poses from HDF5 data')
    visualize_parser.add_argument('env_name', type=str, help='Environment name')
    visualize_parser.add_argument('--rigid-body', type=int, required=True,
                                 help='Rigid body ID')
    visualize_parser.add_argument('--save-image', action='store_true',
                                 help='Save visualization as image')
    visualize_parser.add_argument('--output-file', type=str, default=None,
                                 help='Output image file name')
    visualize_parser.add_argument('--no-show', action='store_true',
                                 help='Do not display the plot')
    visualize_parser.add_argument('--max-poses', type=int, default=20,
                                 help='Maximum number of poses to display (default: 20)')
    visualize_parser.add_argument('--show-pairs', action='store_true',
                                 help='Visualize pose pairs with arrows')
    
    # batch 커맨드
    batch_parser = subparsers.add_parser('batch', help='Batch generate poses for multiple environments')
    batch_parser.add_argument('env_directory', type=str, 
                             help='Directory containing environment files (e.g., circles_only/)')
    batch_parser.add_argument('--rigid-bodies', type=str, required=True,
                             help='Rigid body IDs (e.g., "0,1,2" or "3")')
    batch_parser.add_argument('--num-poses', type=int, default=20,
                             help='Number of poses to generate per rigid body (default: 20)')
    batch_parser.add_argument('--num-pairs', type=int, default=10,
                             help='Number of pose pairs to generate per rigid body (default: 10)')
    batch_parser.add_argument('--start-from', type=int, default=0,
                             help='Start from specific file index (for resuming)')
    batch_parser.add_argument('--max-files', type=int, default=None,
                             help='Maximum number of files to process')
    batch_parser.add_argument('--seed', type=int, default=42,
                             help='Base random seed')
    
    # reset 커맨드
    reset_parser = subparsers.add_parser('reset', help='Reset (delete) HDF5 file')
    reset_parser.add_argument('--force', action='store_true',
                             help='Force reset without confirmation')
    
    # 인수 파싱
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # 커맨드 실행
    if args.command == 'generate':
        return cmd_generate(args)
    elif args.command == 'summary':
        return cmd_summary(args)
    elif args.command == 'status':
        return cmd_status(args)
    elif args.command == 'show':
        return cmd_show(args)
    elif args.command == 'visualize':
        return cmd_visualize(args)
    elif args.command == 'batch':
        return cmd_batch(args)
    elif args.command == 'reset':
        return cmd_reset(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
