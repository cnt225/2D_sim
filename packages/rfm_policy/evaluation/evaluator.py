"""
SE(3) RFM Model Evaluation System

Comprehensive evaluation system for SE(3) Riemannian Flow Matching models
including trajectory generation, collision detection, and performance metrics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
import json

from ..utils.se3_utils import SE3Utils
from ..models.se3_rfm import SE3RFM


class SE3RFMEvaluator:
    """
    Comprehensive evaluator for SE(3) RFM models
    
    Provides trajectory generation, collision detection, performance metrics,
    and visualization capabilities.
    """
    
    def __init__(
        self,
        model: SE3RFM,
        device: torch.device,
        collision_threshold: float = 0.05
    ):
        self.model = model
        self.device = device
        self.collision_threshold = collision_threshold
        self.se3_utils = SE3Utils()
        
    def evaluate_model(
        self,
        test_data: List[Dict[str, torch.Tensor]],
        ode_steps: List[int] = [10, 20, 50],
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            test_data: List of test samples
            ode_steps: Different ODE step sizes to test
            save_dir: Directory to save results
            
        Returns:
            results: Comprehensive evaluation results
        """
        self.model.eval()
        results = {
            'trajectory_metrics': {},
            'performance_metrics': {},
            'collision_analysis': {},
            'generated_trajectories': []
        }
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        with torch.no_grad():
            for step_size in ode_steps:
                step_results = self._evaluate_with_step_size(
                    test_data, step_size, save_dir
                )
                results['trajectory_metrics'][f'steps_{step_size}'] = step_results
        
        # Aggregate results
        results['summary'] = self._compute_summary_metrics(results)
        
        # Save results
        if save_dir:
            with open(save_dir / 'evaluation_results.json', 'w') as f:
                # Convert tensors to lists for JSON serialization
                json_results = self._tensordict_to_json(results)
                json.dump(json_results, f, indent=2)
        
        return results
    
    def _evaluate_with_step_size(
        self,
        test_data: List[Dict[str, torch.Tensor]],
        n_steps: int,
        save_dir: Optional[Path]
    ) -> Dict[str, Any]:
        """Evaluate model with specific ODE step size"""
        
        step_results = {
            'success_rates': [],
            'path_lengths': [],
            'smoothness_scores': [],
            'efficiency_scores': [],
            'computation_times': [],
            'nfe_counts': [],
            'collision_counts': []
        }
        
        for i, sample in enumerate(test_data):
            # Move sample to device
            sample = {k: v.to(self.device) if torch.is_tensor(v) else v 
                     for k, v in sample.items()}
            
            # Generate trajectory
            start_time = time.time()
            
            trajectory, times = self.model.generate_trajectory(
                sample['start_pose'],
                sample['goal_pose'], 
                sample['point_cloud'],
                sample['geometry'],
                n_steps=n_steps
            )
            
            computation_time = time.time() - start_time
            nfe_count = self.model.get_nfe()
            
            # Evaluate trajectory
            metrics = self._evaluate_trajectory(
                trajectory, sample['point_cloud'], sample['geometry'],
                sample['start_pose'], sample['goal_pose']
            )
            
            # Store results
            step_results['success_rates'].append(metrics['is_collision_free'])
            step_results['path_lengths'].append(metrics['path_length'])
            step_results['smoothness_scores'].append(metrics['smoothness_score'])
            step_results['efficiency_scores'].append(metrics['efficiency_score'])
            step_results['computation_times'].append(computation_time)
            step_results['nfe_counts'].append(nfe_count)
            step_results['collision_counts'].append(metrics['collision_count'])
            
            # Save visualization
            if save_dir and i < 10:  # Save first 10 visualizations
                vis_path = save_dir / f'trajectory_{i}_steps_{n_steps}.png'
                self.visualize_trajectory(trajectory, sample['point_cloud'], str(vis_path))
        
        # Compute averages
        for key in step_results:
            if step_results[key]:  # Check if list is not empty
                step_results[f'{key}_mean'] = np.mean(step_results[key])
                step_results[f'{key}_std'] = np.std(step_results[key])
        
        return step_results
    
    def _evaluate_trajectory(
        self,
        trajectory: torch.Tensor,      # [n_steps, 4, 4]
        point_cloud: torch.Tensor,     # [n_points, 3]
        geometry: torch.Tensor,        # [3]
        start_pose: torch.Tensor,      # [4, 4]
        goal_pose: torch.Tensor        # [4, 4]
    ) -> Dict[str, float]:
        """Evaluate single trajectory"""
        
        # Collision detection
        collision_count, is_collision_free = self._check_trajectory_collisions(
            trajectory, point_cloud, geometry
        )
        
        # Path length
        path_length = self._compute_path_length(trajectory)
        
        # Smoothness (based on velocity and acceleration variations)
        smoothness_score = self._compute_smoothness(trajectory)
        
        # Efficiency (compare to straight-line distance)
        straight_line_distance = self._compute_straight_line_distance(start_pose, goal_pose)
        efficiency_score = straight_line_distance / path_length if path_length > 0 else 0
        
        return {
            'collision_count': collision_count,
            'is_collision_free': is_collision_free,
            'path_length': path_length,
            'smoothness_score': smoothness_score,
            'efficiency_score': efficiency_score
        }
    
    def _check_trajectory_collisions(
        self,
        trajectory: torch.Tensor,
        point_cloud: torch.Tensor,
        geometry: torch.Tensor
    ) -> Tuple[int, bool]:
        """Check trajectory for collisions"""
        collision_count = 0
        
        a, b, c = geometry[0].item(), geometry[1].item(), geometry[2].item()
        
        for pose in trajectory:
            # Extract position and rotation
            R = pose[:3, :3]
            t = pose[:3, 3]
            
            # Transform point cloud to robot frame
            points_robot = torch.matmul(point_cloud - t.unsqueeze(0), R)
            
            # Check ellipsoid collision
            ellipsoid_values = (
                (points_robot[:, 0] / (a + self.collision_threshold))**2 +
                (points_robot[:, 1] / (b + self.collision_threshold))**2 +
                (points_robot[:, 2] / (c + self.collision_threshold))**2
            )
            
            # Count collisions
            collisions = (ellipsoid_values < 1.0).sum().item()
            collision_count += collisions
        
        is_collision_free = collision_count == 0
        return collision_count, is_collision_free
    
    def _compute_path_length(self, trajectory: torch.Tensor) -> float:
        """Compute total path length"""
        path_length = 0.0
        
        for i in range(len(trajectory) - 1):
            # Compute geodesic distance between consecutive poses
            T1, T2 = trajectory[i], trajectory[i + 1]
            T_rel = self.se3_utils.compose_se3(self.se3_utils.inverse_se3(T1), T2)
            twist = self.se3_utils.log_se3(T_rel)
            distance = torch.norm(twist).item()
            path_length += distance
        
        return path_length
    
    def _compute_smoothness(self, trajectory: torch.Tensor) -> float:
        """Compute trajectory smoothness score"""
        if len(trajectory) < 3:
            return 1.0
        
        # Compute velocity and acceleration
        velocities = []
        for i in range(len(trajectory) - 1):
            T1, T2 = trajectory[i], trajectory[i + 1]
            T_rel = self.se3_utils.compose_se3(self.se3_utils.inverse_se3(T1), T2)
            velocity = self.se3_utils.log_se3(T_rel)
            velocities.append(velocity)
        
        velocities = torch.stack(velocities)  # [n-1, 6]
        
        # Compute acceleration (velocity differences)
        accelerations = torch.diff(velocities, dim=0)  # [n-2, 6]
        
        # Smoothness score (lower acceleration variance = smoother)
        acc_variance = torch.var(accelerations, dim=0).sum().item()
        smoothness_score = 1.0 / (1.0 + acc_variance)  # Higher = smoother
        
        return smoothness_score
    
    def _compute_straight_line_distance(
        self,
        start_pose: torch.Tensor,
        goal_pose: torch.Tensor
    ) -> float:
        """Compute straight-line geodesic distance"""
        T_rel = self.se3_utils.compose_se3(
            self.se3_utils.inverse_se3(start_pose), goal_pose
        )
        twist = self.se3_utils.log_se3(T_rel)
        return torch.norm(twist).item()
    
    def visualize_trajectory(
        self,
        trajectory: torch.Tensor,      # [n_steps, 4, 4]
        point_cloud: torch.Tensor,     # [n_points, 3]
        save_path: str = None,
        show_plot: bool = False
    ):
        """Visualize trajectory with environment"""
        
        # Extract trajectory positions
        positions = trajectory[:, :3, 3].cpu().numpy()  # [n_steps, 3]
        point_cloud_np = point_cloud.cpu().numpy()      # [n_points, 3]
        
        # Create figure
        fig = plt.figure(figsize=(15, 5))
        
        # 3D plot
        ax1 = fig.add_subplot(131, projection='3d')
        
        # Plot obstacles
        ax1.scatter(point_cloud_np[:, 0], point_cloud_np[:, 1], point_cloud_np[:, 2],
                   c='red', s=1, alpha=0.6, label='Obstacles')
        
        # Plot trajectory
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                'b-', linewidth=3, alpha=0.8, label='Trajectory')
        
        # Plot start and goal
        ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                   c='green', s=200, marker='o', label='Start', edgecolors='black')
        ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                   c='red', s=200, marker='s', label='Goal', edgecolors='black')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        ax1.set_title('3D Trajectory')
        
        # 2D top-down view
        ax2 = fig.add_subplot(132)
        ax2.scatter(point_cloud_np[:, 0], point_cloud_np[:, 1],
                   c='red', s=1, alpha=0.6, label='Obstacles')
        ax2.plot(positions[:, 0], positions[:, 1],
                'b-', linewidth=3, alpha=0.8, label='Trajectory')
        ax2.scatter(positions[0, 0], positions[0, 1],
                   c='green', s=200, marker='o', label='Start', edgecolors='black')
        ax2.scatter(positions[-1, 0], positions[-1, 1],
                   c='red', s=200, marker='s', label='Goal', edgecolors='black')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.legend()
        ax2.set_title('Top-down View')
        ax2.axis('equal')
        
        # Orientation plot (yaw angles over time)
        ax3 = fig.add_subplot(133)
        
        # Extract yaw angles (assuming 2D rotation)
        yaw_angles = []
        for pose in trajectory:
            R = pose[:3, :3]
            # Extract yaw from rotation matrix (assuming 2D rotation in xy-plane)
            yaw = torch.atan2(R[1, 0], R[0, 0]).item()
            yaw_angles.append(yaw)
        
        time_steps = np.linspace(0, 1, len(yaw_angles))
        ax3.plot(time_steps, yaw_angles, 'g-', linewidth=2, label='Yaw angle')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Yaw (radians)')
        ax3.legend()
        ax3.set_title('Orientation Profile')
        ax3.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Trajectory visualization saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def _compute_summary_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Compute summary metrics across all configurations"""
        summary = {}
        
        # Aggregate across different step sizes
        all_success_rates = []
        all_path_lengths = []
        all_smoothness_scores = []
        all_computation_times = []
        
        for step_key, step_results in results['trajectory_metrics'].items():
            all_success_rates.extend(step_results.get('success_rates', []))
            all_path_lengths.extend(step_results.get('path_lengths', []))
            all_smoothness_scores.extend(step_results.get('smoothness_scores', []))
            all_computation_times.extend(step_results.get('computation_times', []))
        
        # Overall metrics
        if all_success_rates:
            summary['overall_success_rate'] = np.mean(all_success_rates)
            summary['overall_path_length_mean'] = np.mean(all_path_lengths)
            summary['overall_smoothness_mean'] = np.mean(all_smoothness_scores)
            summary['overall_computation_time_mean'] = np.mean(all_computation_times)
        
        return summary
    
    def _tensordict_to_json(self, data: Any) -> Any:
        """Convert tensor dictionary to JSON-serializable format"""
        if isinstance(data, dict):
            return {k: self._tensordict_to_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._tensordict_to_json(item) for item in data]
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy().tolist()
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.floating, np.integer)):
            return float(data)
        else:
            return data
    
    def generate_evaluation_report(
        self,
        results: Dict[str, Any],
        save_path: str
    ):
        """Generate comprehensive evaluation report"""
        
        report_lines = [
            "# SE(3) RFM Model Evaluation Report",
            f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"Overall Success Rate: {results['summary'].get('overall_success_rate', 0.0):.2%}",
            f"Average Path Length: {results['summary'].get('overall_path_length_mean', 0.0):.4f}",
            f"Average Smoothness: {results['summary'].get('overall_smoothness_mean', 0.0):.4f}",
            f"Average Computation Time: {results['summary'].get('overall_computation_time_mean', 0.0):.4f}s",
            "",
            "## Detailed Results by ODE Steps",
        ]
        
        for step_key, step_results in results['trajectory_metrics'].items():
            n_steps = step_key.split('_')[1]
            report_lines.extend([
                f"### {n_steps} ODE Steps",
                f"- Success Rate: {step_results.get('success_rates_mean', 0.0):.2%}",
                f"- Path Length: {step_results.get('path_lengths_mean', 0.0):.4f} ± {step_results.get('path_lengths_std', 0.0):.4f}",
                f"- Smoothness: {step_results.get('smoothness_scores_mean', 0.0):.4f} ± {step_results.get('smoothness_scores_std', 0.0):.4f}",
                f"- Computation Time: {step_results.get('computation_times_mean', 0.0):.4f}s ± {step_results.get('computation_times_std', 0.0):.4f}s",
                f"- Average NFE: {step_results.get('nfe_counts_mean', 0.0):.1f}",
                ""
            ])
        
        # Write report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Evaluation report saved to {save_path}")


def run_evaluation(
    model_path: str,
    test_data_path: str,
    config: Dict[str, Any],
    save_dir: str
):
    """
    Standalone evaluation function
    
    Args:
        model_path: Path to trained model checkpoint
        test_data_path: Path to test data
        config: Model configuration
        save_dir: Directory to save evaluation results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SE3RFM(**config['model']).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load test data
    from ..loaders.se3_trajectory_dataset import SE3TrajectoryDataset
    test_dataset = SE3TrajectoryDataset(
        test_data_path,
        max_trajectories=50,
        augment_data=False
    )
    
    # Prepare test samples
    test_samples = []
    for i in range(min(len(test_dataset), 20)):  # Evaluate on 20 samples
        sample = test_dataset[i]
        test_samples.append(sample)
    
    # Run evaluation
    evaluator = SE3RFMEvaluator(model, device)
    results = evaluator.evaluate_model(test_samples, save_dir=save_dir)
    
    # Generate report
    evaluator.generate_evaluation_report(
        results, os.path.join(save_dir, 'evaluation_report.md')
    )
    
    return results