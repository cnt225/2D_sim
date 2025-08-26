import torch
import numpy as np
from .Lie import exp_se3, log_SE3, bracket_se3, is_SE3

def twist_to_se3_matrix(twist_vectors):
    """
    Convert 6D twist vectors to SE(3) matrices using exponential map
    
    Args:
        twist_vectors: [B, 6] twist vectors [ω, v]
    
    Returns:
        se3_matrices: [B, 4, 4] SE(3) matrices
    """
    if isinstance(twist_vectors, np.ndarray):
        twist_vectors = torch.from_numpy(twist_vectors).float()
    
    batch_size = twist_vectors.shape[0]
    assert twist_vectors.shape == (batch_size, 6), f"Expected [B, 6], got {twist_vectors.shape}"
    
    # Use Lie group exponential map: exp(ξ^) where ξ^ is se(3) algebra element
    se3_matrices = exp_se3(twist_vectors)
    
    return se3_matrices

def se3_matrix_to_twist(se3_matrices):
    """
    Convert SE(3) matrices to 6D twist vectors using logarithmic map
    
    Args:
        se3_matrices: [B, 4, 4] SE(3) matrices
    
    Returns:
        twist_vectors: [B, 6] twist vectors [ω, v]
    """
    if isinstance(se3_matrices, np.ndarray):
        se3_matrices = torch.from_numpy(se3_matrices).float()
    
    batch_size = se3_matrices.shape[0]
    assert se3_matrices.shape == (batch_size, 4, 4), f"Expected [B, 4, 4], got {se3_matrices.shape}"
    assert is_SE3(se3_matrices), "Input must be valid SE(3) matrices"
    
    # Use Lie group logarithmic map: log(T) → se(3) algebra
    se3_algebra = log_SE3(se3_matrices)
    twist_vectors = bracket_se3(se3_algebra)  # Extract 6D vector from 4x4 skew matrix
    
    return twist_vectors

def integrate_twist_trajectory(initial_pose, twist_sequence, dt_sequence=None):
    """
    Integrate a sequence of twist vectors to reconstruct SE(3) trajectory
    
    Args:
        initial_pose: [4, 4] initial SE(3) pose
        twist_sequence: [N, 6] sequence of twist vectors
        dt_sequence: [N] time intervals (optional, defaults to 1.0)
    
    Returns:
        trajectory: [N+1, 4, 4] SE(3) trajectory including initial pose
    """
    if isinstance(initial_pose, np.ndarray):
        initial_pose = torch.from_numpy(initial_pose).float()
    if isinstance(twist_sequence, np.ndarray):
        twist_sequence = torch.from_numpy(twist_sequence).float()
    
    n_steps = twist_sequence.shape[0]
    
    if dt_sequence is None:
        dt_sequence = torch.ones(n_steps)
    elif isinstance(dt_sequence, np.ndarray):
        dt_sequence = torch.from_numpy(dt_sequence).float()
    
    # Initialize trajectory with initial pose
    trajectory = torch.zeros(n_steps + 1, 4, 4)
    trajectory[0] = initial_pose
    
    current_pose = initial_pose.clone()
    
    for i in range(n_steps):
        # Scale twist by time interval
        scaled_twist = twist_sequence[i] * dt_sequence[i]
        
        # Compute incremental transformation
        delta_T = exp_se3(scaled_twist.unsqueeze(0))
        
        # Left-multiply for body frame velocity
        current_pose = current_pose @ delta_T.squeeze(0)
        trajectory[i + 1] = current_pose
    
    return trajectory

def compute_trajectory_twists(trajectory, dt_sequence=None):
    """
    Compute twist vectors from SE(3) trajectory
    
    Args:
        trajectory: [N, 4, 4] SE(3) trajectory
        dt_sequence: [N-1] time intervals (optional, defaults to 1.0)
    
    Returns:
        twist_sequence: [N-1, 6] twist vectors
    """
    if isinstance(trajectory, np.ndarray):
        trajectory = torch.from_numpy(trajectory).float()
    
    n_poses = trajectory.shape[0]
    
    if dt_sequence is None:
        dt_sequence = torch.ones(n_poses - 1)
    elif isinstance(dt_sequence, np.ndarray):
        dt_sequence = torch.from_numpy(dt_sequence).float()
    
    twist_sequence = torch.zeros(n_poses - 1, 6)
    
    for i in range(n_poses - 1):
        T1, T2 = trajectory[i], trajectory[i + 1]
        dt = dt_sequence[i]
        
        # Compute relative transformation
        T_rel = torch.inverse(T1) @ T2
        
        # Extract twist using logarithmic map
        twist = se3_matrix_to_twist(T_rel.unsqueeze(0)).squeeze(0)
        
        # Scale by time
        twist_sequence[i] = twist / dt
    
    return twist_sequence

def pose_difference_twist(pose1, pose2, frame='body'):
    """
    Compute twist vector between two SE(3) poses
    
    Args:
        pose1: [4, 4] SE(3) pose 1
        pose2: [4, 4] SE(3) pose 2  
        frame: 'body' or 'world' frame for twist
    
    Returns:
        twist: [6] twist vector from pose1 to pose2
    """
    if isinstance(pose1, np.ndarray):
        pose1 = torch.from_numpy(pose1).float()
    if isinstance(pose2, np.ndarray):
        pose2 = torch.from_numpy(pose2).float()
    
    if frame == 'body':
        # Body frame: T_rel = T1^{-1} * T2
        T_rel = torch.inverse(pose1) @ pose2
    elif frame == 'world':
        # World frame: T_rel = T2 * T1^{-1}  
        T_rel = pose2 @ torch.inverse(pose1)
    else:
        raise ValueError(f"Frame must be 'body' or 'world', got {frame}")
    
    # Extract twist
    twist = se3_matrix_to_twist(T_rel.unsqueeze(0)).squeeze(0)
    
    return twist

def interpolate_se3_poses(pose1, pose2, t):
    """
    Interpolate between two SE(3) poses using exponential map
    
    Args:
        pose1: [4, 4] SE(3) pose at t=0
        pose2: [4, 4] SE(3) pose at t=1
        t: scalar or [N] interpolation parameter(s)
    
    Returns:
        interpolated_poses: [4, 4] or [N, 4, 4] interpolated poses
    """
    if isinstance(pose1, np.ndarray):
        pose1 = torch.from_numpy(pose1).float()
    if isinstance(pose2, np.ndarray):
        pose2 = torch.from_numpy(pose2).float()
    if isinstance(t, (float, int)):
        t = torch.tensor(t).float()
    elif isinstance(t, np.ndarray):
        t = torch.from_numpy(t).float()
    
    # Compute relative transformation
    T_rel = torch.inverse(pose1) @ pose2
    
    # Extract twist
    twist_full = se3_matrix_to_twist(T_rel.unsqueeze(0)).squeeze(0)
    
    if t.dim() == 0:  # scalar
        # Scale twist by interpolation parameter
        scaled_twist = twist_full * t
        delta_T = exp_se3(scaled_twist.unsqueeze(0)).squeeze(0)
        return pose1 @ delta_T
    else:  # vector
        n_points = len(t)
        interpolated_poses = torch.zeros(n_points, 4, 4)
        
        for i, t_val in enumerate(t):
            scaled_twist = twist_full * t_val
            delta_T = exp_se3(scaled_twist.unsqueeze(0)).squeeze(0)
            interpolated_poses[i] = pose1 @ delta_T
        
        return interpolated_poses

def cfm_twist_integration(velocity_field, initial_twist, t_span, ode_solver='rk4', n_steps=20):
    """
    Integrate twist velocity field for CFM trajectory generation
    
    Args:
        velocity_field: function that takes (twist, t) -> twist_velocity
        initial_twist: [6] initial twist vector
        t_span: [t_start, t_end] time span
        ode_solver: integration method
        n_steps: number of integration steps
    
    Returns:
        twist_trajectory: [n_steps+1, 6] integrated twist trajectory
    """
    if isinstance(initial_twist, np.ndarray):
        initial_twist = torch.from_numpy(initial_twist).float()
    
    t_start, t_end = t_span
    dt = (t_end - t_start) / n_steps
    
    twist_trajectory = torch.zeros(n_steps + 1, 6)
    twist_trajectory[0] = initial_twist
    
    current_twist = initial_twist.clone()
    
    for i in range(n_steps):
        t = t_start + i * dt
        
        if ode_solver == 'rk4':
            # RK4 integration
            k1 = velocity_field(current_twist, t)
            k2 = velocity_field(current_twist + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = velocity_field(current_twist + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = velocity_field(current_twist + dt * k3, t + dt)
            
            current_twist = current_twist + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        elif ode_solver == 'euler':
            # Euler integration
            twist_velocity = velocity_field(current_twist, t)
            current_twist = current_twist + dt * twist_velocity
        else:
            raise ValueError(f"Unsupported ODE solver: {ode_solver}")
        
        twist_trajectory[i + 1] = current_twist
    
    return twist_trajectory

def validate_twist_data(twist_vectors, max_angular=10.0, max_linear=5.0):
    """
    Validate twist vector data for reasonable magnitudes
    
    Args:
        twist_vectors: [N, 6] twist vectors
        max_angular: maximum angular velocity (rad/s)
        max_linear: maximum linear velocity (m/s)
    
    Returns:
        is_valid: [N] boolean mask of valid twists
        statistics: dict with validation statistics
    """
    if isinstance(twist_vectors, np.ndarray):
        twist_vectors = torch.from_numpy(twist_vectors).float()
    
    n_twists = twist_vectors.shape[0]
    is_valid = torch.ones(n_twists, dtype=torch.bool)
    
    # Check angular velocity magnitudes
    angular_mags = torch.norm(twist_vectors[:, :3], dim=1)
    angular_valid = angular_mags <= max_angular
    
    # Check linear velocity magnitudes  
    linear_mags = torch.norm(twist_vectors[:, 3:], dim=1)
    linear_valid = linear_mags <= max_linear
    
    is_valid = angular_valid & linear_valid
    
    statistics = {
        'n_total': n_twists,
        'n_valid': torch.sum(is_valid).item(),
        'angular_mean': torch.mean(angular_mags).item(),
        'angular_max': torch.max(angular_mags).item(),
        'angular_std': torch.std(angular_mags).item(),
        'linear_mean': torch.mean(linear_mags).item(),
        'linear_max': torch.max(linear_mags).item(),
        'linear_std': torch.std(linear_mags).item(),
    }
    
    return is_valid, statistics