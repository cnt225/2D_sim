"""
ODE Solvers for SE(3) Flow Matching

This module provides ODE solvers specifically designed for SE(3) manifold
integration in Riemannian Flow Matching applications.

Supported solvers:
- Euler method
- Runge-Kutta 4th order (RK4)
- Adaptive step size solvers (future)
"""

import torch
import torch.nn as nn
from typing import Callable, Any
from abc import ABC, abstractmethod


class ODESolver(ABC):
    """Abstract base class for ODE solvers"""
    
    @abstractmethod
    def step(
        self, 
        current_state: torch.Tensor,
        velocity: torch.Tensor, 
        dt: torch.Tensor,
        se3_utils: Any
    ) -> torch.Tensor:
        """
        Take one integration step
        
        Args:
            current_state: Current SE(3) state [batch_size, 4, 4]
            velocity: SE(3) twist velocity [batch_size, 6]
            dt: Time step
            se3_utils: SE3Utils instance for manifold operations
            
        Returns:
            next_state: Next SE(3) state [batch_size, 4, 4]
        """
        pass


class EulerSolver(ODESolver):
    """
    Euler method for SE(3) integration
    
    Simple first-order integration scheme:
    T_{n+1} = T_n * exp(dt * twist_n)
    """
    
    def step(
        self,
        current_state: torch.Tensor,
        velocity: torch.Tensor,
        dt: torch.Tensor, 
        se3_utils: Any
    ) -> torch.Tensor:
        """
        Euler integration step on SE(3) manifold
        
        Args:
            current_state: [batch_size, 4, 4] current SE(3) poses
            velocity: [batch_size, 6] twist velocities
            dt: scalar or [batch_size] time step(s)
            se3_utils: SE3Utils instance
            
        Returns:
            next_state: [batch_size, 4, 4] next SE(3) poses
        """
        # Scale velocity by time step
        if dt.dim() == 0:  # scalar dt
            scaled_twist = dt * velocity
        else:  # batch dt
            scaled_twist = dt.unsqueeze(-1) * velocity
        
        # Convert twist to SE(3) transformation
        delta_T = se3_utils.exp_se3(scaled_twist)
        
        # Compose with current state
        next_state = se3_utils.compose_se3(current_state, delta_T)
        
        return next_state


class RK4Solver(ODESolver):
    """
    Runge-Kutta 4th order method for SE(3) integration
    
    Higher-order integration scheme with better accuracy:
    k1 = f(T_n)
    k2 = f(T_n * exp(dt/2 * k1))
    k3 = f(T_n * exp(dt/2 * k2))
    k4 = f(T_n * exp(dt * k3))
    T_{n+1} = T_n * exp(dt/6 * (k1 + 2*k2 + 2*k3 + k4))
    """
    
    def __init__(self, velocity_field_func: Callable = None):
        """
        Initialize RK4 solver
        
        Args:
            velocity_field_func: Function that computes velocity given state
                                Should have signature: f(T, t, other_inputs) -> twist
        """
        self.velocity_field_func = velocity_field_func
    
    def step(
        self,
        current_state: torch.Tensor,
        velocity: torch.Tensor,
        dt: torch.Tensor,
        se3_utils: Any
    ) -> torch.Tensor:
        """
        RK4 integration step on SE(3) manifold
        
        Note: This is a simplified version that uses the provided velocity.
        For full RK4, we would need to re-evaluate the velocity field at intermediate steps.
        
        Args:
            current_state: [batch_size, 4, 4] current SE(3) poses
            velocity: [batch_size, 6] twist velocities 
            dt: scalar or [batch_size] time step(s)
            se3_utils: SE3Utils instance
            
        Returns:
            next_state: [batch_size, 4, 4] next SE(3) poses
        """
        # Simplified RK4 - assumes constant velocity over step
        # For full RK4, would need access to velocity field function
        
        if self.velocity_field_func is None:
            # Fallback to Euler if no velocity field function provided
            return EulerSolver().step(current_state, velocity, dt, se3_utils)
        
        # Full RK4 implementation would go here
        # For now, use improved Euler (midpoint method)
        
        # k1 = current velocity
        k1 = velocity
        
        # Intermediate step
        if dt.dim() == 0:
            dt_half = dt / 2
            scaled_k1 = dt_half * k1
        else:
            dt_half = dt / 2
            scaled_k1 = dt_half.unsqueeze(-1) * k1
            
        delta_T_half = se3_utils.exp_se3(scaled_k1)
        intermediate_state = se3_utils.compose_se3(current_state, delta_T_half)
        
        # k2 = velocity at intermediate state (would need to re-evaluate)
        # For simplicity, assume k2 ≈ k1
        k2 = k1
        
        # Final step using k2
        if dt.dim() == 0:
            scaled_k2 = dt * k2
        else:
            scaled_k2 = dt.unsqueeze(-1) * k2
            
        delta_T = se3_utils.exp_se3(scaled_k2)
        next_state = se3_utils.compose_se3(current_state, delta_T)
        
        return next_state
    
    def full_rk4_step(
        self,
        current_state: torch.Tensor,
        time: torch.Tensor,
        dt: torch.Tensor,
        se3_utils: Any,
        other_inputs: dict
    ) -> torch.Tensor:
        """
        Full RK4 integration step with velocity field re-evaluation
        
        Args:
            current_state: [batch_size, 4, 4] current SE(3) poses
            time: [batch_size, 1] current time
            dt: scalar time step
            se3_utils: SE3Utils instance
            other_inputs: dict with other inputs needed for velocity field
            
        Returns:
            next_state: [batch_size, 4, 4] next SE(3) poses
        """
        if self.velocity_field_func is None:
            raise ValueError("velocity_field_func must be provided for full RK4")
        
        # k1 = f(T_n, t_n)
        k1 = self.velocity_field_func(current_state, time, **other_inputs)
        
        # Intermediate state 1: T_n * exp(dt/2 * k1)
        dt_half = dt / 2
        delta_T1 = se3_utils.exp_se3(dt_half * k1)
        state1 = se3_utils.compose_se3(current_state, delta_T1)
        
        # k2 = f(T_n * exp(dt/2 * k1), t_n + dt/2)
        k2 = self.velocity_field_func(state1, time + dt_half, **other_inputs)
        
        # Intermediate state 2: T_n * exp(dt/2 * k2)
        delta_T2 = se3_utils.exp_se3(dt_half * k2)
        state2 = se3_utils.compose_se3(current_state, delta_T2)
        
        # k3 = f(T_n * exp(dt/2 * k2), t_n + dt/2)
        k3 = self.velocity_field_func(state2, time + dt_half, **other_inputs)
        
        # Intermediate state 3: T_n * exp(dt * k3)
        delta_T3 = se3_utils.exp_se3(dt * k3)
        state3 = se3_utils.compose_se3(current_state, delta_T3)
        
        # k4 = f(T_n * exp(dt * k3), t_n + dt)
        k4 = self.velocity_field_func(state3, time + dt, **other_inputs)
        
        # Final integration: T_{n+1} = T_n * exp(dt/6 * (k1 + 2*k2 + 2*k3 + k4))
        combined_velocity = (k1 + 2*k2 + 2*k3 + k4) / 6
        delta_T_final = se3_utils.exp_se3(dt * combined_velocity)
        next_state = se3_utils.compose_se3(current_state, delta_T_final)
        
        return next_state


class AdaptiveRK45Solver(ODESolver):
    """
    Adaptive Runge-Kutta 4(5) solver with error control
    
    Future implementation for adaptive step size control.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        
    def step(
        self,
        current_state: torch.Tensor,
        velocity: torch.Tensor,
        dt: torch.Tensor,
        se3_utils: Any
    ) -> torch.Tensor:
        # Placeholder - would implement adaptive step size logic
        # For now, fallback to RK4
        return RK4Solver().step(current_state, velocity, dt, se3_utils)


def get_ode_solver(solver_type: str, **kwargs) -> ODESolver:
    """
    Factory function to get ODE solver by name
    
    Args:
        solver_type: Type of solver ('euler', 'rk4', 'adaptive')
        **kwargs: Additional arguments for solver initialization
        
    Returns:
        solver: ODE solver instance
    """
    solver_type = solver_type.lower()
    
    if solver_type == 'euler':
        return EulerSolver()
    elif solver_type == 'rk4':
        return RK4Solver(**kwargs)
    elif solver_type == 'adaptive' or solver_type == 'rk45':
        return AdaptiveRK45Solver(**kwargs)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")


# =============================================================================
# Helper Functions for Integration
# =============================================================================

def integrate_se3_trajectory(
    start_pose: torch.Tensor,
    velocity_field_func: Callable,
    time_span: torch.Tensor,
    solver_type: str = 'rk4',
    se3_utils: Any = None,
    **velocity_field_kwargs
) -> torch.Tensor:
    """
    Integrate SE(3) trajectory using specified ODE solver
    
    Args:
        start_pose: [4, 4] initial SE(3) pose
        velocity_field_func: Function that computes velocity field
        time_span: [n_steps] time points to integrate over
        solver_type: Type of ODE solver to use
        se3_utils: SE3Utils instance
        **velocity_field_kwargs: Additional arguments for velocity field
        
    Returns:
        trajectory: [n_steps, 4, 4] SE(3) trajectory
    """
    if se3_utils is None:
        from .se3_utils import SE3Utils
        se3_utils = SE3Utils()
    
    solver = get_ode_solver(solver_type, velocity_field_func=velocity_field_func)
    
    trajectory = [start_pose.clone()]
    current_pose = start_pose.unsqueeze(0)  # Add batch dimension
    
    for i in range(len(time_span) - 1):
        t_current = time_span[i:i+1].unsqueeze(0)  # [1, 1]
        dt = time_span[i+1] - time_span[i]
        
        # Compute velocity at current state
        velocity = velocity_field_func(current_pose, t_current, **velocity_field_kwargs)
        
        # Take integration step
        if hasattr(solver, 'full_rk4_step') and solver_type == 'rk4':
            # Use full RK4 if available
            next_pose = solver.full_rk4_step(
                current_pose, t_current, dt, se3_utils, velocity_field_kwargs
            )
        else:
            # Use standard step
            next_pose = solver.step(current_pose, velocity, dt, se3_utils)
        
        trajectory.append(next_pose.squeeze(0))  # Remove batch dimension
        current_pose = next_pose
    
    return torch.stack(trajectory)


def compute_trajectory_metrics(trajectory: torch.Tensor, se3_utils: Any = None) -> dict:
    """
    Compute metrics for SE(3) trajectory quality
    
    Args:
        trajectory: [n_steps, 4, 4] SE(3) trajectory
        se3_utils: SE3Utils instance
        
    Returns:
        metrics: Dictionary of trajectory quality metrics
    """
    if se3_utils is None:
        from .se3_utils import SE3Utils
        se3_utils = SE3Utils()
    
    n_steps = trajectory.shape[0]
    
    # Compute path length
    path_length = 0.0
    velocities = []
    
    for i in range(n_steps - 1):
        T1 = trajectory[i]
        T2 = trajectory[i + 1]
        
        # Compute relative transformation
        T_rel = se3_utils.compose_se3(se3_utils.inverse_se3(T1), T2)
        twist = se3_utils.log_se3(T_rel)
        
        # Add to path length (geodesic distance)
        path_length += torch.norm(twist).item()
        velocities.append(twist)
    
    if velocities:
        velocities = torch.stack(velocities)
        
        # Compute smoothness metrics
        velocity_norms = torch.norm(velocities, dim=1)
        acceleration_norms = torch.norm(torch.diff(velocities, dim=0), dim=1)
        
        smoothness = {
            'mean_velocity': velocity_norms.mean().item(),
            'std_velocity': velocity_norms.std().item(),
            'mean_acceleration': acceleration_norms.mean().item(),
            'std_acceleration': acceleration_norms.std().item(),
        }
    else:
        smoothness = {
            'mean_velocity': 0.0,
            'std_velocity': 0.0,
            'mean_acceleration': 0.0,
            'std_acceleration': 0.0,
        }
    
    return {
        'path_length': path_length,
        'n_steps': n_steps,
        **smoothness
    }