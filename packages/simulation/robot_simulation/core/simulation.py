"""
simulation.py - SE(3) Rigid Body 시뮬레이션 로직
SE(3) 포즈를 목표로 하는 위치/방향 제어 구현
"""
import numpy as np
from Box2D.b2 import revoluteJointDef, dynamicBody
from robot_simulation.config_loader import get_se3_simulation_config


class RobotSimulation:
    """SE(3) Rigid Body 시뮬레이션 클래스"""
    
    def __init__(self, world, robot_body, obstacles, robot_config, target_pose, init_pose=None, policy_type="position_control"):
        """
        Args:
            world: Box2D world
            robot_body: Single rigid body (Box2D body)
            obstacles: List of obstacle bodies
            robot_config: Rigid body configuration dict
            target_pose: Target SE(3) pose [x, y, z, roll, pitch, yaw]
            init_pose: Initial SE(3) pose [x, y, z, roll, pitch, yaw]. If None, uses current pose.
            policy_type: Control policy ("position_control" or "velocity_control")
        """
        self.world = world
        self.robot_body = robot_body
        self.obstacles = obstacles
        self.robot_config = robot_config
        self.target_pose = np.array(target_pose)  # SE(3) target pose
        self.policy_type = policy_type
        
        # Get SE(3) simulation configuration
        se3_config = get_se3_simulation_config()
        control_config = se3_config.get('control', {})
        
        # Control parameters
        self.position_gain = control_config.get('position_gain', 10.0)
        self.orientation_gain = control_config.get('orientation_gain', 5.0)
        self.max_linear_velocity = control_config.get('max_linear_velocity', 5.0)
        self.max_angular_velocity = control_config.get('max_angular_velocity', 2.0)
        
        # Physics settings
        self.TIME_STEP = 1.0/60.0
        self.VEL_ITERS = 10
        self.POS_ITERS = 10
        
        # Set initial pose if specified
        if init_pose is not None:
            print(f"Setting initial SE(3) pose: {init_pose}")
            self.set_robot_pose(init_pose)
        
        print(f"SE(3) Robot Simulation initialized:")
        print(f"  Robot: {robot_config['name']}")
        print(f"  Target pose: {self.target_pose}")
        print(f"  Control gains: pos={self.position_gain}, ori={self.orientation_gain}")
    
    def set_robot_pose(self, pose):
        """
        Set robot to specific SE(3) pose
        
        Args:
            pose: SE(3) pose [x, y, z, roll, pitch, yaw]
        """
        x, y, z, roll, pitch, yaw = pose
        # For 2D simulation, use only x, y, yaw
        self.robot_body.position = (x, y)
        self.robot_body.angle = yaw
        # Reset velocities
        self.robot_body.linearVelocity = (0, 0)
        self.robot_body.angularVelocity = 0
    
    def get_current_pose(self):
        """
        Get current SE(3) pose of robot
        
        Returns:
            SE(3) pose [x, y, z=0, roll=0, pitch=0, yaw]
        """
        pos = self.robot_body.position
        angle = self.robot_body.angle
        return np.array([pos.x, pos.y, 0.0, 0.0, 0.0, angle])
    
    def step(self):
        """Execute one simulation step with SE(3) pose control"""
        if self.policy_type == "position_control":
            self.apply_position_control()
        elif self.policy_type == "velocity_control":
            self.apply_velocity_control()
        else:
            print(f"Warning: Unknown policy type '{self.policy_type}', using position_control")
            self.apply_position_control()
        
        # Physics step
        self.world.Step(self.TIME_STEP, self.VEL_ITERS, self.POS_ITERS)
        self.world.ClearForces()
    
    def apply_position_control(self):
        """Apply position-based SE(3) pose control"""
        current_pose = self.get_current_pose()
        
        # Position error (x, y)
        pos_error = self.target_pose[:2] - current_pose[:2]
        
        # Orientation error (yaw)
        yaw_error = self.target_pose[5] - current_pose[5]
        # Normalize yaw error to [-π, π]
        while yaw_error > np.pi:
            yaw_error -= 2*np.pi
        while yaw_error < -np.pi:
            yaw_error += 2*np.pi
        
        # Calculate desired velocities
        desired_linear_vel = self.position_gain * pos_error
        desired_angular_vel = self.orientation_gain * yaw_error
        
        # Limit velocities
        linear_speed = np.linalg.norm(desired_linear_vel)
        if linear_speed > self.max_linear_velocity:
            desired_linear_vel = desired_linear_vel * self.max_linear_velocity / linear_speed
        
        if abs(desired_angular_vel) > self.max_angular_velocity:
            desired_angular_vel = np.sign(desired_angular_vel) * self.max_angular_velocity
        
        # Apply velocities to robot body
        self.robot_body.linearVelocity = (desired_linear_vel[0], desired_linear_vel[1])
        self.robot_body.angularVelocity = desired_angular_vel
    
    def apply_velocity_control(self):
        """Apply velocity-based SE(3) pose control (smoother)"""
        current_pose = self.get_current_pose()
        
        # Position error (x, y)
        pos_error = self.target_pose[:2] - current_pose[:2]
        
        # Orientation error (yaw)
        yaw_error = self.target_pose[5] - current_pose[5]
        # Normalize yaw error to [-π, π]
        while yaw_error > np.pi:
            yaw_error -= 2*np.pi
        while yaw_error < -np.pi:
            yaw_error += 2*np.pi
        
        # Smooth velocity control with damping
        current_linear_vel = np.array([self.robot_body.linearVelocity.x, self.robot_body.linearVelocity.y])
        current_angular_vel = self.robot_body.angularVelocity
        
        # Calculate desired velocities with damping
        kp_pos = self.position_gain * 0.5  # Reduced gain for smoother control
        kd_pos = 2.0  # Damping factor
        
        kp_ori = self.orientation_gain * 0.5
        kd_ori = 1.0
        
        desired_linear_vel = kp_pos * pos_error - kd_pos * current_linear_vel
        desired_angular_vel = kp_ori * yaw_error - kd_ori * current_angular_vel
        
        # Limit velocities
        linear_speed = np.linalg.norm(desired_linear_vel)
        if linear_speed > self.max_linear_velocity:
            desired_linear_vel = desired_linear_vel * self.max_linear_velocity / linear_speed
        
        if abs(desired_angular_vel) > self.max_angular_velocity:
            desired_angular_vel = np.sign(desired_angular_vel) * self.max_angular_velocity
        
        # Apply velocities to robot body
        self.robot_body.linearVelocity = (desired_linear_vel[0], desired_linear_vel[1])
        self.robot_body.angularVelocity = desired_angular_vel
    
    def get_pose_error(self):
        """Calculate SE(3) pose error magnitude"""
        current_pose = self.get_current_pose()
        
        # Position error (x, y)
        pos_error = np.linalg.norm(self.target_pose[:2] - current_pose[:2])
        
        # Orientation error (yaw)
        yaw_error = abs(self.target_pose[5] - current_pose[5])
        # Normalize to [0, π]
        yaw_error = min(yaw_error, 2*np.pi - yaw_error)
        
        # Combined error (position + 0.5 * orientation)
        total_error = pos_error + 0.5 * yaw_error
        
        return total_error
    
    def is_pose_reached(self, position_tolerance=0.05, orientation_tolerance=0.1):
        """Check if target SE(3) pose is reached"""
        current_pose = self.get_current_pose()
        
        # Position error (x, y)
        pos_error = np.linalg.norm(self.target_pose[:2] - current_pose[:2])
        
        # Orientation error (yaw)
        yaw_error = abs(self.target_pose[5] - current_pose[5])
        yaw_error = min(yaw_error, 2*np.pi - yaw_error)
        
        return pos_error < position_tolerance and yaw_error < orientation_tolerance
    
    def get_robot_vertices(self):
        """Get current robot vertices in world coordinates for visualization"""
        # Get ellipse parameters
        semi_major = self.robot_config['semi_major_axis']
        semi_minor = self.robot_config['semi_minor_axis']
        
        # Create ellipse vertices in local coordinates
        num_points = 20
        local_vertices = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = semi_major * np.cos(angle)
            y = semi_minor * np.sin(angle)
            local_vertices.append((x, y))
        
        # Transform to world coordinates
        pos = self.robot_body.position
        robot_angle = self.robot_body.angle
        
        world_vertices = []
        for lx, ly in local_vertices:
            # Rotate and translate
            wx = pos.x + lx * np.cos(robot_angle) - ly * np.sin(robot_angle)
            wy = pos.y + lx * np.sin(robot_angle) + ly * np.cos(robot_angle)
            world_vertices.append((wx, wy))
        
        return world_vertices
    
    def get_target_vertices(self):
        """Get target pose vertices for visualization (red outline)"""
        # Get ellipse parameters
        semi_major = self.robot_config['semi_major_axis']
        semi_minor = self.robot_config['semi_minor_axis']
        
        # Create ellipse vertices in local coordinates
        num_points = 20
        local_vertices = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = semi_major * np.cos(angle)
            y = semi_minor * np.sin(angle)
            local_vertices.append((x, y))
        
        # Transform to target pose coordinates
        target_x, target_y = self.target_pose[:2]
        target_yaw = self.target_pose[5]
        
        world_vertices = []
        for lx, ly in local_vertices:
            # Rotate and translate to target pose
            wx = target_x + lx * np.cos(target_yaw) - ly * np.sin(target_yaw)
            wy = target_y + lx * np.sin(target_yaw) + ly * np.cos(target_yaw)
            world_vertices.append((wx, wy))
        
        return world_vertices
