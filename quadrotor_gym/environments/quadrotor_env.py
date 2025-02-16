import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pybullet as p
import pybullet_data

class QuadrotorEnv(gym.Env):
    """
    Quadrotor environment for trajectory tracking and hovering tasks.
    Follows the Gymnasium interface.
    """
    
    def __init__(self, task='hover'):
        super().__init__()
        
        # Initialize PyBullet
        self.client = p.connect(p.DIRECT)  # Headless mode
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Get absolute path to URDF file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.urdf_path = os.path.join(current_dir, "..", "assets", "quadrotor.urdf")
        print(f"Current directory: {current_dir}")
        print(f"Looking for URDF at: {self.urdf_path}")
        print(f"File exists: {os.path.exists(self.urdf_path)}")
        
        # print(f"Looking for URDF at: {self.urdf_path}")
        # print(f"File exists: {os.path.exists(self.urdf_path)}")
        
        # Verify file exists
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF file not found at {self.urdf_path}")
        
        # Load quadrotor URDF with absolute path
        self.drone = p.loadURDF(self.urdf_path, [0, 0, 0])
        
        # Define action and observation spaces
        # Actions: [thrust, roll_torque, pitch_torque, yaw_torque]
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1]),
            high=np.array([1, 1, 1, 1]),
            dtype=np.float32
        )
        
        # Observations: [x, y, z, roll, pitch, yaw, vx, vy, vz, angular_vel_x, angular_vel_y, angular_vel_z]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * 12),
            high=np.array([np.inf] * 12),
            dtype=np.float32
        )
        
        # Task settings
        self.task = task
        self.target_position = np.zeros(3)
        self.trajectory = None
        self.trajectory_time = 0
        
        # Episode settings
        self.max_steps = 1000
        self.current_step = 0

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset simulation
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81)
        
        # Reset drone position and orientation
        initial_pos = [0., 0., 0.5]
        initial_orientation = p.getQuaternionFromEuler([0., 0., 0.])
        self.drone = p.loadURDF(self.urdf_path, initial_pos, initial_orientation)
        
        # Reset target based on task
        if self.task == 'hover':
            self.target_position = np.array([0., 0., 1.])
        else:  # trajectory tracking
            self.trajectory_time = 0
            
        self.current_step = 0
        
        # Get initial observation
        observation = self._get_observation()
        info = {}
        
        return observation, info

    def step(self, action):
        """Execute one step in the environment."""
        # Apply action
        self._apply_action(action)
        
        # Simulate one step
        p.stepSimulation()
        
        # Get new state
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._compute_reward(observation)
        
        # Check if episode is done
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        # Check if drone is in valid state
        truncated = not self._is_valid_state(observation)
        
        info = {}
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Get current state observation."""
        pos, ori = p.getBasePositionAndOrientation(self.drone)
        vel, ang_vel = p.getBaseVelocity(self.drone)
        euler = p.getEulerFromQuaternion(ori)
        
        return np.array(list(pos) + list(euler) + list(vel) + list(ang_vel))

    def _apply_action(self, action):
        """Apply the given action to the drone."""
        # Scale actions from [-1, 1] to actual force/torque values
        thrust = (action[0] + 1.0) * 4.9  # Scaled to counteract gravity
        torques = action[1:] * 0.1
        
        # Apply thrust (in drone's frame)
        p.applyExternalForce(self.drone, -1, [0, 0, thrust], [0, 0, 0], p.LINK_FRAME)
        
        # Apply torques (in drone's frame)
        p.applyExternalTorque(self.drone, -1, torques, p.LINK_FRAME)

    def _compute_reward(self, observation):
        """Compute reward based on current state."""
        if self.task == 'hover':
            # For hovering: negative distance to target position
            pos = observation[:3]
            distance = np.linalg.norm(pos - self.target_position)
            return -distance
        else:
            # For trajectory tracking: negative distance to current target point
            current_target = self._get_target_point(self.trajectory_time)
            pos = observation[:3]
            distance = np.linalg.norm(pos - current_target)
            self.trajectory_time += 0.01  # Update trajectory time
            return -distance

    def _is_valid_state(self, observation):
        """Check if the current state is valid."""
        pos = observation[:3]
        orientation = observation[3:6]
        
        # Check position bounds
        if any(abs(p) > 5.0 for p in pos):
            return False
        
        # Check orientation bounds (in radians)
        if any(abs(o) > np.pi/2 for o in orientation):
            return False
            
        return True

    def _get_target_point(self, t):
        """Get target point for trajectory tracking at time t."""
        if self.trajectory is None:
            # Default circular trajectory
            radius = 1.0
            x = radius * np.cos(t)
            y = radius * np.sin(t)
            z = 1.0
            return np.array([x, y, z])
        return self.trajectory(t)

    def set_trajectory(self, trajectory_func):
        """Set custom trajectory function."""
        self.trajectory = trajectory_func

    def render(self):
        """
        Render the environment.
        This method should be implemented based on your visualization needs.
        """
        pass

    def close(self):
        """Clean up resources."""
        p.disconnect(self.client)