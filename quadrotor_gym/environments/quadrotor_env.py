import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data

class QuadrotorEnv(gym.Env):
    """
    Quadrotor environment with full dynamics implementation.
    Based on gym-pybullet-drones implementation.
    """
    
    def __init__(self, task='hover'):
        super().__init__()
        
        #### Define physical parameters and constants ####
        # Basic constants
        self.GRAVITY = 9.81
        self.RAD2DEG = 180/np.pi
        self.DEG2RAD = np.pi/180
        self.TIMESTEP = 0.01  # 100Hz control
        
        # Quadrotor physical parameters
        self.MASS = 0.027  # kg
        self.L = 0.039  # arm length in meters
        self.THRUST2WEIGHT_RATIO = 2.25
        self.MAX_SPEED_KMH = 30
        self.MAX_RPM = 44000
        self.MAX_THRUST = 0.16  # N
        self.MAX_XY_TORQUE = 0.031  # N*m
        self.MAX_Z_TORQUE = 0.0056  # N*m
        self.A = 0.1  # Rotor area
        
        # Dynamics coefficients
        self.KF = 3.16e-10  # thrust coefficient
        self.KM = 7.94e-12  # moment coefficient
        self.INERTIA = np.array([1.4e-5, 1.4e-5, 2.17e-5])  # Inertia matrix diagonal
        self.DRAG_COEFF = np.array([0.1, 0.1, 0.1])  # Drag coefficients
        
        # More dynamics parameters
        self.GROUND_EFFECT_COEFF = 11.36859
        self.PROP_RADIUS = 0.0225
        self.DRAG_COEFF_XY = 0.012
        self.DRAG_COEFF_Z = 0.009
        self.BLADE_FLAPPING_COEFF = 0.1
        
        # Motor dynamics
        self.MOTOR_TC = 0.02  # Motor time constant
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        
        #### Initialize simulation ####
        self.client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -self.GRAVITY)
        p.setTimeStep(self.TIMESTEP)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load URDF
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.urdf_path = os.path.join(current_dir, "..", "assets", "quadrotor.urdf")
        
        #### Define spaces ####
        # Action space: [PWM1, PWM2, PWM3, PWM4]
        self.action_space = spaces.Box(
            low=np.array([self.MIN_PWM]*4),
            high=np.array([self.MAX_PWM]*4),
            dtype=np.float32
        )
        
        # Observation space: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf]*13),
            high=np.array([np.inf]*13),
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
        
        # Additional states
        self.last_rpms = np.zeros(4)
        self.last_pos_error = np.zeros(3)

    def reset(self, seed=None, options=None):
        """Reset environment with randomized initial state."""
        super().reset(seed=seed)
        
        # Reset simulation
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -self.GRAVITY)
        p.setTimeStep(self.TIMESTEP)
        
        # Randomize initial position and orientation
        if seed is not None:
            np.random.seed(seed)
            
        init_pos = np.array([0., 0., 0.5])
        init_pos += np.random.uniform(low=-0.1, high=0.1, size=3)
        
        init_orientation = np.array([0., 0., 0.])  # euler angles
        init_orientation += np.random.uniform(low=-5, high=5, size=3) * self.DEG2RAD
        init_quat = p.getQuaternionFromEuler(init_orientation)
        
        # Load quadrotor with initial state
        self.drone = p.loadURDF(
            self.urdf_path,
            init_pos,
            init_quat,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )
        
        # Reset internal variables
        self.last_rpms = np.zeros(4)
        self.last_pos_error = np.zeros(3)
        self.current_step = 0
        
        # Set task target
        if self.task == 'hover':
            self.target_position = np.array([0., 0., 1.])
        else:
            self.trajectory_time = 0
        
        # Get initial observation
        observation = self._get_observation()
        info = {}
        
        return observation, info

    def step(self, action):
        """Execute one step with full dynamics."""
        # Convert PWM to RPM
        rpms = self._pwm_to_rpm(action)
        
        # Apply motor dynamics
        rpms = self._motor_dynamics(rpms)
        
        # Compute forces and torques including all dynamic effects
        forces, torques = self._compute_dynamics(rpms)
        
        # Apply forces and torques
        self._apply_forces_and_torques(forces, torques)
        
        # Step simulation
        p.stepSimulation()
        
        # Update internal state
        self.last_rpms = rpms
        
        # Get new state
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._compute_reward(observation)
        
        # Update counter and check termination
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = not self._is_valid_state(observation)
        
        info = {}
        
        return observation, reward, terminated, truncated, info

    def _pwm_to_rpm(self, pwm):
        """Convert PWM signals to RPM."""
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

    def _motor_dynamics(self, target_rpms):
        """Apply motor dynamics with time constant."""
        motor_up = self.TIMESTEP / self.MOTOR_TC
        return self.last_rpms + motor_up * (target_rpms - self.last_rpms)

    def _compute_dynamics(self, rpms):
        """Compute full quadrotor dynamics."""
        # Get current state
        pos, quat = p.getBasePositionAndOrientation(self.drone)
        vel, ang_vel = p.getBaseVelocity(self.drone)
        rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        
        # Compute thrust forces from each rotor
        thrust_forces = self.KF * rpms**2
        thrust = np.array([0, 0, np.sum(thrust_forces)])  # Total thrust in z-direction
        
        # Compute torques from motors
        torques = np.array([
            self.L * (thrust_forces[0] - thrust_forces[2]),
            self.L * (thrust_forces[1] - thrust_forces[3]),
            self.KM * (rpms[0]**2 - rpms[1]**2 + rpms[2]**2 - rpms[3]**2)
        ])
        
        # Add drag effects
        drag = -self.DRAG_COEFF * np.array(vel)
        drag_body = rot_mat.T @ drag
        
        # Add ground effect
        z_ground = pos[2]
        ge_factor = np.exp(-self.GROUND_EFFECT_COEFF * z_ground / self.L)
        ground_effect = np.array([0, 0, ge_factor * np.sum(thrust_forces) * self.A / (4 * np.pi * self.L**2)])
        
        # Add blade flapping effects
        mean_rpm = np.mean(rpms)
        if mean_rpm > 0:  # Prevent division by zero
            flap_effect = -self.BLADE_FLAPPING_COEFF * np.array([
                vel[0] / (mean_rpm * self.PROP_RADIUS),
                vel[1] / (mean_rpm * self.PROP_RADIUS),
                0
            ])
        else:
            flap_effect = np.zeros(3)
        
        # Combine all forces and torques
        total_force = thrust + drag_body + ground_effect + flap_effect
        
        return total_force, torques

    def _apply_forces_and_torques(self, forces, torques):
        """Apply forces and torques to the drone."""
        p.applyExternalForce(
            self.drone,
            linkIndex=-1,
            forceObj=forces,
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME
        )
        
        p.applyExternalTorque(
            self.drone,
            linkIndex=-1,
            torqueObj=torques,
            flags=p.LINK_FRAME
        )

    def _get_observation(self):
        """Get current state observation."""
        pos, quat = p.getBasePositionAndOrientation(self.drone)
        vel, ang_vel = p.getBaseVelocity(self.drone)
        
        observation = np.concatenate([
            pos,
            quat,
            vel,
            ang_vel
        ]).astype(np.float32)
        
        return observation

    def _compute_reward(self, observation):
        """Compute reward based on task."""
        pos = observation[:3]
        
        if self.task == 'hover':
            pos_error = pos - self.target_position
            vel = observation[7:10]
            
            # Position error cost
            position_cost = np.sum(pos_error**2)
            
            # Velocity cost (penalize high velocities)
            velocity_cost = 0.1 * np.sum(vel**2)
            
            # Change in position error (smoothness)
            d_pos_error = pos_error - self.last_pos_error
            smoothness_cost = 0.1 * np.sum(d_pos_error**2)
            
            # Update last position error
            self.last_pos_error = pos_error
            
            return -(position_cost + velocity_cost + smoothness_cost)
        else:
            current_target = self._get_target_point(self.trajectory_time)
            distance = np.linalg.norm(pos - current_target)
            self.trajectory_time += self.TIMESTEP
            return -distance

    def _is_valid_state(self, observation):
        """Check if current state is valid."""
        pos = observation[:3]
        quat = observation[3:7]
        euler = p.getEulerFromQuaternion(quat)
        
        # Position bounds
        if any(abs(p) > 2.0 for p in pos):
            return False
        
        # Attitude bounds (45 degrees)
        if any(abs(e) > np.pi/4 for e in euler):
            return False
        
        return True

    def _get_target_point(self, t):
        """Get target point for trajectory tracking."""
        if self.trajectory is None:
            # Default circular trajectory
            radius = 1.0
            omega = 0.5  # rad/s
            x = radius * np.cos(omega * t)
            y = radius * np.sin(omega * t)
            z = 1.0
            return np.array([x, y, z])
        return self.trajectory(t)

    def set_trajectory(self, trajectory_func):
        """
        Set custom trajectory function.
        
        Args:
            trajectory_func: Function that takes time t as input and returns [x, y, z] position
        """
        self.trajectory = trajectory_func

    def close(self):
        """Clean up resources."""
        p.disconnect(self.client)