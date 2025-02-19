import numpy as np
import pybullet as p

class PIDController:
    """
    Advanced PID controller for quadrotor control with dynamic compensation.
    Implements a cascaded control structure: position -> attitude -> motor commands.
    """
    
    def __init__(self):
        # Physical parameters matching the environment
        self.MASS = 0.027  # kg - must match environment
        self.GRAVITY = 9.81
        self.L = 0.039  # arm length in meters
        self.KF = 3.16e-10  # thrust coefficient
        self.KM = 7.94e-12  # moment coefficient
        self.MAX_RPM = 44000
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        
        # Position controller gains (outer loop)
        # Tuned for good tracking while maintaining stability
        self.Kp_pos = np.array([10.0, 10.0, 20.0])  # Higher Z gain for altitude
        self.Ki_pos = np.array([2.0, 2.0, 5.0])     # Integral helps with steady-state
        self.Kd_pos = np.array([7.0, 7.0, 10.0])    # Derivative for damping
        
        # Attitude controller gains (inner loop)
        # Higher gains for faster attitude response
        self.Kp_att = np.array([20.0, 20.0, 10.0])  # Roll, Pitch, Yaw
        self.Ki_att = np.array([0.0, 0.0, 0.0])     # Usually not needed for attitude
        self.Kd_att = np.array([10.0, 10.0, 5.0])   # Damping for stability
        
        # Error integrals for position and attitude
        self.pos_integral = np.zeros(3)
        self.att_integral = np.zeros(3)
        
        # Previous errors for derivative computation
        self.prev_pos_error = np.zeros(3)
        self.prev_att_error = np.zeros(3)
        
        # Anti-windup limits
        self.pos_integral_limit = 2.0
        self.att_integral_limit = 1.0
        
        # Time step matching environment
        self.dt = 0.01  # 100Hz control
        
        # Previous motor commands for smoothing
        self.prev_pwm = np.ones(4) * self.MIN_PWM
        
        # Mixing matrix for quadrotor in X configuration
        # Maps desired thrust and torques to individual motor commands
        self.mixing_matrix = np.array([
            [1.0, 1.0, 1.0, 1.0],     # Total thrust
            [1.0, -1.0, -1.0, 1.0],   # Roll torque
            [1.0, 1.0, -1.0, -1.0],   # Pitch torque
            [-1.0, 1.0, -1.0, 1.0]    # Yaw torque
        ])
        
    def reset(self):
        """Reset controller state for new episode."""
        self.pos_integral = np.zeros(3)
        self.att_integral = np.zeros(3)
        self.prev_pos_error = np.zeros(3)
        self.prev_att_error = np.zeros(3)
        self.prev_pwm = np.ones(4) * self.MIN_PWM

    def compute_control(self, state, target_pos, target_yaw=0.0):
        """
        Compute PWM commands for motors based on current state and target.
        
        Args:
            state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
            target_pos: Desired position [x, y, z]
            target_yaw: Desired yaw angle (radians)
            
        Returns:
            numpy.array: PWM commands for each motor [pwm1, pwm2, pwm3, pwm4]
        """
        # Extract state components
        pos = state[:3]
        quat = state[3:7]
        vel = state[7:10]
        ang_vel = state[10:]
        
        # Get rotation matrix and current euler angles
        rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        euler = np.array(p.getEulerFromQuaternion(quat))
        
        # Position control (outer loop)
        pos_error = target_pos - pos
        
        # Update position integral with anti-windup
        self.pos_integral += pos_error * self.dt
        self.pos_integral = np.clip(
            self.pos_integral, 
            -self.pos_integral_limit, 
            self.pos_integral_limit
        )
        
        # Compute position derivative with velocity feedback
        pos_derivative = (pos_error - self.prev_pos_error) / self.dt - vel
        
        # Desired acceleration in world frame
        acc_desired = (
            self.Kp_pos * pos_error +
            self.Ki_pos * self.pos_integral +
            self.Kd_pos * pos_derivative +
            np.array([0., 0., self.GRAVITY])  # Gravity compensation
        )
        
        # Convert desired acceleration to thrust and attitude commands
        thrust_magnitude = self.MASS * np.linalg.norm(acc_desired)
        
        # Compute desired orientation
        z_body_desired = acc_desired / np.linalg.norm(acc_desired)
        x_body_desired = np.array([
            np.cos(target_yaw),
            np.sin(target_yaw),
            0
        ])
        y_body_desired = np.cross(z_body_desired, x_body_desired)
        y_body_desired = y_body_desired / np.linalg.norm(y_body_desired)
        x_body_desired = np.cross(y_body_desired, z_body_desired)
        
        rot_mat_desired = np.vstack([x_body_desired, y_body_desired, z_body_desired]).T
        euler_desired = self._rot_mat_to_euler(rot_mat_desired)
        
        # Attitude control (inner loop)
        att_error = self._wrap_angle(euler_desired - euler)
        
        # Update attitude integral with anti-windup
        self.att_integral += att_error * self.dt
        self.att_integral = np.clip(
            self.att_integral,
            -self.att_integral_limit,
            self.att_integral_limit
        )
        
        # Compute attitude derivative with angular velocity feedback
        att_derivative = (att_error - self.prev_att_error) / self.dt - ang_vel
        
        # Compute desired torques
        torques = (
            self.Kp_att * att_error +
            self.Ki_att * self.att_integral +
            self.Kd_att * att_derivative
        )
        
        # Combine thrust and torques
        thrust_torques = np.array([
            thrust_magnitude,
            torques[0] * self.L,
            torques[1] * self.L,
            torques[2]
        ])
        
        # Convert to motor PWM commands
        pwm = self._thrust_torques_to_pwm(thrust_torques)
        
        # Apply motor dynamics (smoothing)
        alpha = 0.2  # Smoothing factor
        pwm = alpha * pwm + (1 - alpha) * self.prev_pwm
        
        # Store values for next iteration
        self.prev_pos_error = pos_error
        self.prev_att_error = att_error
        self.prev_pwm = pwm
        
        return np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
    
    def _thrust_torques_to_pwm(self, thrust_torques):
        """Convert desired thrust and torques to PWM commands."""
        # Convert thrust and torques to individual motor thrusts
        motor_thrusts = np.linalg.pinv(self.mixing_matrix) @ thrust_torques
        
        # Convert thrusts to RPM
        rpms = np.sqrt(np.maximum(motor_thrusts / self.KF, 0))
        
        # Convert RPM to PWM
        return (rpms - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
    
    def _rot_mat_to_euler(self, R):
        """Convert rotation matrix to euler angles (roll, pitch, yaw)."""
        # Extract pitch angle (about y axis)
        pitch = np.arcsin(-R[2, 0])
        
        # Check for gimbal lock
        if abs(pitch - np.pi/2) < 1e-3:
            roll = 0  # Roll is undefined, set to zero
            yaw = np.arctan2(R[1, 2], R[0, 2])
        elif abs(pitch + np.pi/2) < 1e-3:
            roll = 0  # Roll is undefined, set to zero
            yaw = -np.arctan2(R[1, 2], R[0, 2])
        else:
            # Extract roll (about x axis) and yaw (about z axis)
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        
        return np.array([roll, pitch, yaw])
    
    def _wrap_angle(self, angle):
        """Wrap angle to [-pi, pi]."""
        return np.mod(angle + np.pi, 2 * np.pi) - np.pi