import numpy as np

class PIDController:
    """
    Improved PID controller for quadrotor control with better tuned gains.
    Implements cascaded PID control structure.
    """
    
    def __init__(self):
        # Position control gains - Increased for better tracking
        self.Kp_pos = np.array([3.0, 3.0, 6.0])  # Higher Z gain for altitude
        self.Ki_pos = np.array([0.4, 0.4, 0.8])  # Integral gain to eliminate steady-state error
        self.Kd_pos = np.array([2.0, 2.0, 3.0])  # Derivative for damping
        
        # Attitude control gains - More aggressive for stability
        self.Kp_att = np.array([6.0, 6.0, 4.0])  # Higher roll/pitch gains
        self.Ki_att = np.array([0.3, 0.3, 0.2])
        self.Kd_att = np.array([1.5, 1.5, 1.0])
        
        # Initialize error integrals
        self.pos_integral = np.zeros(3)
        self.att_integral = np.zeros(3)
        
        # Store previous errors for derivative term
        self.prev_pos_error = np.zeros(3)
        self.prev_att_error = np.zeros(3)
        
        # Anti-windup limits
        self.integral_limit = 1.0
        
        # Time step
        self.dt = 0.01

    def reset(self):
        """Reset controller state."""
        self.pos_integral = np.zeros(3)
        self.att_integral = np.zeros(3)
        self.prev_pos_error = np.zeros(3)
        self.prev_att_error = np.zeros(3)

    def compute_control(self, state, target_pos, target_yaw=0.0):
        """
        Compute control actions for the quadrotor.
        
        Args:
            state: Current state [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
            target_pos: Target position [x, y, z]
            target_yaw: Target yaw angle
        """
        # Extract current position and attitude
        current_pos = state[:3]
        current_att = state[3:6]
        current_vel = state[6:9]
        current_angular_vel = state[9:]
        
        # Position control
        pos_error = target_pos - current_pos
        
        # Anti-windup for position integral
        self.pos_integral += pos_error * self.dt
        self.pos_integral = np.clip(self.pos_integral, -self.integral_limit, self.integral_limit)
        
        pos_derivative = (pos_error - self.prev_pos_error) / self.dt
        
        # Add velocity damping
        pos_derivative += -0.5 * current_vel
        
        # Compute desired acceleration
        desired_acc = (
            self.Kp_pos * pos_error +
            self.Ki_pos * self.pos_integral +
            self.Kd_pos * pos_derivative
        )
        
        # Add feed-forward gravity compensation
        desired_acc[2] += 9.81
        
        # Convert desired acceleration to desired attitude
        thrust_magnitude = np.linalg.norm(desired_acc)
        
        # Prevent division by zero and limit maximum tilt
        if thrust_magnitude > 0.01:
            desired_roll = np.arcsin(np.clip(desired_acc[1] / thrust_magnitude, -0.5, 0.5))
            desired_pitch = -np.arcsin(np.clip(desired_acc[0] / thrust_magnitude, -0.5, 0.5))
        else:
            desired_roll = 0
            desired_pitch = 0
            
        desired_att = np.array([desired_roll, desired_pitch, target_yaw])
        
        # Attitude control
        att_error = desired_att - current_att
        att_error[2] = (att_error[2] + np.pi) % (2 * np.pi) - np.pi  # Normalize yaw error
        
        # Anti-windup for attitude integral
        self.att_integral += att_error * self.dt
        self.att_integral = np.clip(self.att_integral, -self.integral_limit, self.integral_limit)
        
        att_derivative = (att_error - self.prev_att_error) / self.dt
        
        # Add angular velocity damping
        att_derivative += -0.5 * current_angular_vel
        
        # Compute torques
        torques = (
            self.Kp_att * att_error +
            self.Ki_att * self.att_integral +
            self.Kd_att * att_derivative
        )
        
        # Update previous errors
        self.prev_pos_error = pos_error
        self.prev_att_error = att_error
        
        # Normalize thrust to [-1, 1]
        normalized_thrust = (thrust_magnitude - 9.81) / 20.0
        normalized_thrust = np.clip(normalized_thrust, -1.0, 1.0)
        
        # Normalize torques to [-1, 1]
        normalized_torques = np.clip(torques, -1.0, 1.0)
        
        return np.array([normalized_thrust, *normalized_torques])