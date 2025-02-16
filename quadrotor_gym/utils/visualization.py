import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TrajectoryVisualizer:
    """
    Utility class for visualizing quadrotor trajectories.
    Provides both 2D time-series plots and 3D trajectory visualization.
    """
    
    def __init__(self):
        self.position_history = []
        self.target_history = []
        self.time_history = []
        self.attitude_history = []
    
    def reset(self):
        """Reset all visualization history."""
        self.position_history = []
        self.target_history = []
        self.time_history = []
        self.attitude_history = []
    
    def update(self, position, target, time, attitude=None):
        """
        Update visualization with new state information.
        
        Args:
            position (np.array): Current position [x, y, z]
            target (np.array): Target position [x, y, z]
            time (float): Current time
            attitude (np.array, optional): Current attitude [roll, pitch, yaw]
        """
        self.position_history.append(position)
        self.target_history.append(target)
        self.time_history.append(time)
        if attitude is not None:
            self.attitude_history.append(attitude)
    
    def plot_2d(self, save_path=None):
        """
        Create 2D plots showing position and target trajectories over time.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        positions = np.array(self.position_history)
        targets = np.array(self.target_history)
        times = np.array(self.time_history)
        
        # Create figure with position and error plots
        fig, axes = plt.subplots(4, 1, figsize=(12, 16))
        
        # Plot position vs time for each axis
        labels = ['X', 'Y', 'Z']
        for i, (ax, label) in enumerate(zip(axes[:3], labels)):
            ax.plot(times, positions[:, i], 'b-', label=f'Actual {label}')
            ax.plot(times, targets[:, i], 'r--', label=f'Target {label}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'{label} Position (m)')
            ax.legend()
            ax.grid(True)
        
        # Plot position error
        errors = np.linalg.norm(positions - targets, axis=1)
        axes[3].plot(times, errors, 'g-', label='Position Error')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Error (m)')
        axes[3].legend()
        axes[3].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_3d(self, save_path=None):
        """
        Create 3D visualization of the flight trajectory.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        positions = np.array(self.position_history)
        targets = np.array(self.target_history)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot actual trajectory
        ax.plot3D(positions[:, 0], positions[:, 1], positions[:, 2], 
                 'b-', label='Actual Trajectory')
        
        # Plot target trajectory
        ax.plot3D(targets[:, 0], targets[:, 1], targets[:, 2], 
                 'r--', label='Target Trajectory')
        
        # Plot start and end points
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                  c='g', marker='o', label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                  c='r', marker='o', label='End')
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Quadrotor Trajectory')
        ax.legend()
        
        # Make the plot aspect ratio equal
        max_range = np.array([
            positions[:, 0].max() - positions[:, 0].min(),
            positions[:, 1].max() - positions[:, 1].min(),
            positions[:, 2].max() - positions[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_attitude(self, save_path=None):
        """
        Plot attitude (roll, pitch, yaw) over time.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        if not self.attitude_history:
            return
        
        attitudes = np.array(self.attitude_history)
        times = np.array(self.time_history)
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        labels = ['Roll', 'Pitch', 'Yaw']
        
        for i, (ax, label) in enumerate(zip(axes, labels)):
            ax.plot(times, np.degrees(attitudes[:, i]), 'b-', label=label)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'{label} (degrees)')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()