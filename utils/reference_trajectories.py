import numpy as np

class Figure8Reference:
    def __init__(self, A=0.5, w=2*np.pi/5):
        self.A = A
        self.w = w

    def generate_reference(self, t):
        """Generate figure-8 reference with smooth start"""
        # Convert t to float to avoid array operations
        t = float(t)
        
        # Smooth start factor (ramps up in first second)
        smooth_start = np.minimum(t/1.0, 1.0)  # using np.minimum instead of min
        
        x_ref = np.zeros(12)
        
        # Positions with smooth start
        x_ref[0] = self.A * np.sin(self.w*t) * smooth_start
        x_ref[2] = self.A * np.sin(2*self.w*t)/2 * smooth_start
        
        # Velocities (derivatives with smooth start)
        x_ref[6] = self.A * self.w * np.cos(self.w*t) * smooth_start 
        x_ref[8] = self.A * self.w * np.cos(2*self.w*t) * smooth_start
        
        # Zero attitude and angular velocity
        x_ref[3:6] = np.zeros(3)
        x_ref[9:12] = np.zeros(3)
        
        return x_ref

    def get_final_reference(self, t):
        """Get reference position at time t"""
        t = float(t)  # Convert to float here too
        return {
            'x': self.A * np.sin(self.w*t),
            'z': self.A * np.sin(2*self.w*t)/2
        }

    def get_trajectory_points(self, t_array):
        """Get array of reference points for visualization"""
        return np.array([self.generate_reference(float(t)) for t in t_array])