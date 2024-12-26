import numpy as np

class Figure8Reference:
    def __init__(self, A=0.5, w=2*np.pi/4):
        self.A = A
        self.w = w

    def generate_reference(self, t):
        """Generate reference with smooth start for control"""
        t = float(t)
        smooth_start = np.minimum(t/1.0, 1.0)
        
        x_ref = np.zeros(12)
        x_ref[0] = self.A * np.sin(self.w*t) * smooth_start
        x_ref[2] = self.A * np.sin(2*self.w*t)/2 * smooth_start
        x_ref[6] = self.A * self.w * np.cos(self.w*t) * smooth_start 
        x_ref[8] = self.A * self.w * np.cos(2*self.w*t) * smooth_start
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
        """Get pure figure-8 points for visualization (no smooth start)"""
        points = np.zeros((len(t_array), 12))
        for i, t in enumerate(t_array):
            t = float(t)
            points[i, 0] = self.A * np.sin(self.w*t)
            points[i, 2] = self.A * np.sin(2*self.w*t)/2
        return points