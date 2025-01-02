import numpy as np

class Figure8Reference:
    def __init__(self, A=0.5, w=2*np.pi/6):
        self.A = A
        self.w = w

    def generate_reference(self, t):
        """Generate reference state for time t"""
        t = float(t)
        smooth_start = np.minimum(t/1.0, 1.0)
        
        x_ref = np.zeros(12)
        
        # Position reference (x, y, z)
        x_ref[0] = self.A * np.sin(self.w*t) * smooth_start        # x
        x_ref[1] = 0.0                                             # y
        x_ref[2] = self.A * np.sin(2*self.w*t)/2 * smooth_start   # z
        
        # Velocities (analytical derivatives)
        x_ref[6] = self.A * self.w * np.cos(self.w*t) * smooth_start           # x_dot
        x_ref[7] = 0.0                                                         # y_dot
        x_ref[8] = self.A * self.w * np.cos(2*self.w*t) * smooth_start        # z_dot
        
        # Debug prints
        # print(f"Reference generation at t={t:.2f}:")
        # print(f"Position ref: {x_ref[0:3]}")
        # print(f"Velocity ref: {x_ref[6:9]}")
        
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

    # def compute_nominal_control(self, t, quad):
    #     """Compute feed-forward control for figure-8"""
    #     smooth_start = np.minimum(t/1.0, 1.0)
        
    #     # Get accelerations
    #     x_ddot = -self.A * self.w**2 * np.sin(self.w*t) * smooth_start
    #     z_ddot = -self.A * (2*self.w)**2 * np.sin(2*self.w*t)/2 * smooth_start
        
    #     # Compute required forces in world frame (scaled by mass)
    #     f_x = quad.mass * x_ddot
    #     f_z = quad.mass * (z_ddot + quad.g)  # Add gravity compensation
        
    #     # Compute total thrust and pitch angle
    #     thrust_total = np.sqrt(f_x**2 + f_z**2)
    #     pitch = np.arctan2(-f_x, f_z)
        
    #     # Convert to motor commands (scaled by thrust constant)
    #     u_thrust = thrust_total / (4 * quad.kt)
    #     u_pitch = pitch * quad.el / (2 * quad.kt)
        
    #     # Debug prints
    #     # print(f"\nNominal Control Calculation (t={t:.2f}):")
    #     # print(f"Forces - fx: {f_x:.4f}, fz: {f_z:.4f}")
    #     # print(f"Total thrust: {thrust_total:.4f}")
    #     # print(f"Pitch angle: {np.degrees(pitch):.2f} degrees")
    #     # print(f"u_thrust: {u_thrust:.4f}")
    #     # print(f"u_pitch: {u_pitch:.4f}")
        
    #     # Compute final control (differential thrust for pitch)
    #     control = np.array([
    #         u_thrust + u_pitch,  # Front motors
    #         u_thrust + u_pitch,
    #         u_thrust - u_pitch,  # Back motors
    #         u_thrust - u_pitch
    #     ])
        
    #     #print(f"Final control values: {control}")
    #     return control
