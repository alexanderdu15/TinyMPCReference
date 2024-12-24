import math
import autograd.numpy as np
from autograd.numpy.linalg import norm, inv
from autograd import jacobian

class QuadrotorDynamics:
    def __init__(self):
        # Quadrotor parameters
        self.mass = 0.035
        self.J = np.array([[1.66e-5, 0.83e-6, 0.72e-6], 
                          [0.83e-6, 1.66e-5, 1.8e-6], 
                          [0.72e-6, 1.8e-6, 2.93e-5]])
        self.g = 9.81
        self.thrustToTorque = 0.0008
        self.el = 0.046/1.414213562
        self.scale = 65535
        self.kt = 2.245365e-6 * self.scale
        self.km = self.kt * self.thrustToTorque
        
        # Simulation parameters
        self.freq = 50.0
        self.dt = 1/self.freq
        
        # State and input dimensions
        self.nx = 12  # State dimension
        self.nu = 4   # Input dimension
        
        # Hover thrust
        self.hover_thrust = (self.mass * self.g / self.kt / 4) * np.ones(4)

    def get_linearized_dynamics(self, x_ref, u_ref):
        """Get linearized dynamics matrices around reference point"""
        A_jac = jacobian(self.dynamics_rk4, 0)
        B_jac = jacobian(self.dynamics_rk4, 1)
        
        A = A_jac(x_ref, u_ref)
        B = B_jac(x_ref, u_ref)
        
        return self.E(x_ref[3:7]).T @ A @ self.E(x_ref[3:7]), self.E(x_ref[3:7]).T @ B

    def dynamics(self, x, u, wind=np.array([0.0, 0.0, 0.0])):
        """Continuous-time dynamics"""
        r = x[0:3]    # position
        q = x[3:7]/norm(x[3:7])   # normalized quaternion
        v = x[7:10]   # velocity
        omega = x[10:13]  # angular velocity
        
        Q = self.qtoQ(q)
        
        # Position dynamics
        dr = v
        
        # Quaternion dynamics
        dq = 0.5 * self.L(q) @ self.H @ omega
        
        # Velocity dynamics
        thrust_matrix = np.array([[0, 0, 0, 0], 
                                [0, 0, 0, 0], 
                                [self.kt, self.kt, self.kt, self.kt]])
        dv = np.array([0, 0, -self.g]) + (1/self.mass) * Q @ thrust_matrix @ u + wind
        
        # Angular velocity dynamics
        moment_matrix = np.array([[-self.el*self.kt, -self.el*self.kt, self.el*self.kt, self.el*self.kt],
                                [-self.el*self.kt, self.el*self.kt, self.el*self.kt, -self.el*self.kt],
                                [-self.km, self.km, -self.km, self.km]])
        domega = inv(self.J) @ (-self.hat(omega) @ self.J @ omega + moment_matrix @ u)
        
        return np.hstack([dr, dq, dv, domega])

    def dynamics_rk4(self, x, u):
        """Discrete-time dynamics using RK4 integration"""
        f1 = self.dynamics(x, u)
        f2 = self.dynamics(x + 0.5*self.dt*f1, u)
        f3 = self.dynamics(x + 0.5*self.dt*f2, u)
        f4 = self.dynamics(x + self.dt*f3, u)
        
        xn = x + (self.dt/6.0)*(f1 + 2*f2 + 2*f3 + f4)
        xn[3:7] = xn[3:7]/norm(xn[3:7])  # normalize quaternion
        
        return xn

    # Quaternion utilities
    @staticmethod
    def hat(v):
        """Skew-symmetric matrix from vector"""
        return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0.0]])

    @staticmethod
    def L(q):
        """Left quaternion multiplication matrix"""
        s = q[0]
        v = q[1:4]
        return np.vstack([
            np.hstack([s, -v]),
            np.hstack([v.reshape(3,1), s*np.eye(3) + QuadrotorDynamics.hat(v)])
        ])

    # Constants for quaternion operations
    T = np.diag([1.0, -1, -1, -1])
    H = np.vstack([np.zeros((1,3)), np.eye(3)])

    @classmethod
    def qtoQ(cls, q):
        """Quaternion to rotation matrix"""
        return cls.H.T @ cls.T @ cls.L(q) @ cls.T @ cls.L(q) @ cls.H

    @classmethod
    def G(cls, q):
        """G matrix for quaternion dynamics"""
        return cls.L(q) @ cls.H

    @staticmethod
    def rptoq(phi):
        """Rodriguez parameters to quaternion"""
        return (1./math.sqrt(1 + phi.T @ phi)) * np.hstack([1, phi])

    @staticmethod
    def qtorp(q):
        """Quaternion to Rodriguez parameters"""
        return q[1:4]/q[0]

    @classmethod
    def E(cls, q):
        """Error state mapping matrix"""
        return np.vstack([
            np.hstack([np.eye(3), np.zeros((3,3)), np.zeros((3,6))]),
            np.hstack([np.zeros((4,3)), cls.G(q), np.zeros((4,6))]),
            np.hstack([np.zeros((6,3)), np.zeros((6,3)), np.eye(6)])
        ])