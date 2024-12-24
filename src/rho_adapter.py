import numpy as np
from scipy.linalg import block_diag

class RhoAdapter:
    def __init__(self, rho_base=1.0, rho_min=1.0, rho_max=10000.0, tolerance=1.1):
        self.rho_base = rho_base
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.tolerance = tolerance
        
        # For analysis
        self.rho_history = []
        self.residual_history = []

    def format_matrices(self, x_prev, u_prev, v_prev, z_prev, g_prev, y_prev, cache, N):
        """Format matrices into the form needed for residual computation"""
        nx = x_prev.shape[0]
        nu = u_prev.shape[0]
        
        # 1. Form decision variable x
        x_decision = []
        for i in range(N):
            x_decision.append(x_prev[:, i].reshape(-1, 1))
            if i < N-1:
                x_decision.append(u_prev[:, i].reshape(-1, 1))
        x = np.vstack(x_decision)

        # 2. Form constraint matrices
        A_dynamics = []
        A_inputs = []
        A_base = cache['A']
        B_base = cache['B']

        for i in range(N-1):
            # Dynamics constraints
            dyn_block = np.zeros((nx, (nx+nu)*(N-1) + nx))
            col_idx = i*(nx+nu)
            dyn_block[:, col_idx:col_idx+nx] = A_base
            dyn_block[:, col_idx+nx:col_idx+nx+nu] = B_base
            dyn_block[:, col_idx+nx+nu:col_idx+2*nx+nu] = -np.eye(nx)
            A_dynamics.append(dyn_block)
            
            # Input constraints
            input_block = np.zeros((nu, (nx+nu)*(N-1) + nx))
            input_block[:, col_idx+nx:col_idx+nx+nu] = np.eye(nu)
            A_inputs.append(input_block)

        A = np.vstack([np.vstack(A_inputs), np.vstack(A_dynamics)])

        # 3. Form z vector
        z_inputs = []
        z_dynamics = []
        for i in range(N-1):
            z_inputs.append(z_prev[:, i].reshape(-1, 1))
            z_dynamics.append(v_prev[:, i].reshape(-1, 1))
        z = np.vstack([np.vstack(z_inputs), np.vstack(z_dynamics)])

        # 4. Form y vector
        y_inputs = []
        y_dynamics = []
        for i in range(N-1):
            y_inputs.append(y_prev[:, i].reshape(-1, 1))
            y_dynamics.append(g_prev[:, i].reshape(-1, 1))
        y = np.vstack([np.vstack(y_inputs), np.vstack(y_dynamics)])

        return x, A, z, y

    def compute_residuals(self, x, A, z, y, P, q):
        """Compute ADMM residuals"""
        # Primal residual
        Ax = A @ x
        r_prim = Ax - z
        pri_res = np.linalg.norm(r_prim, ord=np.inf)
        pri_norm = max(np.linalg.norm(Ax, ord=np.inf), 
                      np.linalg.norm(z, ord=np.inf))

        # Dual residual
        r_dual = P @ x + q + A.T @ y
        dual_res = np.linalg.norm(r_dual, ord=np.inf)
        
        # Normalization terms
        Px = P @ x
        ATy = A.T @ y
        dual_norm = max(np.linalg.norm(Px, ord=np.inf),
                       np.linalg.norm(ATy, ord=np.inf),
                       np.linalg.norm(q, ord=np.inf))

        return pri_res, dual_res, pri_norm, dual_norm

    def compute_taylor_terms(self, cache, eps=1e-4):
        """Compute Taylor expansion terms for rho update"""
        rho = cache['rho']
        
        def compute_lqr(rho_val):
            R_rho = cache['R'] + rho_val * np.eye(cache['R'].shape[0])
            A, B = cache['A'], cache['B']
            Q = cache['Q']
            
            P = np.copy(Q)
            for _ in range(10):
                K = np.linalg.inv(R_rho + B.T @ P @ B) @ B.T @ P @ A
                P = Q + A.T @ P @ (A - B @ K)
            
            K = np.linalg.inv(R_rho + B.T @ P @ B) @ B.T @ P @ A
            C1 = np.linalg.inv(R_rho + B.T @ P @ B)
            C2 = A - B @ K
            
            return K, P, C1, C2

        # Compute derivatives using finite differences
        K_plus, P_plus, C1_plus, C2_plus = compute_lqr(rho + eps)
        K_minus, P_minus, C1_minus, C2_minus = compute_lqr(rho - eps)
        
        derivatives = {
            'dKinf_drho': (K_plus - K_minus) / (2 * eps),
            'dPinf_drho': (P_plus - P_minus) / (2 * eps),
            'dC1_drho': (C1_plus - C1_minus) / (2 * eps),
            'dC2_drho': (C2_plus - C2_minus) / (2 * eps)
        }
        
        return derivatives

    def predict_rho(self, pri_res, dual_res, pri_norm, dual_norm, current_rho):
        """Predict new rho value based on residuals"""
        normalized_pri = pri_res / (pri_norm + 1e-10)
        normalized_dual = dual_res / (dual_norm + 1e-10)
        
        rho_new = current_rho * np.sqrt(normalized_pri / (normalized_dual + 1e-10))
        rho_new = np.clip(rho_new, self.rho_min, self.rho_max)
        
        self.rho_history.append(rho_new)
        return rho_new

    def update_matrices(self, cache, new_rho, derivatives):
        """Update matrices using Taylor expansion"""
        old_rho = cache['rho']
        delta_rho = new_rho - old_rho
        
        updates = {
            'rho': new_rho,
            'Kinf': cache['Kinf'] + delta_rho * derivatives['dKinf_drho'],
            'Pinf': cache['Pinf'] + delta_rho * derivatives['dPinf_drho'],
            'C1': cache['C1'] + delta_rho * derivatives['dC1_drho'],
            'C2': cache['C2'] + delta_rho * derivatives['dC2_drho']
        }
        
        return updates