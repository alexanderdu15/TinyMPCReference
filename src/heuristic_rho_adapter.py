class HeuristicRhoAdapter:
    def __init__(self, rho_base=85.0, rho_min=70.0, rho_max=100.0):
        self.rho_base = rho_base
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.rho_history = []

    def initialize_derivatives(self, cache, eps=1e-4):
        """Initialize derivatives using autodiff"""
        print("Computing LQR sensitivity")
        
        def lqr_direct(rho):
            R_rho = cache['R'] + rho * np.eye(cache['R'].shape[0])
            A, B = cache['A'], cache['B']
            Q = cache['Q']
            
            # Compute P
            P = Q
            for _ in range(10):
                K = np.linalg.inv(R_rho + B.T @ P @ B) @ B.T @ P @ A
                P = Q + A.T @ P @ (A - B @ K)
            
            K = np.linalg.inv(R_rho + B.T @ P @ B) @ B.T @ P @ A
            C1 = np.linalg.inv(R_rho + B.T @ P @ B)
            C2 = A - B @ K
            
            return np.concatenate([K.flatten(), P.flatten(), C1.flatten(), C2.flatten()])
        
        # Get derivatives using autodiff
        m, n = cache['Kinf'].shape
        derivs = jacobian(lqr_direct)(cache['rho'])
        
        # Reshape derivatives into matrices and store in cache
        k_size = m * n
        p_size = n * n
        c1_size = m * m
        c2_size = n * n
        
        # Store derivatives in the cache directly
        cache['dKinf_drho'] = derivs[:k_size].reshape(m, n)
        cache['dPinf_drho'] = derivs[k_size:k_size+p_size].reshape(n, n)
        cache['dC1_drho'] = derivs[k_size+p_size:k_size+p_size+c1_size].reshape(m, m)
        cache['dC2_drho'] = derivs[k_size+p_size+c1_size:].reshape(n, n)

    def predict_rho(self, pri_res, dual_res, iterations, current_rho):
        """Heuristic adaptation based on primal-dual residual ratio"""
        print(f"Pri Res: {pri_res}")
        ratio = pri_res / (dual_res + 1e-8)
        
        old_rho = current_rho
        
        # Normal adaptation based on residual ratio
        if ratio > 2.0:  # Primal residual much larger
            new_rho = min(current_rho * 1.1, self.max_rho)
        elif ratio < 3.0:  # Dual residual much larger
            new_rho = max(current_rho * 0.9, self.min_rho)
        else:
            new_rho = current_rho

        if abs(new_rho - old_rho) > 1e-6:
            print(f"\nRho adaptation:")
            print(f"Ratio: {ratio}")
            print(f"Old rho: {old_rho}, New rho: {new_rho}")
        
        self.rho_history.append(new_rho)
        return new_rho

    def update_matrices(self, cache, new_rho):
        """Update matrices using derivatives stored in cache"""
        old_rho = cache['rho']
        delta_rho = new_rho - old_rho
        
        updates = {
            'rho': new_rho,
            'Kinf': cache['Kinf'] + delta_rho * cache['dKinf_drho'],
            'Pinf': cache['Pinf'] + delta_rho * cache['dPinf_drho'],
            'C1': cache['C1'] + delta_rho * cache['dC1_drho'],
            'C2': cache['C2'] + delta_rho * cache['dC2_drho']
        }
        
        return updates