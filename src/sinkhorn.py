"""
Sinkhorn Algorithm for Entropic Regularized Optimal Transport.
"""

import numpy as np
from scipy.spatial.distance import cdist

class SinkhornSolver:
    """
    Sinkhorn algorithm for entropic regularized optimal transport.
    
    Key relationships:
        u^(0) = 1
        v^(k) = ν / (K^T · u^(k))
        u^(k+1) = μ / (K · v^(k))
        K_ij = exp(-c(x_i, y_j) / ε)
    """
    
    def __init__(self, epsilon=0.1, max_iter=1000, tol=1e-6):
        """
        Args:
            epsilon: Regularization parameter
            max_iter: Maximum iterations
            tol: Convergence tolerance
        """
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
    
    def compute_kernel(self, supply_locations, demand_locations,
                       cost_function=None):
        """
        Compute Gibbs kernel: K_ij = exp(-c(x_i, y_j) / ε)
        """
        if cost_function is None:
            # Default: squared Euclidean distance
            C = cdist(supply_locations, demand_locations) ** 2
        else:
            C = cost_function(supply_locations, demand_locations)
        
        K = np.exp(-C / self.epsilon)
        return K
    
    def sinkhorn(self, supply_locations, demand_locations,
                 supply_amounts, demand_amounts,
                 cost_function=None):
        """
        Run Sinkhorn algorithm.
        
        Returns:
            Transport plan, convergence history
        """
        n = len(supply_locations)
        m = len(demand_locations)
        
        # Compute kernel
        K = self.compute_kernel(supply_locations, demand_locations,
                               cost_function)
        
        # Initialize
        u = np.ones(n)
        v = np.ones(m)
        
        # Normalize supplies and demands
        mu = supply_amounts / np.sum(supply_amounts)
        nu = demand_amounts / np.sum(demand_amounts)
        
        history = []
        
        for iteration in range(self.max_iter):
            # Update v: v = ν / (K^T · u)
            v_new = nu / (K.T @ u)
            
            # Update u: u = μ / (K · v)
            u_new = mu / (K @ v_new)
            
            # Compute convergence
            diff_u = np.max(np.abs(u - u_new) / (np.abs(u) + 1e-10))
            diff_v = np.max(np.abs(v - v_new) / (np.abs(v) + 1e-10))
            diff = max(diff_u, diff_v)
            
            history.append(diff)
            
            u = u_new
            v = v_new
            
            if diff < self.tol:
                break
        
        # Compute transport plan
        plan = np.diag(u) @ K @ np.diag(v)
        plan = plan * np.sum(supply_amounts)  # Scale back to original units
        
        return plan, history
    
    def compute_entropy(self, plan):
        """
        Compute entropy of transport plan.
        """
        plan_norm = plan / (np.sum(plan) + 1e-10)
        mask = plan_norm > 0
        return -np.sum(plan_norm[mask] * np.log(plan_norm[mask] + 1e-10))
    
    def compute_total_cost(self, plan, supply_locations, demand_locations,
                          cost_function=None):
        """
        Compute total transport cost.
        """
        if cost_function is None:
            C = cdist(supply_locations, demand_locations) ** 2
        else:
            C = cost_function(supply_locations, demand_locations)
        
        return np.sum(plan * C) / np.sum(plan)
    
    def sinkhorn_with_cost(self, supply_locations, demand_locations,
                          supply_amounts, demand_amounts,
                          C_matrix):
        """
        Sinkhorn with precomputed cost matrix.
        """
        K = np.exp(-C_matrix / self.epsilon)
        
        mu = supply_amounts / np.sum(supply_amounts)
        nu = demand_amounts / np.sum(demand_amounts)
        
        u = np.ones(len(supply_locations))
        v = np.ones(len(demand_locations))
        history = []
        
        for iteration in range(self.max_iter):
            v_new = nu / (K.T @ u)
            u_new = mu / (K @ v_new)
            
            diff = max(np.max(np.abs(u - u_new) / (np.abs(u) + 1e-10)),
                      np.max(np.abs(v - v_new) / (np.abs(v) + 1e-10)))
            
            history.append(diff)
            u = u_new
            v = v_new
            
            if diff < self.tol:
                break
        
        plan = np.diag(u) @ K @ np.diag(v)
        plan = plan * np.sum(supply_amounts)
        
        return plan, history
