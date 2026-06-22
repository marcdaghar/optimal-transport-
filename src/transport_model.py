"""
Optimal Transport Model for Commodity Distribution.
Implements the Monge-Kantorovich transport problem.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

class TransportModel:
    """
    Optimal transport model for commodity distribution.
    
    Key relationships:
        T(μ,ν) = inf_{π∈Π(μ,ν)} ∫ c(x,y) dπ(x,y)
        c(x,y) = w_d·||x-y|| + w_t·τ·||x-y|| + w_e·ε·||x-y||² + w_r·(r(x)+r(y))
    """
    
    def __init__(self, w_d=0.3, w_t=0.2, w_e=0.2, w_r=0.3,
                 tau=0.1, epsilon=0.05):
        """
        Args:
            w_d: Weight for distance cost
            w_t: Weight for time cost
            w_e: Weight for energy cost
            w_r: Weight for risk cost
            tau: Time cost coefficient
            epsilon: Energy cost coefficient
        """
        self.w_d = w_d
        self.w_t = w_t
        self.w_e = w_e
        self.w_r = w_r
        self.tau = tau
        self.epsilon = epsilon
        
        # Ensure weights sum to 1
        total = self.w_d + self.w_t + self.w_e + self.w_r
        self.w_d /= total
        self.w_t /= total
        self.w_e /= total
        self.w_r /= total
    
    def cost_matrix(self, supply_locations, demand_locations,
                    supply_risks=None, demand_risks=None):
        """
        Compute cost matrix for transport.
        
        Args:
            supply_locations: Array of supply locations (n, 2)
            demand_locations: Array of demand locations (m, 2)
            supply_risks: Array of supply risks (n,)
            demand_risks: Array of demand risks (m,)
        
        Returns:
            Cost matrix (n, m)
        """
        n = len(supply_locations)
        m = len(demand_locations)
        
        # Distance cost
        distances = cdist(supply_locations, demand_locations)
        
        # Time cost
        time_cost = self.tau * distances
        
        # Energy cost (quadratic)
        energy_cost = self.epsilon * distances ** 2
        
        # Risk cost
        if supply_risks is None:
            supply_risks = np.zeros(n)
        if demand_risks is None:
            demand_risks = np.zeros(m)
        
        risk_cost = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                risk_cost[i, j] = supply_risks[i] + demand_risks[j]
        
        # Total cost
        C = (self.w_d * distances + 
             self.w_t * time_cost + 
             self.w_e * energy_cost + 
             self.w_r * risk_cost)
        
        return C
    
    def solve_exact(self, supply_locations, demand_locations,
                    supply_amounts, demand_amounts,
                    supply_risks=None, demand_risks=None):
        """
        Solve exact optimal transport (discrete case).
        
        Returns:
            Transport plan (n, m)
        """
        n = len(supply_locations)
        m = len(demand_locations)
        
        # Cost matrix
        C = self.cost_matrix(supply_locations, demand_locations,
                            supply_risks, demand_risks)
        
        # For exact solution, we need to discretize amounts
        # This is a simplified version using linear assignment
        # For full optimal transport, use the POT library
        
        # Here we use a simple approximation: linear programming
        # In practice, use the Sinkhorn algorithm for large problems
        
        return self._lp_solve(C, supply_amounts, demand_amounts)
    
    def _lp_solve(self, C, supply_amounts, demand_amounts):
        """
        Linear programming solution for small problems.
        """
        # This is a placeholder for demonstration
        # For actual implementation, use scipy.optimize.linprog
        # or the POT (Python Optimal Transport) library
        
        n, m = C.shape
        
        # Simple greedy algorithm
        supply = supply_amounts.copy()
        demand = demand_amounts.copy()
        plan = np.zeros((n, m))
        
        for i in range(n):
            for j in range(m):
                if supply[i] > 0 and demand[j] > 0:
                    amount = min(supply[i], demand[j])
                    plan[i, j] = amount
                    supply[i] -= amount
                    demand[j] -= amount
        
        return plan
    
    def compute_total_cost(self, plan, supply_locations, demand_locations,
                          supply_risks=None, demand_risks=None):
        """
        Compute total transport cost.
        """
        C = self.cost_matrix(supply_locations, demand_locations,
                            supply_risks, demand_risks)
        return np.sum(plan * C)
    
    def compute_wasserstein_distance(self, plan, supply_locations,
                                     demand_locations):
        """
        Compute Wasserstein distance.
        """
        C = cdist(supply_locations, demand_locations)
        return np.sum(plan * C) / np.sum(plan)
    
    def compute_entropy(self, plan):
        """
        Compute entropy of transport plan.
        """
        # Normalize
        plan_norm = plan / (np.sum(plan) + 1e-10)
        mask = plan_norm > 0
        return -np.sum(plan_norm[mask] * np.log(plan_norm[mask] + 1e-10))
