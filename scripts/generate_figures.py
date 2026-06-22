#!/usr/bin/env python3
"""
Generate all figures from saved results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from src.visualization import FigureGenerator

def main():
    """
    Load saved results and generate figures.
    """
    print("Loading saved results...")
    
    with open('data/transport_results.pkl', 'rb') as f:
        transport_results = pickle.load(f)
    
    with open('data/storage_results.pkl', 'rb') as f:
        storage_results = pickle.load(f)
    
    with open('data/comparison_results.pkl', 'rb') as f:
        comparison_results = pickle.load(f)
    
    results_optimal = comparison_results['optimal']
    results_heuristic = comparison_results['heuristic']
    
    print("Generating figures...")
    fig_gen = FigureGenerator()
    
    fig_gen.figure_all(
        transport_results['supply_locations'],
        transport_results['demand_locations'],
        transport_results['plan'],
        transport_results['supply_amounts'],
        transport_results['demand_amounts'],
        transport_results['history'],
        storage_results['demand_nodes'],
        storage_results['potential_locations'],
        storage_results['selected_locations'],
        storage_results['capacities'],
        results_optimal,
        results_heuristic
    )
    
    print("All figures generated successfully!")

if __name__ == "__main__":
    main()
