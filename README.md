# Optimal Transport and Logistic Optimization

This repository contains the code for the paper "Optimal Transport and Logistic Optimization: A Framework for Commodity Reserve Distribution" by Marc Daghar.

## Overview

The framework implements:
1. Monge-Kantorovich optimal transport: T(μ,ν) = inf∫c(x,y)dπ
2. Entropic regularization: T_ε(μ,ν) = inf∫c + ε∫πlogπ
3. Sinkhorn algorithm: u^(k+1) = μ/(K·v^(k)), v^(k) = ν/(K^T·u^(k))
4. Multi-objective optimization for storage location
5. Resilience metrics: redundancy, diversity, concentration

## Requirements

```bash
pip install -r requirements.txt
