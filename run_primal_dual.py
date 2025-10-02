#!/usr/bin/env python3
"""
Runner for AR-GSE primal–dual (fixed-point alpha) training.
Uses src/train/train_argse.py as the main entrypoint.
"""

import sys
sys.path.append('.')

from src.train.train_argse import main as train_argse_main, CONFIG

if __name__ == '__main__':
    print("🚀 AR-GSE Primal–Dual (Fixed-Point α) Training")
    print("=" * 60)
    print(f"Dataset: {CONFIG['dataset']['name']}")
    print(f"Experts: {CONFIG['experts']['names']}")
    print("Mode: worst-group by default (CONFIG['argse_params']['mode'])")
    print("Note: Dual updates (rho) are off; α uses fixed-point updates for stability.")
    print("=" * 60)
    train_argse_main()


