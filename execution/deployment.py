#!/usr/bin/env python3
"""
DEPLOYMENT MANAGER - Semua phase deployment management
"""

import logging
import json
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentManager:
    """Manage semua phase deployment"""
    
    def __init__(self):
        self.deployment_phases = {
            'PAPER_TRADING': {
                'duration_days': 7,
                'capital_allocation': 0.0,
                'success_criteria': {
                    'sharpe_ratio': 1.0,
                    'max_drawdown': -0.10,
                    'consistency_score': 0.60
                }
            },
            'SMALL_CAPITAL': {
                'duration_days': 14,
                'capital_allocation': 0.05,  # 5%
                'success_criteria': {
                    'sharpe_ratio': 1.2,
                    'max_drawdown': -0.08,
                    'win_rate': 0.50
                }
            },
            'FULL_DEPLOYMENT': {
                'duration_days': 30,
                'capital_allocation': 0.10,  # 10% (scale up gradually)
                'success_criteria': {
                    'sharpe_ratio': 1.5,
                    'max_drawdown': -0.15,
                    'win_rate': 0.55
                }
            }
        }
    
    def evaluate_phase_transition(self, current_phase, performance_data):
        """Evaluate jika ready untuk phase transition"""
        criteria = self.deployment_phases[current_phase]['success_criteria']
        
        meets_criteria = all([
            performance_data.get('sharpe_ratio', 0) >= criteria['sharpe_ratio'],
            performance_data.get('max_drawdown', 0) >= criteria['max_drawdown'],
            performance_data.get('win_rate', 0) >= criteria.get('win_rate', 0.4),
            performance_data.get('consistency_score', 0) >= criteria.get('consistency_score', 0.5)
        ])
        
        return meets_criteria
    
    def get_next_phase(self, current_phase):
        """Get next phase dalam deployment pipeline"""
        phases = list(self.deployment_phases.keys())
        current_index = phases.index(current_phase)
        
        if current_index < len(phases) - 1:
            return phases[current_index + 1]
        return None

# Usage example
if __name__ == "__main__":
    manager = DeploymentManager()
    
    # Simulate performance evaluation
    performance = {
        'sharpe_ratio': 2.1,
        'max_drawdown': -0.05,
        'win_rate': 0.58,
        'consistency_score': 0.72
    }
    
    ready_for_next = manager.evaluate_phase_transition('PAPER_TRADING', performance)
    next_phase = manager.get_next_phase('PAPER_TRADING')
    
    if ready_for_next:
        logger.info(f"✅ Ready to advance to {next_phase} phase!")
    else:
        logger.info("⏳ Continue current phase - criteria not yet met")