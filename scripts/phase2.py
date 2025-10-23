#!/usr/bin/env python3
"""
PHASE 2 MAIN - WITH JSON FIX
"""

import sys
from pathlib import Path
import json
import logging
import pandas as pd
import numpy as np

# Add to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))
sys.path.insert(0, str(current_dir / "config"))

from src.historical_validator import HistoricalValidator
from config.phase2_settings import SUPPORT_LEVELS_DB, VALIDATION_RESULTS

def convert_to_serializable(obj):
    """Convert non-serializable objects untuk JSON"""
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return str(obj)

def setup_logging():
    """Setup logging untuk Phase 2"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(current_dir / "logs" / "phase2_validation.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main Phase 2 execution - FIXED VERSION"""
    logger = setup_logging()
    
    logger.info("🔬 PHASE 2: HISTORICAL VALIDATION - REALITY CHECK")
    logger.info("=" * 60)
    logger.info("⚠️  WARNING: This will determine if we continue or pivot")
    logger.info("📊 Testing theoretical edge against real market data")
    logger.info("=" * 60)
    
    # Check if Phase 1 database exists
    if not SUPPORT_LEVELS_DB.exists():
        logger.error("❌ Phase 1 database not found! Run Phase 1 first.")
        return
    
    try:
        # Initialize validator
        validator = HistoricalValidator()
        
        # Run validation (sample 10 coins dulu)
        logger.info("🎯 Starting validation with 10 priority coins...")
        results = validator.run_validation(SUPPORT_LEVELS_DB, max_coins=10)
        
        # Save results dengan FIX
        with open(VALIDATION_RESULTS, 'w') as f:
            json.dump(results, f, indent=2, default=convert_to_serializable)
        
        # Display critical summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 VALIDATION RESULTS - CRITICAL ASSESSMENT")
        logger.info("=" * 60)
        
        metrics = results.get('overall_metrics', {})
        logger.info(f"🎯 Coins Tested: {metrics.get('total_coins_tested', 0)}")
        logger.info(f"📈 Valid Levels Found: {metrics.get('valid_levels_found', 0)}/{metrics.get('total_levels_tested', 0)}")
        logger.info(f"📊 Average Bounce Rate: {metrics.get('average_bounce_rate', 0):.3f}")
        logger.info(f"🔍 Confidence Gap: {metrics.get('confidence_gap_avg', 0):.3f}")
        
        verdict = metrics.get('validation_verdict', 'UNKNOWN')
        logger.info(f"🚨 VERDICT: {verdict}")
        
        logger.info("\n🎯 NEXT ACTIONS BASED ON RESULTS:")
        if "❌" in verdict:
            logger.info("   • STOP - Fundamental edge tidak ada")
            logger.info("   • Pivot ke strategy lain")
            logger.info("   • Jangan buang waktu optimize")
        elif "⚠️" in verdict:
            logger.info("   • CAUTION - Partial edge detected")  
            logger.info("   • Test more coins & timeframes")
            logger.info("   • Optimize selection criteria")
        elif "✅" in verdict:
            logger.info("   • CONTINUE - Strong edge confirmed")
            logger.info("   • Scale to more coins")
            logger.info("   • Proceed to Phase 3 (Execution)")
        
        logger.info("=" * 60)
        logger.info(f"💾 Full results saved: {VALIDATION_RESULTS}")
        
    except Exception as e:
        logger.error(f"❌ Phase 2 validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()