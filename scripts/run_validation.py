#!/usr/bin/env python3
"""
SIMPLE CLI FOR VALIDATION - Unified validation runner
"""
import sys
import os
import pandas as pd

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from strategies.validation.orchestrator import ValidationOrchestrator, ValidationMode

def main():
    # Default mode
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "quick"  # Default to quick test
    
    # Map arguments to validation modes
    mode_map = {
        "quick": ValidationMode.QUICK_TEST,
        "standard": ValidationMode.STANDARD, 
        "comprehensive": ValidationMode.COMPREHENSIVE,
        "fast": ValidationMode.QUICK_TEST,
        "normal": ValidationMode.STANDARD,
        "full": ValidationMode.COMPREHENSIVE
    }
    
    if mode not in mode_map:
        print(f"❌ Unknown mode: {mode}")
        print("Available modes: quick, standard, comprehensive")
        print("Usage: python run_validation.py [quick|standard|comprehensive]")
        print("Examples:")
        print("  python run_validation.py quick        # Fast test (10 coins)")
        print("  python run_validation.py standard     # Normal test (25 coins)") 
        print("  python run_validation.py comprehensive # Full test (50 coins)")
        return
    
    print(f"🚀 Starting {mode} validation...")
    
    try:
        orchestrator = ValidationOrchestrator()
        results = orchestrator.run_validation(mode_map[mode])
        
        print(f"\n🎯 VALIDATION COMPLETED: {results.get('recommendation', 'UNKNOWN')}")
        print(f"📊 Success Rate: {results.get('overall_success_rate', 0):.1%}")
        print(f"✅ Validation Passed: {results.get('validation_passed', False)}")
        
    except Exception as e:
        print(f"❌ Error running validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()