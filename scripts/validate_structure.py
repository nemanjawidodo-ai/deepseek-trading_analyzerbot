# scripts/validate_structure.py
import os
from pathlib import Path

REQUIRED_STRUCTURE = {
    'config': ['config.yaml', 'strategies.yaml'],
    'data/collectors': ['binance.py'],
    'data/processors': ['database_builder.py'],
    'data/validators': ['historical.py', 'enhanced.py'],
    'strategies': ['base.py'],
    'risk': ['risk_manager.py', 'position_sizer.py', 'kill_switch.py'],
    'execution': ['order_manager.py', 'paper_trading.py'],
    'monitoring': ['dashboard.py', 'alerts.py'],
    'tests/unit': [],
    'tests/integration': []
}

def validate_structure():
    errors = []
    for folder, files in REQUIRED_STRUCTURE.items():
        if not Path(folder).exists():
            errors.append(f"❌ Missing folder: {folder}")
        for file in files:
            filepath = Path(folder) / file
            if not filepath.exists():
                errors.append(f"❌ Missing file: {filepath}")
    
    if errors:
        print("\n".join(errors))
        return False
    else:
        print("✅ All required files and folders present!")
        return True

if __name__ == "__main__":
    validate_structure()