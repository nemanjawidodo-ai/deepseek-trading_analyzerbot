#!/usr/bin/env python3
"""
SETUP SCRIPT FIXED VERSION
"""

import shutil
from pathlib import Path

def setup_project():
    """Setup project structure - FIXED CSV COPY"""
    print("🚀 SETUP TRADING ANALYZER PROJECT - FIXED")
    print("=" * 50)
    
    current_dir = Path(__file__).parent
    
    # Define directory structure
    directories = [
        "config",
        "src", 
        "data/raw",
        "data/processed",
        "data/historical",
        "logs",
        "outputs/reports"
    ]
    
    # Create all directories
    print("📁 Creating directory structure...")
    for dir_path in directories:
        full_path = current_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {dir_path}/")
    
    # Check for CSV files in root - FIXED LOGIC
    print("\n📊 Looking for CSV files in project root...")
    csv_files = list(current_dir.glob("*.csv"))
    
    if csv_files:
        print(f"✅ Found {len(csv_files)} CSV file(s) in root:")
        target_dir = current_dir / "data" / "raw"
        
        for csv_file in csv_files:
            print(f"  📄 {csv_file.name}")
            
            # Copy to data/raw - ALWAYS COPY
            target_path = target_dir / csv_file.name
            shutil.copy2(csv_file, target_path)
            print(f"    ✅ COPIED to: data/raw/{csv_file.name}")
            
            # Verify copy worked
            if target_path.exists():
                print(f"    ✅ VERIFIED: Copy successful")
            else:
                print(f"    ❌ COPY FAILED!")
                
    else:
        print("❌ No CSV files found in project root!")
        print("💡 Please add your CSV file to this folder and run setup again")
        return False
    
    print("\n🎯 SETUP COMPLETED SUCCESSFULLY!")
    print("📁 Project structure ready")
    print("📊 CSV files copied to data/raw/")
    print("\n🚀 NEXT: Run 'python main.py'")
    
    return True

if __name__ == "__main__":
    success = setup_project()
    if not success:
        print("\n❌ Setup failed - please check above errors")