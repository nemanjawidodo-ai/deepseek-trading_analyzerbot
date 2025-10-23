import pandas as pd
from pathlib import Path

def debug_csv():
    """Debug CSV loading issues"""
    csv_files = list(Path('.').glob("*.csv"))
    print(f"📁 Found {len(csv_files)} CSV files:")
    
    for csv_file in csv_files:
        print(f"\n🔍 Analyzing: {csv_file}")
        print(f"Size: {csv_file.stat().st_size} bytes")
        
        try:
            # Try to read first few lines
            with open(csv_file, 'r', encoding='utf-8') as f:
                lines = [f.readline() for _ in range(5)]
            
            print("✅ Can read file")
            print("First 2 lines:")
            for i, line in enumerate(lines[:2]):
                print(f"  {i+1}: {line.strip()}")
                
        except Exception as e:
            print(f"❌ Error reading: {e}")

if __name__ == "__main__":
    debug_csv()