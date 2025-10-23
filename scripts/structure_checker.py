# deep_structure_checker.py
import os
from pathlib import Path

def scan_deep_structure(start_path='.'):
    """Scan semua file dan folder secara recursive dengan pengecualian"""
    
    print("=" * 70)
    print("🔍 DEEP STRUCTURE SCAN - FULL RECURSIVE")
    print("=" * 70)
    
    start_path = Path(start_path)
    print(f"📂 Starting scan from: {start_path.absolute()}")
    print()
    
    # Folder yang akan di-skip karena isinya terlalu banyak
    EXCLUDED_DIRS = {
        'venv', '.venv', 'env', '.env',  # Virtual environments
        '__pycache__', '.pytest_cache',  # Python cache
        'node_modules',  # Node.js modules
        '.git',  # Git repository
        '.idea', '.vscode',  # IDE folders
        'build', 'dist',  # Build folders
        '.mypy_cache', '.ruff_cache'  # Linter caches
    }
    
    all_files = []
    
    def scan_recursive(current_path, level=0):
        """Recursive scanner dengan pengecualian"""
        try:
            items = list(current_path.iterdir())
            items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
            
            for item in items:
                # Skip hidden files dan folder yang dikecualikan
                if item.name.startswith('.') or item.name in EXCLUDED_DIRS:
                    continue
                    
                indent = "    " * level
                
                if item.is_dir():
                    print(f"{indent}📁 {item.name}/")
                    scan_recursive(item, level + 1)
                else:
                    file_info = f"{indent}📄 {item.name}"
                    print(file_info)
                    all_files.append(str(item))
                        
        except PermissionError:
            print(f"{'    ' * level}❌ Permission denied: {current_path.name}")
        except Exception as e:
            print(f"{'    ' * level}❌ Error scanning {current_path.name}: {e}")
    
    # Start scanning
    scan_recursive(start_path)
    
    return all_files

def find_project_files(all_files):
    """Cari file-file project kita"""
    print()
    print("=" * 70)
    print("🎯 PROJECT FILE SEARCH")
    print("=" * 70)
    
    project_keywords = [
        'binance_client', 'historical_validator', 'expanded_validator',
        'validation_orchestrator', 'run_validation'
    ]
    
    found_files = []
    
    for file_path in all_files:
        filename = Path(file_path).name.lower()
        for keyword in project_keywords:
            if keyword in filename and file_path.endswith('.py'):
                found_files.append(file_path)
                print(f"✅ FOUND: {file_path}")
                break
    
    if not found_files:
        print("❌ No project files found with known names")
    
    return found_files

def check_specific_locations():
    """Check lokasi spesifik yang mungkin"""
    print()
    print("=" * 70)
    print("📍 CHECKING COMMON LOCATIONS")
    print("=" * 70)
    
    common_locations = [
        'trading_analyzerbot',
        'deepseek trading_analyzerbot',
        'research/trading_analyzerbot',
        'research/deepseek trading_analyzerbot',
        'Crypto/trading_analyzerbot',
        'Crypto/deepseek trading_analyzerbot'
    ]
    
    for location in common_locations:
        if Path(location).exists():
            print(f"📍 Found: {location}/")
            # Scan isi folder ini
            items = list(Path(location).iterdir())
            if items:
                print(f"   Contents: {[item.name for item in items[:5]]}")
                if len(items) > 5:
                    print(f"   ... and {len(items) - 5} more items")
        else:
            print(f"   Not found: {location}/")

def show_excluded_folders():
    """Tampilkan folder yang dikecualikan"""
    print()
    print("=" * 70)
    print("🚫 EXCLUDED FOLDERS")
    print("=" * 70)
    
    excluded_dirs = [
        'venv, .venv, env, .env - Virtual environments',
        '__pycache__, .pytest_cache - Python cache folders',
        'node_modules - Node.js modules (bisa sangat besar)',
        '.git - Git repository',
        '.idea, .vscode - IDE configuration folders',
        'build, dist - Build folders',
        '.mypy_cache, .ruff_cache - Linter cache folders'
    ]
    
    for excluded in excluded_dirs:
        print(f"   ⚠️  {excluded}")

if __name__ == "__main__":
    # Tampilkan folder yang dikecualikan
    show_excluded_folders()
    
    # Scan dari current directory
    all_files = scan_deep_structure('')
    
    # Cari file project
    project_files = find_project_files(all_files)
    
    # Check lokasi umum
    check_specific_locations()
    
    print()
    print("=" * 70)
    print("📊 SCAN SUMMARY")
    print("=" * 70)
    
    if project_files:
        print(f"✅ Found {len(project_files)} project files")
        print("🚀 Project files are located somewhere in the scanned structure")
        print("💡 Look for the file paths above to find the project location")
    else:
        print("❌ No project files found in current location")
        print("💡 Try running this script from different directories")
    
    print("=" * 70)