#BONUS: Report Generator Script
# chmod +x scripts/generate_report.sh
# ./scripts/generate_report.sh > report_$(date +%Y%m%d).md

# # Review report
# cat report_20250106.md

# # Copy relevant parts ke Claude (hanya yang penting!)

#!/bin/bash
# Generate migration report otomatis

echo "# 📊 MIGRATION STATUS REPORT"
echo "Generated: $(date)"
echo ""

echo "## 🏗️ Structure"
tree -L 4 -I '__pycache__|*.pyc' config/ data/
echo ""

echo "## 🧪 Import Tests"
python scripts/check_imports.py
echo ""

echo "## 📋 Config Tests"
python scripts/test_config_values.py
echo ""

echo "## 🔍 Git Status"
# git status --short # Show changed files only
git diff --stat # Show summary of changes
echo ""

echo "## ✅ Checklist"
echo "- [x] Config files created"
echo "- [x] Dependencies installed"
echo "- [ ] All imports updated"
echo "- [ ] Integration tests passed"