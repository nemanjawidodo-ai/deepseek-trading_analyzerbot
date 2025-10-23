#!/bin/bash
# scripts/migrate_structure.sh

echo "🚀 Starting Project Restructuring..."

# Create new structure
mkdir -p config data/{collectors,processors,storage,validators}
mkdir -p strategies/{indicators,signals,portfolio}
mkdir -p execution risk backtesting/{validators,stress_tests}
mkdir -p monitoring tests/{unit,integration,backtests}
mkdir -p scripts docs/runbooks

# Move config files
echo "📁 Moving config files..."
cp config/trading_strategy.py config/strategies.yaml.template
cp config/phase2_settings.py config/config.yaml.template

# Move data pipeline
echo "📊 Moving data pipeline..."
cp src/binance_client.py data/collectors/binance.py
cp src/database_builder.py data/processors/database_builder.py
cp src/historical_validator.py data/validators/historical.py

# Move risk management
echo "⚖️ Moving risk management..."
cp src/position_sizer.py risk/position_sizer.py
cp src/risk_manager.py risk/risk_manager.py
cp src/kill_switch_manager.py risk/kill_switch.py

# Move execution
echo "🚀 Moving execution layer..."
cp deploy_paper_trading.py execution/paper_trading.py
cp deployment_manager.py execution/deployment_manager.py

# Move monitoring
echo "📈 Moving monitoring..."
cp monitoring_dashboard.py monitoring/dashboard.py
cp advanced_monitoring.py monitoring/alerts.py

# Create __init__.py files
echo "📝 Creating __init__.py files..."
touch data/__init__.py
touch strategies/__init__.py
touch execution/__init__.py
touch risk/__init__.py
touch monitoring/__init__.py

# Generate requirements.txt
echo "📦 Generating requirements.txt..."
pip freeze > requirements.txt

# Create .env.example
echo "🔐 Creating .env.example..."
cat > .env.example << EOF
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
REDIS_HOST=localhost
REDIS_PORT=6379
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id
EOF

echo "✅ Restructuring completed!"
echo "🧪 Run 'pytest tests/' to validate"