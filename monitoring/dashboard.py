#!/usr/bin/env python3
"""
REAL-TIME MONITORING DASHBOARD
Live performance tracking vs backtest expectations
"""

import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template_string, jsonify

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LivePerformanceMonitor:
    """Real-time performance monitoring vs backtest expectations"""
    
    def __init__(self, validation_path='validation_results.json'):
        self.validation_path = validation_path
        self.expected_metrics = self._load_expected_metrics()
        self.live_metrics = {}
        self.alert_history = []
        
    def _load_expected_metrics(self):
        """Load expected metrics dari validation results"""
        try:
            with open(self.validation_path, 'r') as f:
                validation_data = json.load(f)
            
            basic_validation = validation_data.get('basic_validation', {})
            validation_results = basic_validation.get('validation_results', {})
            
            return {
                'sharpe_ratio': validation_results.get('monte_carlo', {}).get('sharpe_ratio', {}).get('mean', 2.0),
                'win_rate': 0.55,
                'max_drawdown': -0.15,
                'daily_volatility': 0.02,
                'consistency_score': 0.75
            }
        except:
            return {
                'sharpe_ratio': 2.0,
                'win_rate': 0.55,
                'max_drawdown': -0.15,
                'daily_volatility': 0.02,
                'consistency_score': 0.75
            }
    
    def start_live_monitoring(self):
        """Start real-time monitoring"""
        logger.info("📊 Starting live performance monitoring...")
        
        # Start background monitoring thread
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
        
        # Start web dashboard
        self._start_web_dashboard()
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                # Update live metrics
                self._update_live_metrics()
                
                # Check for alerts
                self._check_performance_alerts()
                
                # Generate performance report
                self._generate_performance_report()
                
                time.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(30)
    
    def _update_live_metrics(self):
        """Update live performance metrics"""
        try:
            # Simulate live metrics (dalam real implementation, ini akan connect ke exchange/execution engine)
            performance_data = self._simulate_live_performance()
            
            self.live_metrics = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': performance_data['portfolio_value'],
                'daily_pnl': performance_data['daily_pnl'],
                'drawdown': performance_data['drawdown'],
                'sharpe_ratio': performance_data['sharpe_ratio'],
                'win_rate': performance_data['win_rate'],
                'total_trades': performance_data['total_trades'],
                'active_positions': performance_data['active_positions']
            }
            
            # Save to performance tracking
            self._save_performance_snapshot()
            
        except Exception as e:
            logger.error(f"Failed to update live metrics: {e}")
    
    def _simulate_live_performance(self):
        """Simulate live performance data (placeholder untuk real implementation)"""
        # Dalam real implementation, ini akan mengambil data real dari trading engine
        base_value = 10000
        daily_return = np.random.normal(0.001, 0.01)  # 0.1% mean return, 1% std
        
        return {
            'portfolio_value': base_value * (1 + np.random.normal(0.0005, 0.005)),
            'daily_pnl': base_value * daily_return,
            'drawdown': max(-0.02, np.random.normal(-0.005, 0.01)),
            'sharpe_ratio': max(0, np.random.normal(2.0, 0.5)),
            'win_rate': max(0.4, min(0.8, np.random.normal(0.6, 0.1))),
            'total_trades': np.random.randint(5, 20),
            'active_positions': np.random.randint(1, 5)
        }
    
    def _save_performance_snapshot(self):
        """Save performance snapshot ke CSV"""
        try:
            snapshot = self.live_metrics.copy()
            df = pd.DataFrame([snapshot])
            
            file_path = 'deployment/performance_tracking.csv'
            if Path(file_path).exists():
                existing_df = pd.read_csv(file_path)
                updated_df = pd.concat([existing_df, df], ignore_index=True)
                updated_df.to_csv(file_path, index=False)
            else:
                df.to_csv(file_path, index=False)
                
        except Exception as e:
            logger.error(f"Failed to save performance snapshot: {e}")
    
    def _check_performance_alerts(self):
        """Check for performance alerts vs expectations"""
        alerts = []
        
        # Sharpe ratio alert
        live_sharpe = self.live_metrics.get('sharpe_ratio', 0)
        expected_sharpe = self.expected_metrics['sharpe_ratio']
        if live_sharpe < expected_sharpe * 0.7:  # 30% below expectation
            alerts.append(f"Sharpe ratio underperforming: {live_sharpe:.2f} vs expected {expected_sharpe:.2f}")
        
        # Drawdown alert
        live_drawdown = self.live_metrics.get('drawdown', 0)
        if live_drawdown < -0.10:  # Beyond 10% drawdown
            alerts.append(f"Significant drawdown: {live_drawdown:.2%}")
        
        # Win rate alert
        live_win_rate = self.live_metrics.get('win_rate', 0)
        expected_win_rate = self.expected_metrics['win_rate']
        if live_win_rate < expected_win_rate * 0.8:  # 20% below expectation
            alerts.append(f"Win rate underperforming: {live_win_rate:.1%} vs expected {expected_win_rate:.1%}")
        
        # Log alerts
        for alert in alerts:
            alert_record = {
                'timestamp': datetime.now().isoformat(),
                'alert': alert,
                'severity': 'WARNING'
            }
            self.alert_history.append(alert_record)
            logger.warning(f"🚨 ALERT: {alert}")
    
    def _generate_performance_report(self):
        """Generate performance report vs expectations"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'live_metrics': self.live_metrics,
            'expected_metrics': self.expected_metrics,
            'performance_gap': self._calculate_performance_gap(),
            'alerts_count': len(self.alert_history),
            'status': self._calculate_overall_status()
        }
        
        with open('deployment/live_performance_report.json', 'w') as f:
            json.dump(report, f, indent=2)
    
    def _calculate_performance_gap(self):
        """Calculate performance gap vs expectations"""
        gaps = {}
        
        for metric in ['sharpe_ratio', 'win_rate']:
            live = self.live_metrics.get(metric, 0)
            expected = self.expected_metrics.get(metric, 0)
            if expected != 0:
                gap_pct = (live - expected) / expected * 100
                gaps[metric] = gap_pct
        
        return gaps
    
    def _calculate_overall_status(self):
        """Calculate overall performance status"""
        gaps = self._calculate_performance_gap()
        
        # Jika Sharpe ratio within 20% of expectation, consider GREEN
        sharpe_gap = abs(gaps.get('sharpe_ratio', 100))
        if sharpe_gap <= 20:
            return 'GREEN'
        elif sharpe_gap <= 40:
            return 'YELLOW'
        else:
            return 'RED'
    
    def _start_web_dashboard(self):
        """Start web-based monitoring dashboard"""
        app = Flask(__name__)
        
        @app.route('/')
        def dashboard():
            return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Trading Bot Live Dashboard</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .metric { background: #f5f5f5; padding: 15px; margin: 10px; border-radius: 5px; }
                    .green { border-left: 5px solid #4CAF50; }
                    .yellow { border-left: 5px solid #FFC107; }
                    .red { border-left: 5px solid #F44336; }
                    .alert { background: #FFEBEE; padding: 10px; margin: 5px; border-radius: 3px; }
                </style>
            </head>
            <body>
                <h1>🤖 Trading Bot Live Dashboard</h1>
                <div id="metrics"></div>
                <div id="alerts"></div>
                <script>
                    function updateDashboard() {
                        fetch('/api/metrics')
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById('metrics').innerHTML = data.html;
                                document.getElementById('alerts').innerHTML = data.alerts_html;
                            });
                    }
                    setInterval(updateDashboard, 5000);
                    updateDashboard();
                </script>
            </body>
            </html>
            ''')
        
        @app.route('/api/metrics')
        def api_metrics():
            status_class = {
                'GREEN': 'green',
                'YELLOW': 'yellow', 
                'RED': 'red'
            }.get(self._calculate_overall_status(), 'yellow')
            
            html = f'''
            <div class="metric {status_class}">
                <h2>Live Performance Metrics</h2>
                <p><strong>Portfolio Value:</strong> ${self.live_metrics.get('portfolio_value', 0):.2f}</p>
                <p><strong>Daily P&L:</strong> ${self.live_metrics.get('daily_pnl', 0):.2f}</p>
                <p><strong>Drawdown:</strong> {self.live_metrics.get('drawdown', 0):.2%}</p>
                <p><strong>Sharpe Ratio:</strong> {self.live_metrics.get('sharpe_ratio', 0):.2f}</p>
                <p><strong>Win Rate:</strong> {self.live_metrics.get('win_rate', 0):.1%}</p>
                <p><strong>Status:</strong> {self._calculate_overall_status()}</p>
            </div>
            '''
            
            alerts_html = '<h2>Recent Alerts</h2>'
            for alert in self.alert_history[-5:]:  # Last 5 alerts
                alerts_html += f'<div class="alert">[{alert["timestamp"]}] {alert["alert"]}</div>'
            
            return jsonify({'html': html, 'alerts_html': alerts_html})
        
        # Run Flask app in background thread
        flask_thread = threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False),
            daemon=True
        )
        flask_thread.start()
        
        logger.info("🌐 Web dashboard started: http://localhost:8080")

def main():
    """Start monitoring dashboard"""
    logger.info("🎯 Starting Live Performance Monitoring Dashboard")
    
    try:
        monitor = LivePerformanceMonitor()
        monitor.start_live_monitoring()
        
        print("\n" + "="*80)
        print("📊 LIVE MONITORING DASHBOARD ACTIVE")
        print("="*80)
        print("🌐 Dashboard URL: http://localhost:8080")
        print("📈 Performance tracking: deployment/performance_tracking.csv")
        print("🚨 Alerts: Check logs and dashboard")
        print("⏰ Monitoring active - Press Ctrl+C to stop")
        print("="*80)
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")

if __name__ == "__main__":
    main()