import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import logging
import sys
import os
from pathlib import Path
from config.config_loader import load_strategies


# 🔥 CRITICAL FIX: Setup Python path sebelum import apapun
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # Naik ke root project
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# NEW: Import dari config loader
try:
    from config.config_loader import load_paths
    from utils.helpers import (
        calculate_confidence_score, 
        get_quality_label,
        load_csv_data, 
        save_json_data,
        setup_logging
    )
    print("✅ All imports successful in database_builder.py")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    # Fallback functions jika import gagal
    def calculate_confidence_score(events_count, recovery_days, weights):
        events_score = min(events_count / 50, 1.0)
        recovery_score = 1.0 - min(recovery_days / 30, 1.0)
        return events_score * weights.get('events', 0.6) + recovery_score * weights.get('recovery_speed', 0.4)
    
    def get_quality_label(score, thresholds):
        if score >= thresholds.get('high', 0.7): return 'HIGH'
        elif score >= thresholds.get('medium', 0.4): return 'MEDIUM'
        return 'LOW'
    
    def load_csv_data(file_path): return pd.DataFrame()
    def save_json_data(data, file_path): pass
    def setup_logging(log_file): return logging.getLogger(__name__)

class DatabaseBuilder:
    """Main class untuk build support levels database"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load paths config
        paths_config = load_paths()
        
        # Extract column mapping
        self.column_map = paths_config['csv_columns']
        
        # Extract scoring config
        scoring = paths_config['scoring']
        self.weights = scoring['weights']
        self.thresholds = scoring['quality_thresholds']
        
        # Get output paths
        self.output_path = Path(paths_config['output_files']['database'])
        self.log_path = Path(paths_config['output_files']['execution_log'])
        
    def extract_support_levels(self, df: pd.DataFrame) -> Dict[str, List]:
        """Extract support levels dari DataFrame"""
        self.logger.info("🎯 Extracting support levels from signals...")
        
        support_levels = {}
        stats = {'processed': 0, 'errors': 0, 'skipped': 0}
        
        for index, row in df.iterrows():
            try:
                market = str(row[self.column_map['market']]).strip()
                buy_val = row[self.column_map['buy']]
                
                # Skip empty BUY values
                if pd.isna(buy_val) or buy_val in ['', '-', ' ', None]:
                    stats['skipped'] += 1
                    continue
                
                # Process the signal
                level_data = self._process_signal_row(row, market)
                if level_data:
                    if market not in support_levels:
                        support_levels[market] = []
                    
                    support_levels[market].append(level_data)
                    stats['processed'] += 1
                    
                    # Progress logging
                    if stats['processed'] % 100 == 0:
                        self.logger.info(f"  📈 Processed {stats['processed']} signals...")
                        
            except Exception as e:
                stats['errors'] += 1
                continue
        
        self.logger.info(f"✅ Extraction complete: {stats['processed']} signals, "
                        f"{stats['errors']} errors, {stats['skipped']} skipped")
        self.logger.info(f"📊 Total coins: {len(support_levels)}")
        
        return support_levels
    
    def _process_signal_row(self, row: pd.Series, market: str) -> Dict[str, Any]:
        """Process individual signal row"""
        try:
            # Extract and clean data
            buy_price = float(str(row[self.column_map['buy']]).replace(',', '.'))
            
            events = self._safe_int_extract(row, self.column_map['events'])
            avg_days = self._safe_int_extract(row, self.column_map['avg_days'])
            
            tp_target = self._safe_float_extract(row, self.column_map['sell'])
            signal_date = str(row[self.column_map['date']]) if pd.notna(row[self.column_map['date']]) else ""
            
            # Calculate TP percentage
            tp_percentage = None
            if tp_target:
                tp_percentage = round((tp_target / buy_price - 1) * 100, 2)
            
            return {
                'support_price': buy_price,
                'events_count': events,
                'avg_recovery_days': avg_days,
                'tp_target': tp_target,
                'tp_percentage': tp_percentage,
                'signal_date': signal_date,
                'original_index': int(row.name)  # Untuk tracking
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error processing row {row.name}: {e}")
            return None
    
    def _safe_int_extract(self, row: pd.Series, column: str) -> int:
        """Safely extract integer values"""
        try:
            if pd.notna(row[column]):
                return int(row[column])
        except:
            pass
        return 0
    
    def _safe_float_extract(self, row: pd.Series, column: str) -> float:
        """Safely extract float values"""
        try:
            if pd.notna(row[column]) and row[column] not in ['', '-']:
                return float(str(row[column]).replace(',', '.'))
        except:
            pass
        return None
    
    def build_database(self, support_levels: Dict) -> Dict[str, List]:
        """Build complete database dengan confidence scoring"""
        self.logger.info("🏗️ Building database dengan confidence scoring...")
        
        database = {}
        
        for coin, levels in support_levels.items():
            database[coin] = []
            
            for level_data in levels:
                # Calculate confidence metrics
                confidence_score = calculate_confidence_score(
                    level_data['events_count'], 
                    level_data['avg_recovery_days'],
                    self.weights
                )
                
                quality = get_quality_label(confidence_score, self.thresholds)
                
                # Build database entry
                db_entry = {
                    **level_data,
                    'confidence_score': round(confidence_score, 3),
                    'quality': quality,
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'analysis_metadata': {
                        'events_score': round(min(level_data['events_count'] / 50, 1.0), 3),
                        'recovery_score': round(1.0 - min(level_data['avg_recovery_days'] / 30, 1.0), 3),
                        'calculated_at': datetime.now().isoformat()
                    }
                }
                
                database[coin].append(db_entry)
            
            # Sort by confidence score
            database[coin].sort(key=lambda x: x['confidence_score'], reverse=True)
        
        return database
    
    def generate_summary_report(self, database: Dict) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        total_coins = len(database)
        total_levels = sum(len(levels) for levels in database.values())
        
        # Collect statistics
        confidence_scores = []
        events_counts = []
        recovery_days = []
        tp_percentages = []
        
        quality_distribution = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        
        for coin_levels in database.values():
            for level in coin_levels:
                confidence_scores.append(level['confidence_score'])
                events_counts.append(level['events_count'])
                recovery_days.append(level['avg_recovery_days'])
                quality_distribution[level['quality']] += 1
                
                if level['tp_percentage']:
                    tp_percentages.append(level['tp_percentage'])
        
        # Calculate metrics
        metrics = {
            'total_coins': total_coins,
            'total_levels': total_levels,
            'avg_confidence': round(np.mean(confidence_scores), 3) if confidence_scores else 0,
            'avg_events': round(np.mean(events_counts), 1) if events_counts else 0,
            'avg_recovery_days': round(np.mean(recovery_days), 1) if recovery_days else 0,
            'avg_tp_percentage': round(np.mean(tp_percentages), 2) if tp_percentages else 0,
            'max_confidence': round(max(confidence_scores), 3) if confidence_scores else 0,
            'min_confidence': round(min(confidence_scores), 3) if confidence_scores else 0,
            'quality_distribution': quality_distribution,
            'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return metrics
    
    def display_detailed_summary(self, database: Dict, metrics: Dict):
        """Display beautiful detailed summary"""
        self.logger.info("\n" + "="*70)
        self.logger.info("🎉 DATABASE BUILD COMPLETE - DETAILED ANALYSIS")
        self.logger.info("="*70)
        
        self.logger.info(f"📊 TOTAL STATISTICS:")
        self.logger.info(f"  • Coins Analyzed: {metrics['total_coins']}")
        self.logger.info(f"  • Support Levels: {metrics['total_levels']}")
        self.logger.info(f"  • Average TP Target: {metrics['avg_tp_percentage']}%")
        
        self.logger.info(f"\n📈 CONFIDENCE METRICS:")
        self.logger.info(f"  • Average Score: {metrics['avg_confidence']:.3f}")
        self.logger.info(f"  • Range: {metrics['min_confidence']:.3f} - {metrics['max_confidence']:.3f}")
        self.logger.info(f"  • Quality: {'EXCELLENT' if metrics['avg_confidence'] > 0.7 else 'GOOD' if metrics['avg_confidence'] > 0.5 else 'NEEDS IMPROVEMENT'}")
        
        self.logger.info(f"\n🎯 QUALITY DISTRIBUTION:")
        for quality, count in metrics['quality_distribution'].items():
            percentage = (count / metrics['total_levels']) * 100
            self.logger.info(f"  • {quality}: {count} levels ({percentage:.1f}%)")
        
        self.logger.info(f"\n🏅 TOP 10 COINS BY CONFIDENCE:")
        coin_stats = []
        for coin, levels in database.items():
            if levels:
                avg_conf = np.mean([level['confidence_score'] for level in levels])
                total_events = sum([level['events_count'] for level in levels])
                avg_tp = np.mean([level['tp_percentage'] for level in levels if level['tp_percentage']])
                coin_stats.append((coin, avg_conf, len(levels), total_events, avg_tp))
        
        # Sort dan display top 10
        for coin, score, levels_count, total_events, avg_tp in sorted(coin_stats, key=lambda x: x[1], reverse=True)[:10]:
            self.logger.info(f"  {coin:12} | Score: {score:.3f} | Levels: {levels_count:2d} | "
                           f"Events: {total_events:4d} | Avg TP: {avg_tp:.1f}%")
        
        self.logger.info("="*70)

def main():
    """Main execution function"""
    from utils.helpers import setup_logging
    
    # Setup logging
    logger = setup_logging(settings.EXECUTION_LOG)
    
    try:
        logger.info("🚀 PHASE 1: STRUCTURED DATABASE BUILDER STARTED")
        
        # Initialize builder
        builder = DatabaseBuilder()
        
        # Load data
        df = load_csv_data(settings.CSV_SOURCE)
        
        # Extract support levels
        support_levels = builder.extract_support_levels(df)
        
        if not support_levels:
            logger.error("❌ No valid support levels found!")
            return
        
        # Build database
        database = builder.build_database(support_levels)
        
        # Generate report
        metrics = builder.generate_summary_report(database)
        
        # Save database
        save_json_data(database, settings.DATABASE_OUTPUT)
        
        # Display results
        builder.display_detailed_summary(database, metrics)
        
        logger.info("✅ PHASE 1 COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        logger.error(f"❌ Phase 1 failed: {e}")
        raise

if __name__ == "__main__":
    main()