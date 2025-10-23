import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

def load_and_process(csv_file):
    """Load CSV dengan manual column mapping yang tepat"""
    df = pd.read_csv(csv_file, encoding='utf-8')
    print(f"✅ CSV loaded! Shape: {df.shape}")
    
    print("\n📋 MANUAL COLUMN MAPPING:")
    column_map = {
        'date': 'Tanggal Kirim',
        'market': 'Market', 
        'buy': 'BUY',
        'sell': 'SELL',  # ← FIX: dari 'TP' ke 'SELL'
        'tp': 'TP',      # ← Tambah kolom TP
        'events': 'Events',
        'avg_days': 'Avg Days',  # ← FIX: dari 'Longest Days' ke 'Avg Days'
        'longest_days': 'Longest Days'  # ← Tambah kolom longest days
    }
    
    for key, value in column_map.items():
        print(f"  {key}: '{value}'")
    
    return df, column_map

def extract_support_levels(df, column_map):
    """Extract data dengan mapping yang benar"""
    print("\n🎯 EXTRACTING SUPPORT LEVELS...")
    
    support_levels = {}
    signal_count = 0
    error_count = 0
    
    for index, row in df.iterrows():
        try:
            # Get values dengan mapping manual
            market = str(row[column_map['market']]).strip()
            buy_val = row[column_map['buy']]
            
            # Skip jika BUY kosong
            if pd.isna(buy_val) or buy_val in ['', '-', ' ', None]:
                continue
            
            # Convert BUY price
            buy_price = float(str(buy_val).replace(',', '.'))
            
            # Get other values dengan error handling
            events = 0
            if pd.notna(row[column_map['events']]):
                try:
                    events = int(row[column_map['events']])
                except:
                    events = 0
            
            avg_days = 0
            if pd.notna(row[column_map['avg_days']]):
                try:
                    avg_days = int(row[column_map['avg_days']])
                except:
                    avg_days = 0
            
            # Get SELL/TP price
            tp_target = None
            if pd.notna(row[column_map['sell']]) and row[column_map['sell']] not in ['', '-']:
                try:
                    tp_target = float(str(row[column_map['sell']]).replace(',', '.'))
                except:
                    tp_target = None
            
            signal_date = ""
            if pd.notna(row[column_map['date']]):
                signal_date = str(row[column_map['date']])
            
            # Add to database
            if market not in support_levels:
                support_levels[market] = []
            
            support_levels[market].append({
                'support_price': buy_price,
                'events_count': events,
                'avg_recovery_days': avg_days,
                'tp_target': tp_target,
                'signal_date': signal_date,
                'original_index': index  # Untuk debugging
            })
            
            signal_count += 1
            
            # Show progress setiap 100 signals
            if signal_count % 100 == 0:
                print(f"  📈 Processed {signal_count} signals...")
                
        except Exception as e:
            error_count += 1
            continue
    
    print(f"✅ Success: {signal_count} signals")
    print(f"⚠️  Errors: {error_count} rows")
    print(f"📊 Total coins: {len(support_levels)}")
    
    return support_levels

def build_database(support_levels):
    """Build database dengan confidence scoring"""
    print("\n🏗️ BUILDING DATABASE...")
    
    database = {}
    
    for coin, levels in support_levels.items():
        database[coin] = []
        
        for level_data in levels:
            # Calculate confidence score
            events_score = min(level_data['events_count'] / 50, 1.0)
            recovery_score = 1.0 - (min(level_data['avg_recovery_days'] / 30, 1.0))
            confidence_score = (events_score * 0.7 + recovery_score * 0.3)
            
            db_entry = {
                'support_price': level_data['support_price'],
                'events_count': level_data['events_count'],
                'avg_recovery_days': level_data['avg_recovery_days'],
                'tp_target': level_data['tp_target'],
                'tp_percentage': round((level_data['tp_target'] / level_data['support_price'] - 1) * 100, 2) if level_data['tp_target'] else None,
                'confidence_score': round(confidence_score, 3),
                'signal_date': level_data['signal_date'],
                'quality': 'HIGH' if confidence_score > 0.7 else 'MEDIUM' if confidence_score > 0.5 else 'LOW',
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            database[coin].append(db_entry)
        
        # Sort by confidence
        database[coin].sort(key=lambda x: x['confidence_score'], reverse=True)
    
    return database

def save_database(database, filename='support_levels_fixed.json'):
    """Save database"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(database, f, indent=2, ensure_ascii=False)
    print(f"💾 Database saved to {filename}")

def display_summary(database):
    """Show detailed summary"""
    total_coins = len(database)
    total_levels = sum(len(levels) for levels in database.values())
    
    confidence_scores = []
    events_counts = []
    recovery_days = []
    
    for coin_levels in database.values():
        for level in coin_levels:
            confidence_scores.append(level['confidence_score'])
            events_counts.append(level['events_count'])
            recovery_days.append(level['avg_recovery_days'])
    
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
    avg_events = np.mean(events_counts) if events_counts else 0
    avg_recovery = np.mean(recovery_days) if recovery_days else 0
    
    print("\n" + "="*60)
    print("🎉 DATABASE BUILD SUCCESSFUL!")
    print("="*60)
    print(f"📊 Total Coins: {total_coins}")
    print(f"🎯 Support Levels: {total_levels}")
    print(f"⭐ Avg Confidence: {avg_confidence:.3f}")
    print(f"📈 Avg Events: {avg_events:.1f}")
    print(f"⏱️  Avg Recovery: {avg_recovery:.1f} days")
    print(f"🏆 Highest Confidence: {max(confidence_scores):.3f}" if confidence_scores else "🏆 No data")
    print(f"📈 Quality: {'EXCELLENT' if avg_confidence > 0.7 else 'GOOD' if avg_confidence > 0.5 else 'NEEDS WORK'}")
    print("="*60)
    
    # Show detailed coin analysis
    if database:
        print("\n🏅 TOP 10 COINS ANALYSIS:")
        coin_stats = []
        for coin, levels in database.items():
            if levels:
                avg_conf = np.mean([level['confidence_score'] for level in levels])
                total_events = sum([level['events_count'] for level in levels])
                coin_stats.append((coin, avg_conf, len(levels), total_events))
        
        # Sort by confidence
        for coin, score, levels_count, total_events in sorted(coin_stats, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {coin:12} | Score: {score:.3f} | Levels: {levels_count:2d} | Total Events: {total_events:4d}")

# MAIN EXECUTION
if __name__ == "__main__":
    print("🚀 PHASE 1: MANUAL FIX DATABASE BUILDER")
    print("=" * 60)
    
    csv_file = "recap sinyal trading - Sheet4.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ File {csv_file} not found!")
        exit()
    
    print(f"✅ Processing: {csv_file}")
    print("-" * 60)
    
    # Process dengan manual mapping
    df, column_map = load_and_process(csv_file)
    support_levels = extract_support_levels(df, column_map)
    
    if not support_levels:
        print("❌ Still no signals found! Debug info:")
        print("\n🔍 SAMPLE DATA CHECK:")
        print(df[['Market', 'BUY', 'SELL', 'Events', 'Avg Days']].head(10))
        exit()
    
    database = build_database(support_levels)
    save_database(database)
    display_summary(database)
    
    print(f"\n✅ PHASE 1 COMPLETED SUCCESSFULLY!")
    print("📁 Output: support_levels_fixed.json")
    print("🎯 Ready for Phase 2: Historical Analysis!")