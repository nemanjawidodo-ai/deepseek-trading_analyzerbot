import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

def setup_logging(log_file: Path, level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_csv_data(file_path: Path) -> pd.DataFrame:
    """Load CSV data dengan better error handling"""
    logger = logging.getLogger(__name__)
    
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'windows-1252', 'cp1252']
    
    for encoding in encodings:
        try:
            logger.info(f"🔄 Trying encoding: {encoding}")
            df = pd.read_csv(file_path, encoding=encoding)
            logger.info(f"✅ CSV loaded with {encoding} encoding. Shape: {df.shape}")
            
            # Show sample data
            logger.info("📋 First 3 rows preview:")
            logger.info(df[['Market', 'BUY', 'SELL', 'Events']].head(3).to_string())
            
            return df
        except UnicodeDecodeError as e:
            continue
        except Exception as e:
            logger.warning(f"⚠️ Encoding {encoding} failed: {e}")
            continue
    
    # Last attempt dengan error handling
    try:
        logger.info("🔄 Last attempt: reading with error handling...")
        df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
        logger.info(f"✅ CSV loaded with error handling. Shape: {df.shape}")
        return df
    except Exception as e:
        raise ValueError(f"❌ All loading attempts failed: {e}")

def calculate_confidence_score(events_count: int, avg_recovery_days: int, weights: Dict) -> float:
    """Calculate confidence score 0-1"""
    events_score = min(events_count / 50, 1.0)
    recovery_score = 1.0 - (min(avg_recovery_days / 30, 1.0))
    
    confidence_score = (
        events_score * weights['events'] + 
        recovery_score * weights['recovery_days']
    )
    
    return min(confidence_score, 1.0)

def get_quality_label(confidence_score: float, thresholds: Dict) -> str:
    """Get quality label based on confidence score"""
    if confidence_score >= thresholds['HIGH']:
        return 'HIGH'
    elif confidence_score >= thresholds['MEDIUM']:
        return 'MEDIUM'
    else:
        return 'LOW'

def save_json_data(data: Dict, file_path: Path):
    """Save data to JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info(f"💾 Data saved to {file_path}")

def format_currency(value: float) -> str:
    """Format currency values"""
    if value >= 1:
        return f"${value:,.2f}"
    else:
        return f"${value:.4f}"