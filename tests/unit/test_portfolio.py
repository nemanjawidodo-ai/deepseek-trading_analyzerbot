# test_portfolio.py - Simpan di root project
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.portfolio.portfolio import PortfolioBuilder

def main():
    print("🚀 TESTING PORTFOLIO BUILDER")
    print("=" * 50)
    
    portfolio = PortfolioBuilder()
    
    # Test load validation data
    print("📊 Loading validation results...")
    success = portfolio.load_validation_results()
    
    if success:
        print("✅ Validation data loaded successfully")
        
        # Test coin ranking
        print("🎯 Ranking coins by performance...")
        ranked_coins = portfolio.rank_coins_by_performance()
        print(f"✅ Ranked {len(ranked_coins)} coins")
        
        # Show top 10
        print("\n🏆 TOP 10 COINS:")
        for i, coin in enumerate(ranked_coins[:10]):
            print(f"   {i+1}. {coin['symbol']} - {coin.get('success_rate', 0):.1%}")
            
    else:
        print("❌ Failed to load validation data")
        print("💡 Check if data/validation_results/ has JSON files")

if __name__ == "__main__":
    main()