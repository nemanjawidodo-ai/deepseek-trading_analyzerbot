#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.portfolio.portfolio import PortfolioBuilder
from extended_validation.phase3_portfolio.position_sizer import PositionSizer

def main():
    print("🚀 COMPREHENSIVE PORTFOLIO BUILDER TEST")
    print("=" * 60)
    
    # Initialize
    portfolio_builder = PortfolioBuilder()
    position_sizer = PositionSizer()
    
    # Step 1: Load validation data
    print("\n1️⃣ LOADING VALIDATION DATA")
    print("-" * 30)
    success = portfolio_builder.load_validation_results()
    
    if not success:
        print("❌ Cannot proceed without validation data")
        return
    
    # Step 2: Rank coins
    print("\n2️⃣ RANKING COINS BY PERFORMANCE")
    print("-" * 30)
    ranked_coins = portfolio_builder.rank_coins_by_performance()
    print(f"✅ Ranked {len(ranked_coins)} coins")
    
    # Show top performers
    print("\n🏆 TOP 15 PERFORMERS:")
    for i, coin in enumerate(ranked_coins[:15]):
        print(f"   {i+1:2d}. {coin['symbol']:12} {coin['success_rate']:6.1%} "
              f"({coin['tier']})")
    
    # Step 3: Build portfolio
    print("\n3️⃣ BUILDING RISK-ADJUSTED PORTFOLIO")
    print("-" * 30)
    portfolio = portfolio_builder.build_risk_adjusted_portfolio(top_n=20)
    
    # Step 4: Calculate position sizes
    print("\n4️⃣ CALCULATING POSITION SIZES")
    print("-" * 30)
    portfolio_value = 5000  # $5,000 test capital
    
    print(f"💰 Portfolio Value: ${portfolio_value:,}")
    print("\n📊 POSITION ALLOCATIONS:")
    
    for tier in ['high_confidence', 'medium_confidence', 'low_confidence']:
        coins = portfolio[tier]
        if coins:
            print(f"\n🎯 {tier.upper().replace('_', ' ')}:")
            for coin in coins:
                size = position_sizer.calculate_position_size(coin, portfolio_value)
                pct = (size / portfolio_value) * 100
                print(f"   {coin['symbol']:12} ${size:6.2f} ({pct:4.1f}%)")
    
    # Step 5: Risk validation
    print("\n5️⃣ RISK VALIDATION")
    print("-" * 30)
    position_sizer.validate_portfolio_risk(portfolio, portfolio_value)
    
    # Step 6: Save portfolio
    print("\n6️⃣ SAVING PORTFOLIO")
    print("-" * 30)
    portfolio_builder.save_portfolio(portfolio)
    
    print("\n🎉 PORTFOLIO CONSTRUCTION COMPLETE!")

if __name__ == "__main__":
    main()