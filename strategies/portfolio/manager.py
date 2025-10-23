# build_portfolio.py
def main():
    portfolio = PortfolioBuilder()
    
    # Load dari validation results yang ada
    portfolio.load_validation_results()
    
    # Rank coins berdasarkan performance
    ranked_coins = portfolio.rank_coins_by_performance()
    
    # Build risk-adjusted portfolio
    final_portfolio = portfolio.build_risk_adjusted_portfolio(top_n=20)
    
    # Calculate position sizes
    position_sizer = PositionSizer()
    for tier, coins in final_portfolio.items():
        for coin in coins:
            coin['position_size'] = position_sizer.calculate_position_size(coin)
    
    return final_portfolio
