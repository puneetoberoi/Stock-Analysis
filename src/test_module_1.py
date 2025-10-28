"""
🧪 MODULE 1 TEST - Standalone Test File
Tests watchlist generator without modifying main.py
Run: python test_module_1.py
"""

import asyncio
import logging
import sys
import os

# Import functions from main.py
from main import (
    generate_comprehensive_watchlist,
    calculate_all_key_levels,
    calculate_key_levels,
    fetch_macro_sentiment,
    analyze_portfolio_with_predictions,
    aiohttp
)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def test_module_1():
    """
    Test Module 1: Comprehensive watchlist + key levels
    """
    print("\n" + "="*70)
    print("🧪 TESTING MODULE 1: COMPREHENSIVE WATCHLIST GENERATOR")
    print("="*70)
    
    # =========================================================================
    # TEST 1: Key Levels Calculator (SPY & QQQ)
    # =========================================================================
    print("\n📊 TEST 1: Key Levels Calculator (Market Indices)")
    print("-" * 70)
    
    spy_levels = calculate_key_levels('SPY')
    qqq_levels = calculate_key_levels('QQQ')
    
    for ticker, data in [('SPY', spy_levels), ('QQQ', qqq_levels)]:
        if 'error' not in data:
            print(f"\n{ticker}:")
            print(f"  Current Price:  ${data['current']:.2f}")
            print(f"  Support Level:  ${data['support']:.2f} ({data['distance_to_support_pct']:.1f}% away)")
            print(f"  Resistance:     ${data['resistance']:.2f} ({data['distance_to_resistance_pct']:.1f}% away)")
            
            if data['at_support']:
                print(f"  ⚠️  ALERT: At support level!")
            elif data['at_resistance']:
                print(f"  ⚠️  ALERT: At resistance level!")
        else:
            print(f"\n{ticker}: ❌ ERROR - {data['error']}")
    
    # =========================================================================
    # TEST 2: Portfolio Watchlist (if portfolio.json exists)
    # =========================================================================
    print("\n\n📋 TEST 2: Comprehensive Watchlist Generator")
    print("-" * 70)
    
    if not os.path.exists('portfolio.json'):
        print("⚠️  No portfolio.json found")
        print("   Skipping full watchlist test (this is normal for first run)")
        print("   Run full analysis first to generate portfolio data")
        return
    
    # Run full portfolio analysis
    async with aiohttp.ClientSession() as session:
        try:
            print("\n🔄 Step 1: Fetching macro data...")
            macro_data = await fetch_macro_sentiment(session)
            print(f"✅ Macro score: {macro_data.get('overall_macro_score', 0):.1f}")
            
            print("\n🔄 Step 2: Analyzing portfolio with predictions...")
            portfolio_data = await analyze_portfolio_with_predictions(
                session, 
                market_context=macro_data
            )
            print(f"✅ Portfolio analyzed: {len(portfolio_data.get('stocks', []))} stocks")
            
            print("\n🔄 Step 3: Generating comprehensive watchlist...")
            watchlist = await generate_comprehensive_watchlist(
                portfolio_data, 
                None, 
                macro_data
            )
            
            print("\n🔄 Step 4: Calculating key levels for all stocks...")
            all_levels = await calculate_all_key_levels(portfolio_data)
            
            # =====================================================================
            # DISPLAY RESULTS
            # =====================================================================
            print("\n" + "="*70)
            print("📊 WATCHLIST SUMMARY")
            print("="*70)
            
            print(f"\n✅ Generated watchlist with:")
            print(f"   🔥 {len(watchlist['squeeze_breakouts'])} Bollinger squeeze breakouts")
            print(f"   📊 {len(watchlist['rsi_extremes'])} RSI extremes (>70 or <30)")
            print(f"   📈 {len(watchlist['volume_spikes'])} Volume alerts")
            print(f"   🎯 {len(watchlist['week52_alerts'])} 52-week level alerts")
            print(f"   ⬆️ {len(watchlist['gap_alerts'])} Gap alerts")
            print(f"   📅 {len(watchlist['earnings_this_week'])} Earnings this week")
            print(f"   🎨 {len(watchlist['pattern_confirmations'])} Pattern confirmations")
            print(f"   🌍 {len(watchlist['macro_alerts'])} Macro alerts")
            print(f"   🔑 {len(all_levels)} Key levels calculated")
            
            # =====================================================================
            # TOP 3 SQUEEZE BREAKOUTS
            # =====================================================================
            if watchlist['squeeze_breakouts']:
                print("\n" + "="*70)
                print("🔥 TOP SQUEEZE BREAKOUTS (Highest Probability)")
                print("="*70)
                
                for i, item in enumerate(watchlist['squeeze_breakouts'][:3], 1):
                    print(f"\n{i}. {item['ticker']} - {item['name']}")
                    print(f"   Squeeze Width:        {item['squeeze_width']:.1f}%")
                    print(f"   Breakout Probability: {item['probability']}%")
                    print(f"   Current Price:        ${item['current_price']:.2f}")
                    print(f"   Bullish Breakout:     ${item['bullish_break']:.2f}")
                    print(f"   Bearish Breakdown:    ${item['bearish_break']:.2f}")
                    print(f"   Volume Needed:        >{item['volume_needed']:.1f}x")
            
            # =====================================================================
            # RSI EXTREMES
            # =====================================================================
            if watchlist['rsi_extremes']:
                print("\n" + "="*70)
                print("📊 RSI EXTREMES")
                print("="*70)
                
                for item in watchlist['rsi_extremes']:
                    emoji = "🔴" if item['type'] == 'OVERBOUGHT' else "🟢"
                    print(f"{emoji} {item['ticker']}: RSI {item['rsi']:.1f} ({item['type']})")
            
            # =====================================================================
            # VOLUME ALERTS
            # =====================================================================
            if watchlist['volume_spikes']:
                print("\n" + "="*70)
                print("📈 VOLUME ALERTS")
                print("="*70)
                
                for item in watchlist['volume_spikes'][:5]:
                    emoji = "🔥" if item['type'] == 'HIGH' else "⚠️"
                    print(f"{emoji} {item['ticker']}: {item['volume_ratio']:.1f}x ({item['type']})")
            
            # =====================================================================
            # 52-WEEK ALERTS
            # =====================================================================
            if watchlist['week52_alerts']:
                print("\n" + "="*70)
                print("🎯 52-WEEK LEVEL ALERTS")
                print("="*70)
                
                for item in watchlist['week52_alerts']:
                    emoji = "🔺" if item['type'] == 'RESISTANCE' else "🔻"
                    print(f"{emoji} {item['ticker']}: {item['type']} at ${item['level']:.2f}")
            
            # =====================================================================
            # EARNINGS
            # =====================================================================
            if watchlist['earnings_this_week']:
                print("\n" + "="*70)
                print("📅 EARNINGS THIS WEEK")
                print("="*70)
                
                for item in watchlist['earnings_this_week']:
                    emoji = "🚨" if item['urgency'] == 'TOMORROW' else "📆"
                    print(f"{emoji} {item['ticker']}: {item['date']} ({item['days_until']} days)")
            
            # =====================================================================
            # PATTERN CONFIRMATIONS
            # =====================================================================
            if watchlist['pattern_confirmations']:
                print("\n" + "="*70)
                print("🎨 PATTERN CONFIRMATIONS NEEDED")
                print("="*70)
                
                for item in watchlist['pattern_confirmations'][:5]:
                    emoji = "🟢" if item['prediction'] == 'BUY' else "🔴"
                    print(f"\n{emoji} {item['ticker']} → {item['prediction']} ({item['confidence']}%)")
                    for need in item['needs']:
                        print(f"   • {need}")
            
            # =====================================================================
            # MACRO ALERTS
            # =====================================================================
            if watchlist['macro_alerts']:
                print("\n" + "="*70)
                print("🌍 MACRO ALERTS")
                print("="*70)
                
                for item in watchlist['macro_alerts']:
                    print(f"⚠️ {item['type']}: {item['message']}")
            
            # =====================================================================
            # KEY LEVELS SAMPLE
            # =====================================================================
            print("\n" + "="*70)
            print("🔑 KEY SUPPORT/RESISTANCE LEVELS (Sample - Top 3)")
            print("="*70)
            
            count = 0
            for ticker, levels in all_levels.items():
                if ticker in ['SPY', 'QQQ']:
                    continue
                
                if 'error' not in levels and count < 3:
                    print(f"\n{ticker}:")
                    print(f"   Current:    ${levels['current']:.2f}")
                    print(f"   Support:    ${levels['support']:.2f} ({levels['distance_to_support_pct']:+.1f}%)")
                    print(f"   Resistance: ${levels['resistance']:.2f} ({levels['distance_to_resistance_pct']:+.1f}%)")
                    
                    if levels['at_support']:
                        print(f"   🔻 AT SUPPORT")
                    elif levels['at_resistance']:
                        print(f"   🔺 AT RESISTANCE")
                    
                    count += 1
            
            print("\n" + "="*70)
            print("✅ MODULE 1 TEST COMPLETE - ALL SYSTEMS WORKING")
            print("="*70 + "\n")
        
        except Exception as e:
            print(f"\n❌ ERROR during test: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    asyncio.run(test_module_1())
