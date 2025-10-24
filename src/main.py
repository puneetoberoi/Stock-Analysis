def generate_enhanced_html_email(df_stocks, context, market_news, macro_data, memory, portfolio_data, pattern_data, ai_analysis, portfolio_recommendations=None):
    """FIXED v2.0.0: Clear, non-conflicting email display"""
    
    def format_articles(articles):
        if not articles:
            return "<p style='color:#888;'><i>No specific news drivers detected.</i></p>"
        html = "<ul style='margin:0;padding-left:20px;'>"
        for a in articles:
            if a.get('title'):
                html += f'<li style="margin-bottom:5px;"><a href="{a.get("url", "#")}" style="color:#1e3a8a;">{a["title"]}</a> <span style="color:#666;">({a.get("source", "Unknown")})</span></li>'
        return html + "</ul>"
    
    def create_stock_table(df):
        return "".join([f'<tr><td style="padding:10px;border-bottom:1px solid #eee;"><b>{row["ticker"]}</b><br><span style="color:#666;font-size:0.9em;">{row["name"]}</span></td><td style="padding:10px;border-bottom:1px solid #eee;text-align:center;font-weight:bold;font-size:1.1em;">{row["score"]:.0f}</td></tr>' for _, row in df.iterrows()])
    
    def create_context_table(ids):
        rows = ""
        for asset_id in ids:
            if asset := context.get(asset_id):
                price = f"${asset.get('current_price', 0):,.2f}" if asset.get('current_price') else "N/A"
                change_24h = asset.get('price_change_percentage_24h', 0) or 0
                mcap = f"${asset.get('market_cap', 0) / 1_000_000_000:.1f}B" if asset.get('market_cap') else "N/A"
                color_24h = "#16a34a" if change_24h >= 0 else "#dc2626"
                rows += f'<tr><td style="padding:10px;border-bottom:1px solid #eee;"><b>{asset.get("name", "")}</b><br><span style="color:#666;font-size:0.9em;">{asset.get("symbol","").upper()}</span></td><td style="padding:10px;border-bottom:1px solid #eee;">{price}<br><span style="color:{color_24h};font-size:0.9em;">{change_24h:.2f}% (24h)</span></td><td style="padding:10px;border-bottom:1px solid #eee;">{mcap}</td></tr>'
        return rows
    
    # AI Oracle section
    ai_oracle_html = ""
    if ai_analysis:
        analysis_text = ai_analysis['analysis'].replace('\n', '<br>')
        ai_oracle_html = f"""<div class="section" style="background-color:#f0f9ff;border-left:4px solid #0369a1;">
        <h2>🤖 AI MARKET ORACLE</h2>
        <p style="font-size:0.9em;color:#666;margin-bottom:15px;">Powered by Gemini AI</p>
        <div style="line-height:1.8;">{analysis_text}</div>
        </div>"""
    
    # v2.0.0 NEW: High-priority signals section
    v2_signals_html = ""
    if ENABLE_V2_FEATURES and portfolio_data and portfolio_data.get('v2_signals'):
        signals_list = "<br>".join(portfolio_data['v2_signals'][:10])
        v2_signals_html = f"""<div class="section" style="background-color:#fef3c7;border-left:4px solid #f59e0b;">
        <h2>⚡ HIGH-PRIORITY SIGNALS (v2.0)</h2>
        <p style="font-size:0.9em;color:#666;">Advanced technical alerts requiring immediate attention</p>
        <div style="line-height:2; font-weight:500;">{signals_list if signals_list else 'No high-priority signals today'}</div>
        </div>"""
    
    # Portfolio section with v2.0.0 enhancements
    portfolio_html = ""
    if portfolio_data:
        portfolio_table = ""
        for stock in portfolio_data['stocks']:
            color = "#16a34a" if stock['daily_change'] > 0 else "#dc2626"
            rsi_color = "#dc2626" if stock['rsi'] > 70 else "#16a34a" if stock['rsi'] < 30 else "#666"
            
            # v2.0.0: Add Bollinger Band indicator
            bb_indicator = ""
            if stock.get('bollinger'):
                bb = stock['bollinger']
                if bb['squeeze']:
                    bb_indicator = f"<br><span style='color:#f59e0b;font-weight:bold;'>💥 SQUEEZE</span>"
                elif bb['position'] > 90:
                    bb_indicator = f"<br><span style='color:#dc2626;'>BB: {bb['position']:.0f}%</span>"
                elif bb['position'] < 10:
                    bb_indicator = f"<br><span style='color:#16a34a;'>BB: {bb['position']:.0f}%</span>"
            
            # v2.0.0: Add 52-week indicator
            week52_indicator = ""
            if stock.get('week52'):
                w52 = stock['week52']
                if 'AT 52W HIGH' in w52['signal']:
                    week52_indicator = f"<br><span style='color:#16a34a;font-weight:bold;'>🚀 52W HIGH</span>"
                elif 'AT 52W LOW' in w52['signal']:
                    week52_indicator = f"<br><span style='color:#dc2626;font-weight:bold;'>📉 52W LOW</span>"
            
            # v2.0.0: Add earnings countdown
            earnings_indicator = ""
            if stock.get('earnings') and stock['earnings']['days_until'] <= 7:
                earnings_indicator = f"<br><span style='color:#9333ea;font-weight:bold;'>📅 Earnings in {stock['earnings']['days_until']}d</span>"
            
            portfolio_table += f"""<tr>
            <td style="padding:10px;border-bottom:1px solid #eee;">
                <b>{stock['ticker']}</b><br>
                <span style="color:#666;font-size:0.9em;">{stock['name']}</span>
                {bb_indicator}{week52_indicator}{earnings_indicator}
            </td>
            <td style="padding:10px;border-bottom:1px solid #eee;">
                ${stock['price']:.2f}<br>
                <span style="color:{color};font-size:0.9em;">{stock['daily_change']:+.2f}%</span>
            </td>
            <td style="padding:10px;border-bottom:1px solid #eee;">
                RSI: <span style="color:{rsi_color};font-weight:bold;">{stock['rsi']:.1f}</span><br>
                <span style="font-size:0.9em;">Vol: {stock['volume_ratio']:.1f}x</span>
            </td>
            <td style="padding:10px;border-bottom:1px solid #eee;">
                <span style="font-size:0.9em;">W: {stock['weekly_change']:+.1f}%<br>M: {stock['monthly_change']:+.1f}%</span>
            </td>
            </tr>"""
        
        alerts_html = "<br>".join(portfolio_data['alerts'][:5]) if portfolio_data['alerts'] else "No alerts today"
        opps_html = "<br>".join(portfolio_data['opportunities'][:5]) if portfolio_data['opportunities'] else "No immediate opportunities"
        risks_html = "<br>".join(portfolio_data['risks'][:5]) if portfolio_data['risks'] else "No significant risks detected"
        
        portfolio_html = f"""<div class="section" style="background-color:#fefce8;border-left:4px solid #ca8a04;">
        <h2>📊 YOUR PORTFOLIO COMMAND CENTER</h2>
        <table style="width:100%; border-collapse: collapse; margin-bottom:20px;">
            <thead><tr style="background-color:#f8f8f8;">
                <th style="text-align:left; padding:10px;">Stock</th>
                <th style="text-align:left; padding:10px;">Price</th>
                <th style="text-align:left; padding:10px;">Indicators</th>
                <th style="text-align:left; padding:10px;">Performance</th>
            </tr></thead>
            <tbody>{portfolio_table}</tbody>
        </table>
        <div style="margin-top:20px;">
            <h3 style="color:#dc2626;">🔔 Alerts & Signals</h3>
            <p style="line-height:1.8;">{alerts_html}</p>
        </div>
        <div style="margin-top:20px;">
            <h3 style="color:#16a34a;">🎯 Opportunities</h3>
            <p style="line-height:1.8;">{opps_html}</p>
        </div>
        <div style="margin-top:20px;">
            <h3 style="color:#ea580c;">⚠️ Risk Factors</h3>
            <p style="line-height:1.8;">{risks_html}</p>
        </div>
        </div>"""

    # Add this AFTER the portfolio_html section in generate_enhanced_html_email

    # ========================================
    # 🎯 AI PREDICTIONS WITH CONFIDENCE SCORING
    # ========================================
    
    ai_predictions_html = ""
if portfolio_data and portfolio_data.get('learning_active'):
    predictions_made = portfolio_data.get('predictions_made', 0)
    
    if predictions_made > 0:
        prediction_cards = []
        for stock in portfolio_data['stocks']:
            if 'ai_prediction' not in stock or not stock['ai_prediction']:
                continue
            
            pred = stock['ai_prediction']
            conf = stock.get('confidence', {})
            
            action = pred.get('action', 'HOLD')
            conf_score = conf.get('score', 50)
            
            action_color = {'BUY': '#28a745', 'SELL': '#dc3545', 'HOLD': '#6c757d'}.get(action)
            action_icon = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '⚪'}.get(action)
            
            # Create the detailed breakdown list
            breakdown_list = ""
            if conf.get('breakdown'):
                breakdown_list = "<ul>"
                for item in conf['breakdown']:
                    icon = "✅" if "+" in item else "⚠️" if "-" in item else "⚖️"
                    breakdown_list += f'<li style="font-size: 0.9em; color: #555; margin-bottom: 4px;">{icon} {item}</li>'
                breakdown_list += "</ul>"

            prediction_cards.append(f"""
            <div style="border: 1px solid #ddd; border-left: 5px solid {action_color}; border-radius: 8px; margin-bottom: 20px; padding: 20px; background: #fff;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <h3 style="margin: 0 0 5px 0; font-size: 1.5em;">{action_icon} {stock['ticker']} &rarr; {action}</h3>
                        <p style="margin: 0; color: #777; font-size: 1em;">{stock['name']}</p>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 2.2em; font-weight: bold; color: {action_color};">{conf_score}%</div>
                        <div style="font-size: 0.8em; font-weight: bold; color: {action_color};">CONVICTION SCORE</div>
                    </div>
                </div>
                <div style="margin: 20px 0; padding-top: 20px; border-top: 1px solid #eee;">
                    <h4 style="margin: 0 0 10px 0; font-size: 1em; color: #333;">Confidence Factors:</h4>
                    {breakdown_list}
                </div>
                <div style="margin-top: 15px; padding: 10px; background: #f9f9f9; border-radius: 5px; font-size: 0.9em; color: #444;">
                    <strong>Consensus Reasoning:</strong> {pred.get('reasoning', 'N/A')}
                </div>
            </div>
            """)

        if prediction_cards:
            ai_predictions_html = f"""
                <div class="section" style="background-color:#f4f7f6;">
                    <h2 class="section-title">🎯 AI Predictions & Conviction Analysis</h2>
                    {''.join(prediction_cards)}
                </div>
                """
                logging.info("✅ AI predictions HTML generated successfully")
            else:
                logging.warning("⚠️ No prediction cards created despite predictions_made > 0")
        else:
            logging.info("No predictions to display (predictions_made = 0)")
    
    logging.info("=" * 60)
    
    # Pattern analysis section (from v1.0.0 - keeping stable)
    pattern_html = ""
    if pattern_data and pattern_data.get('matches'):
        current_cond_data = pattern_data['current_conditions']
        current_cond = f"""<div style="background-color:#fff;padding:15px;border:2px solid #7c3aed;border-radius:5px;margin-bottom:20px;">
        <h3 style="margin-top:0;">📊 Today's Market DNA:</h3>
        <p style="margin:5px 0;"><b>RSI:</b> {current_cond_data['rsi']:.1f} | <b>Volatility:</b> {current_cond_data['volatility']:.1f}% | <b>Trend:</b> {current_cond_data['trend']:+.1f}%</p>
        <p style="margin:5px 0;"><b>Geopolitical Risk:</b> {current_cond_data['geo_risk']:.0f} | <b>Trade Risk:</b> {current_cond_data['trade_risk']:.0f}</p>
        </div>"""
        
        interpretation_html = ""
        if pattern_data.get('interpretation'):
            for item in pattern_data['interpretation']:
                if item['type'] == 'bias':
                    color = {'bullish': '#16a34a', 'bearish': '#dc2626', 'neutral': '#666'}[item['color']]
                    interpretation_html += f'<p style="font-size:1.2em;font-weight:bold;color:{color};">{item["emoji"]} {item["text"]}</p>'
                else:
                    interpretation_html += f'<p style="line-height:1.8;margin:10px 0;">{item["text"]}</p>'
        
        sector_html_patterns = ""
        if pattern_data.get('sector_performance'):
            sector_rows = ""
            for sector, perf in pattern_data['sector_performance'][:5]:
                color = "#16a34a" if perf > 0 else "#dc2626"
                sector_rows += f'<tr><td style="padding:8px;border-bottom:1px solid #eee;">{sector}</td><td style="padding:8px;border-bottom:1px solid #eee;text-align:right;color:{color};font-weight:bold;">{perf:+.1f}%</td></tr>'
            
            sector_html_patterns = f"""<div style="margin:20px 0;">
            <h3>🎯 Sector Performance in Similar Periods:</h3>
            <p style="font-size:0.9em;color:#666;">Based on {pattern_data['matches'][0]['date']} match ({pattern_data['matches'][0]['context']})</p>
            <table style="width:100%;background-color:#fff;border-collapse:collapse;">
                <thead><tr style="background-color:#f3e8ff;">
                    <th style="padding:10px;text-align:left;">Sector</th>
                    <th style="padding:10px;text-align:right;">3-Month Return</th>
                </tr></thead>
                <tbody>{sector_rows}</tbody>
            </table>
            </div>"""
        
        recommendations_html = ""
        if portfolio_recommendations and portfolio_recommendations.get('final_verdicts'):
            rec_items = []
            for ticker, verdict in portfolio_recommendations['final_verdicts'].items():
                action = verdict['action']
                reason = verdict['reason']
                confidence = verdict.get('confidence', 'MEDIUM')
                name = verdict.get('name', ticker)
                
                if action in ['BUY', 'BUY DIP', 'BUY STARTER', 'ADD', 'AVERAGE DOWN']:
                    color = "#16a34a"
                    icon = "🟢"
                elif action in ['SELL', 'TRIM 50%', 'TAKE PROFITS', 'REDUCE']:
                    color = "#dc2626" 
                    icon = "🔴"
                elif action in ['HOLD', 'WATCH']:
                    color = "#666"
                    icon = "⚪"
                elif action == 'HEDGE':
                    color = "#ea580c"
                    icon = "🟡"
                else:
                    color = "#666"
                    icon = "⚪"
                
                conf_badge = ""
                if confidence == 'HIGH':
                    conf_badge = '<span style="background:#dc2626;color:white;padding:2px 6px;border-radius:3px;font-size:0.8em;margin-left:8px;">HIGH CONF</span>'
                elif confidence == 'MEDIUM':
                    conf_badge = '<span style="background:#f59e0b;color:white;padding:2px 6px;border-radius:3px;font-size:0.8em;margin-left:8px;">MED CONF</span>'
                elif confidence == 'LOW':
                    conf_badge = '<span style="background:#6b7280;color:white;padding:2px 6px;border-radius:3px;font-size:0.8em;margin-left:8px;">LOW CONF</span>'
                
                rec_items.append(f"""
                <div style="margin:10px 0;padding:12px;background-color:#f9f9f9;border-left:4px solid {color};border-radius:5px;">
                    <div style="font-size:1.1em;font-weight:bold;color:{color};">
                        {icon} {ticker}: {action} {conf_badge}
                    </div>
                    <div style="font-size:0.9em;color:#333;margin-top:2px;">{name}</div>
                    <div style="font-size:0.9em;color:#666;margin-top:5px;">• {reason}</div>
                </div>
                """)
            
            recommendations_html = f"""<div style="margin:20px 0;">
            <h3 style="color:#7c3aed;">💼 YOUR PORTFOLIO ACTION PLAN</h3>
            <p style="font-size:0.9em;color:#666;">One clear recommendation per stock - no conflicts</p>
            {''.join(rec_items)}
            </div>"""
        
        matches_html = ""
        for i, match in enumerate(pattern_data['matches'][:5], 1):
            outcome_color = "#16a34a" if match['future_3m'] > 0 else "#dc2626"
            matches_html += f"""<div style="margin:15px 0;padding:15px;background-color:#f8f8f8;border-left:4px solid {outcome_color};border-radius:5px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <b style="font-size:1.1em;">{match['date']}</b>
                    <span style="color:#666;margin-left:10px;">({match['context']})</span><br>
                    <span style="font-size:0.9em;color:#666;">Match Strength: {match['similarity']:.1f}%</span>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:1.2em;font-weight:bold;color:{outcome_color};">{match['future_3m']:+.1f}%</div>
                    <div style="font-size:0.9em;color:#666;">S&P 500 outcome</div>
                </div>
            </div>
            </div>"""
        
        pattern_html = f"""<div class="section" style="background-color:#f3e8ff;border-left:4px solid #7c3aed;">
        <h2>🔮 11-YEAR PATTERN ANALYSIS</h2>
        <p style="font-size:0.9em;color:#666;">Analyzing {pattern_data['sample_size']} similar market setups</p>
        {current_cond}
        <div style="background-color:#fff;padding:20px;border-radius:5px;margin:20px 0;">
            <h3 style="margin-top:0;color:#7c3aed;">📖 What History Tells Us:</h3>
            {interpretation_html}
        </div>
        {sector_html_patterns}
        {recommendations_html}
        <div style="margin:20px 0;">
            <h3>📅 Historical Matches:</h3>
            <p style="font-size:0.9em;color:#666;">These show S&P 500 performance. Sector performance varied (see table above).</p>
            {matches_html}
        </div>
        </div>"""
    
    # Editor's note
    prev_score = memory.get('previous_macro_score', 0)
    current_score = macro_data.get('overall_macro_score', 0)
    mood_change = "stayed relatively stable"
    if (diff := current_score - prev_score) > 3:
        mood_change = f"improved since yesterday (from {prev_score:.1f} to {current_score:.1f})"
    elif diff < -3:
        mood_change = f"turned more cautious since yesterday (from {prev_score:.1f} to {current_score:.1f})"
    
    editor_note = f"Good morning. The overall market mood has {mood_change}. This briefing is your daily blueprint for navigating the currents."
    if memory.get('previous_top_stock_name'):
        editor_note += f"<br><br><b>Yesterday's Champion:</b> {memory['previous_top_stock_name']} ({memory['previous_top_stock_ticker']}) led our rankings."
    
    # Sector deep dive
    sector_html = ""
    if not df_stocks.empty:
        top_by_sector = df_stocks.groupby('sector', group_keys=False)[['ticker', 'name', 'score', 'sector', 'summary']].apply(lambda x: x.nlargest(2, 'score'))
        for _, row in top_by_sector.iterrows():
            if row['sector'] and row['sector'] != 'N/A':
                summary_text = "Business summary not available."
                if row["summary"] and isinstance(row["summary"], str):
                    summary_text = '. '.join(row["summary"].split('. ')[:2]) + '.'
                sector_html += f'<div style="margin-bottom:15px;"><b>{row["name"]} ({row["ticker"]})</b> in <i>{row["sector"]}</i><p style="font-size:0.9em;color:#333;margin:5px 0 0 0;">{summary_text}</p></div>'
    
    # Create tables
    top10_html = create_stock_table(df_stocks.head(10)) if not df_stocks.empty else "<tr><td>No data available</td></tr>"
    bottom10_html = create_stock_table(df_stocks.tail(10).iloc[::-1]) if not df_stocks.empty else "<tr><td>No data available</td></tr>"
    crypto_html = create_context_table(["bitcoin", "ethereum", "solana", "ripple"])
    commodities_html = create_context_table(["gold", "silver"])
    market_news_html = "".join([f'<div style="margin-bottom:15px;"><b><a href="{article.get("url", "#")}" style="color:#000;">{article["title"]}</a></b><br><span style="color:#666;font-size:0.9em;">{article.get("source", "Unknown")}</span></div>' for article in market_news[:10]]) or "<p><i>Headlines temporarily unavailable.</i></p>"
    
    # Assemble final email
    return f"""<!DOCTYPE html><html><head><style>
    body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:0;background-color:#f7f7f7;}}
    .container{{width:100%;max-width:700px;margin:20px auto;background-color:#fff;border:1px solid #ddd;}}
    .header{{background-color:#0c0a09;color:#fff;padding:30px;text-align:center;}}
    .section{{padding:25px;border-bottom:1px solid #ddd;}}
    h2{{font-size:1.5em;color:#111;margin-top:0;}}
    h3{{font-size:1.2em;color:#333;border-bottom:2px solid #e2e8f0;padding-bottom:5px;}}
    </style></head><body>
    <div class="container">
        <div class="header">
            <h1>Your Daily Intelligence Briefing</h1>
            <p style="font-size:1.1em; color:#aaa;">{datetime.now().strftime('%A, %B %d, %Y')}</p>
        </div>
        
        <div class="section">
            <h2>EDITOR'S NOTE</h2>
            <p>{editor_note}</p>
        </div>
        
        {v2_signals_html}
        {ai_oracle_html}
        {portfolio_html}
        {ai_predictions_html}
        {pattern_html}
        
        <div class="section">
            <h2>THE BIG PICTURE: The Market Weather Report</h2>
            <h3>Overall Macro Score: {macro_data['overall_macro_score']:.1f} / 30</h3>
            <p><b>How it's calculated:</b> This is our "weather forecast" for investors, combining risks and sentiment.</p>
            <p><b>🌍 Geopolitical Risk ({macro_data['geopolitical_risk']:.0f}/100):</b> Measures global instability.<br>
            <u>Key Drivers:</u> {format_articles(macro_data['geo_articles'])}</p>
            <p><b>🚢 Trade Risk ({macro_data['trade_risk']:.0f}/100):</b> Tracks trade tensions.<br>
            <u>Key Drivers:</u> {format_articles(macro_data['trade_articles'])}</p>
            <p><b>💼 Economic Sentiment ({macro_data['economic_sentiment']:.2f}):</b> Market mood (-1 to +1).<br>
            <u>Key Drivers:</u> {format_articles(macro_data['econ_articles'])}</p>
        </div>
        
        <div class="section">
            <h2>SECTOR DEEP DIVE</h2>
            <p>Top companies from different sectors.</p>
            {sector_html or "<p><i>No sector data available.</i></p>"}
        </div>
        
        <div class="section">
            <h2>STOCK RADAR</h2>
            <h3>📈 Top 10 Strongest Signals</h3>
            <table style="width:100%; border-collapse: collapse;">
                <thead><tr>
                    <th style="text-align:left; padding:10px;">Company</th>
                    <th style="text-align:center; padding:10px;">Score</th>
                </tr></thead>
                <tbody>{top10_html}</tbody>
            </table>
            
            <h3 style="margin-top: 30px;">📉 Top 10 Weakest Signals</h3>
            <table style="width:100%; border-collapse: collapse;">
                <thead><tr>
                    <th style="text-align:left; padding:10px;">Company</th>
                    <th style="text-align:center; padding:10px;">Score</th>
                </tr></thead>
                <tbody>{bottom10_html}</tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>BEYOND STOCKS: Alternative Assets</h2>
            <h3>🪙 Crypto</h3>
            <p><b>Market Sentiment: {context.get('crypto_sentiment', 'N/A')}</b></p>
            <table style="width:100%; border-collapse: collapse;">
                <thead><tr>
                    <th style="text-align:left; padding:10px;">Asset</th>
                    <th style="text-align:left; padding:10px;">Price / 24h</th>
                    <th style="text-align:left; padding:10px;">Market Cap</th>
                </tr></thead>
                <tbody>{crypto_html}</tbody>
            </table>
            
            <h3 style="margin-top: 30px;">💎 Commodities</h3>
            <p><b>Gold/Silver Ratio: {context.get('gold_silver_ratio', 'N/A')}</b></p>
            <table style="width:100%; border-collapse: collapse;">
                <thead><tr>
                    <th style="text-align:left; padding:10px;">Asset</th>
                    <th style="text-align:left; padding:10px;">Price / 24h</th>
                    <th style="text-align:left; padding:10px;">Market Cap</th>
                </tr></thead>
                <tbody>{commodities_html}</tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>FROM THE WIRE: Today's Top Headlines</h2>
            {market_news_html}
        </div>
    </div>
    </body></html>"""
