import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from database.db_manager import DatabaseManager

st.set_page_config(page_title="Signals - AI Trading System", page_icon="📈", layout="wide")

st.title("📈 Trading Signals")

@st.cache_resource
def get_db():
    return DatabaseManager()

db = get_db()

# Filter controls
col1, col2 = st.columns([1, 3])
with col1:
    limit = st.selectbox("Show last N signals", [50, 100, 500])
with col2:
    action_filter = st.multiselect("Filter by Action", ["BUY", "SELL", "HOLD"], default=["BUY", "SELL"])

signals = db.get_recent_signals(limit=limit)

if not signals:
    st.info("No signals generated yet.")
else:
    df = pd.DataFrame(signals)
    
    if action_filter:
        df = df[df["action"].isin(action_filter)]
    
    if df.empty:
        st.warning("No signals match the current filters.")
    else:
        # Style formatting
        def style_action(val):
            if val == "BUY":
                return 'color: green; font-weight: bold'
            elif val == "SELL":
                return 'color: red; font-weight: bold'
            return 'color: gray'

        display_df = pd.DataFrame({
            "Time": pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M"),
            "Ticker": df["ticker"],
            "Action": df["action"],
            "AI Prob": df["ai_probability"].apply(lambda x: f"{x:.1%}"),
            "Price": df["entry_price"].apply(lambda x: f"Rp {x:,.0f}"),
            "Executed": df["executed"].apply(lambda x: "✅" if x else "❌"),
            "Reason": df["reason"]
        })
        
        st.dataframe(
            display_df.style.applymap(style_action, subset=["Action"]),
            use_container_width=True,
            hide_index=True
        )
        
        st.divider()
        
        # Probabilities Distribution
        st.subheader("AI Probability Distribution")
        
        buy_signals = pd.DataFrame(db.get_recent_signals(limit=500))
        if not buy_signals.empty:
            buy_signals = buy_signals[buy_signals["action"] == "BUY"]
            if not buy_signals.empty:
                fig = px.histogram(
                    buy_signals, 
                    x="ai_probability", 
                    nbins=20,
                    title="Distribution of AI Confidence for BUY Signals",
                    labels={"ai_probability": "Probability of Price Increase"},
                    color_discrete_sequence=["#00CC96"]
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough BUY signals for distribution chart.")
