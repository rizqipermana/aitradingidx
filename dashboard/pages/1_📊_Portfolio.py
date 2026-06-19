import os
import sys
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from database.db_manager import DatabaseManager

st.set_page_config(page_title="Portfolio - AI Trading System", page_icon="📊", layout="wide")

st.title("📊 Active Portfolio")

@st.cache_resource
def get_db():
    return DatabaseManager()

db = get_db()

# Fetch open trades
open_trades = db.get_open_trades()

if not open_trades:
    st.info("No active positions currently open.")
else:
    df = pd.DataFrame(open_trades)
    
    # We don't have current real-time price in DB directly without fetching, 
    # so we display what we have at entry
    
    display_df = pd.DataFrame({
        "Ticker": df["ticker"],
        "Action": df["action"],
        "Qty": df["quantity"],
        "Entry Price": df["entry_price"].apply(lambda x: f"Rp {x:,.0f}"),
        "Invested": (df["entry_price"] * df["quantity"]).apply(lambda x: f"Rp {x:,.0f}"),
        "Stop Loss": df["stop_loss"].apply(lambda x: f"Rp {x:,.0f}"),
        "Take Profit": df["take_profit"].apply(lambda x: f"Rp {x:,.0f}"),
        "Entry Time": pd.to_datetime(df["entry_time"]).dt.strftime("%Y-%m-%d %H:%M"),
    })
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Risk Allocation Chart
    st.subheader("Risk Allocation")
    
    df["Invested_Value"] = df["entry_price"] * df["quantity"]
    
    import plotly.express as px
    fig = px.pie(df, values='Invested_Value', names='ticker', hole=0.4, 
                 title="Capital Allocation by Asset")
    st.plotly_chart(fig, use_container_width=True)
