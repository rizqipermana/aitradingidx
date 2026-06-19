import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from database.db_manager import DatabaseManager

st.set_page_config(page_title="Trade History - AI Trading System", page_icon="📜", layout="wide")

st.title("📜 Trade History")

@st.cache_resource
def get_db():
    return DatabaseManager()

db = get_db()

trades = db.get_trade_history(limit=500)

# Filter out OPEN trades for history view
history = [t for t in trades if t["status"] == "CLOSED"]

if not history:
    st.info("No closed trades yet.")
else:
    df = pd.DataFrame(history)
    
    # Summary Metrics
    total_profit = df["profit_loss"].sum()
    win_count = len(df[df["profit_loss"] > 0])
    loss_count = len(df[df["profit_loss"] <= 0])
    win_rate = win_count / len(df) * 100 if len(df) > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total P&L", f"Rp {total_profit:,.0f}")
    col2.metric("Win Rate", f"{win_rate:.1f}%")
    col3.metric("Profitable Trades", f"{win_count}")
    col4.metric("Losing Trades", f"{loss_count}")
    
    st.divider()
    
    # Profit Distribution Chart
    fig_pnl = px.bar(
        df, 
        x=pd.to_datetime(df["exit_time"]), 
        y="profit_loss",
        color=df["profit_loss"] > 0,
        color_discrete_map={True: "green", False: "red"},
        labels={"exit_time": "Exit Date", "profit_loss": "Profit/Loss (Rp)", "color": "Profitable"},
        title="P&L per Trade Over Time"
    )
    st.plotly_chart(fig_pnl, use_container_width=True)
    
    # Trade List
    st.subheader("All Closed Trades")
    
    display_df = pd.DataFrame({
        "Ticker": df["ticker"],
        "Entry Time": pd.to_datetime(df["entry_time"]).dt.strftime("%Y-%m-%d %H:%M"),
        "Exit Time": pd.to_datetime(df["exit_time"]).dt.strftime("%Y-%m-%d %H:%M"),
        "Qty": df["quantity"],
        "Entry": df["entry_price"].apply(lambda x: f"{x:,.0f}"),
        "Exit": df["exit_price"].apply(lambda x: f"{x:,.0f}"),
        "P&L": df["profit_loss"].apply(lambda x: f"Rp {x:,.0f}"),
        "P&L %": df["profit_loss_pct"].apply(lambda x: f"{x:+.2f}%"),
        "Reason": df["close_reason"]
    })
    
    def style_pnl(val):
        if val.startswith("+") or not val.startswith("-") and val != "0.00%":
            return 'color: green'
        elif val.startswith("-"):
            return 'color: red'
        return ''
        
    st.dataframe(
        display_df.style.applymap(style_pnl, subset=["P&L %"]),
        use_container_width=True,
        hide_index=True
    )
