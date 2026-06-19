import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Add parent dir to path so we can import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from database.db_manager import DatabaseManager

# Page config
st.set_page_config(
    page_title="AI Trading System IDX",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize DB connection
@st.cache_resource
def get_db():
    return DatabaseManager()

db = get_db()

def main():
    st.title("🤖 Autonomous AI Trading System")
    st.markdown("### Indonesian Stock Exchange (IDX)")
    
    # System status banner
    mode_color = "green" if settings.is_paper_trading else "red"
    mode_text = "PAPER TRADING" if settings.is_paper_trading else "LIVE TRADING"
    st.markdown(f"**Mode:** <span style='color:{mode_color}; font-weight:bold;'>{mode_text}</span>", unsafe_allow_html=True)
    
    # Fetch latest metrics
    metrics = db.get_latest_metrics()
    
    # Top level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Get latest portfolio snapshot for equity
    snapshots = db.get_equity_curve(days=1)
    current_equity = snapshots[-1]["total_equity"] if snapshots else settings.INITIAL_CAPITAL
    total_pnl = snapshots[-1]["total_pnl"] if snapshots else 0
    total_pnl_pct = snapshots[-1]["total_pnl_pct"] if snapshots else 0
    
    with col1:
        st.metric(
            "Total Equity", 
            f"Rp {current_equity:,.0f}", 
            f"{total_pnl_pct:+.2f}%"
        )
    with col2:
        win_rate = metrics.get("win_rate", 0)
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col3:
        total_trades = metrics.get("total_trades", 0)
        st.metric("Total Trades", f"{int(total_trades)}")
    with col4:
        open_positions = len(db.get_open_trades())
        st.metric("Open Positions", f"{open_positions} / {settings.MAX_OPEN_POSITIONS}")

    st.divider()

    # Equity Curve
    st.subheader("📊 Equity Curve")
    
    days_to_show = st.radio("Timeframe", [7, 30, 90, 365], index=1, horizontal=True, format_func=lambda x: f"{x} Days")
    
    curve_data = db.get_equity_curve(days=days_to_show)
    
    if curve_data:
        df = pd.DataFrame(curve_data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"], 
            y=df["total_equity"],
            mode='lines',
            name='Equity',
            line=dict(color='#00ff00', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)'
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Equity (Rp)",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No equity data available yet. Let the system run to gather data.")

if __name__ == "__main__":
    main()
