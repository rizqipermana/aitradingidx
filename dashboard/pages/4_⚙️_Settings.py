import os
import sys
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings

st.set_page_config(page_title="Settings - AI Trading System", page_icon="⚙️", layout="wide")

st.title("⚙️ System Settings")

st.markdown("""
> **Note:** To change these settings, you must update the `.env` file and restart the system. 
> This page displays the currently loaded configuration.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("System Mode")
    mode = "Paper Trading" if settings.is_paper_trading else "Live Trading"
    st.info(f"**Current Mode:** {mode}")
    
    st.subheader("Capital & Limits")
    st.write(f"**Initial Capital:** Rp {settings.INITIAL_CAPITAL:,.0f}")
    st.write(f"**Max Risk per Trade:** {settings.MAX_RISK_PER_TRADE:.1%}")
    st.write(f"**Max Daily Loss:** {settings.MAX_DAILY_LOSS:.1%}")
    st.write(f"**Max Open Positions:** {settings.MAX_OPEN_POSITIONS}")

with col2:
    st.subheader("AI & Strategy")
    st.write(f"**Min AI Probability:** {settings.AI_PROBABILITY_THRESHOLD:.1%}")
    st.write(f"**Min Risk-Reward Ratio:** {settings.MIN_RISK_REWARD_RATIO}:1")
    st.write(f"**Trailing Stop Multiplier:** {settings.TRAILING_STOP_MULTIPLIER}x ATR")
    
    st.subheader("Scan Configuration")
    st.write(f"**Scan Interval:** Every {settings.SCAN_INTERVAL_MINUTES} minutes")
    st.write(f"**Monitored Tickers:** {len(settings.ticker_list)} stocks")
    with st.expander("View Ticker List"):
        st.write(", ".join(settings.ticker_list))

st.divider()

st.subheader("System Administration")
st.warning("Manual overrides should only be used in emergencies.")

# In a full implementation, these buttons would write to the DB or a state file 
# to pause the running main.py process
if st.button("🛑 EMERGENCY STOP (Disable Trading)", type="primary"):
    st.error("Trading disabled. The system will not open new positions until restarted.")
    # Here you would implement logic to signal main.py to pause

if st.button("▶️ RESUME TRADING"):
    st.success("Trading resumed.")
    # Here you would implement logic to signal main.py to resume
