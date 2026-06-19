"""
AI Trading System IDX - Main Orchestrator

Initializes all system components and schedules the main trading loop.
Handles graceful shutdown and periodic tasks.

Usage:
    python main.py
"""

import signal
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

from config import settings
from core.ai_predictor import AIPredictor
from core.data_layer import MarketDataFetcher
from core.execution_engine import ExecutionEngine
from core.feature_engine import FeatureEngine
from core.portfolio_manager import PortfolioManager
from core.risk_manager import RiskManager
from core.signal_engine import SignalEngine
from database.db_manager import DatabaseManager
from notifications.telegram_bot import TelegramBot
from utils.logger import get_logger, setup_logging

# Set up root logger first
setup_logging(log_level=settings.LOG_LEVEL, log_file=settings.LOG_FILE)
logger = get_logger(__name__)


class TradingSystem:
    """Main orchestrator for the AI Trading System."""

    def __init__(self):
        logger.info("Initializing AI Trading System IDX...")
        
        # 1. Initialize Database
        self.db = DatabaseManager()
        
        # 2. Initialize Telegram Bot
        self.telegram = TelegramBot()
        
        # 3. Initialize Core Components
        self.data_fetcher = MarketDataFetcher()
        self.feature_engine = FeatureEngine()
        self.ai_predictor = AIPredictor()
        
        # 4. Initialize Risk & Portfolio Management
        self.risk_manager = RiskManager()
        self.portfolio = PortfolioManager()
        
        # 5. Initialize Engines
        self.signal_engine = SignalEngine(self.risk_manager, self.ai_predictor)
        self.execution_engine = ExecutionEngine(self.portfolio, self.risk_manager)
        
        # Scheduler
        self.scheduler = BackgroundScheduler()
        
        # State
        self.is_running = False

    def check_open_positions(self) -> None:
        """Check all open positions against SL/TP and trail stops."""
        if not self.portfolio.open_positions:
            return

        logger.info("Checking open positions for SL/TP...")
        
        # Fetch latest prices for open positions
        current_prices = {}
        for ticker in self.portfolio.open_positions.keys():
            price = self.data_fetcher.get_latest_price(ticker)
            if price:
                current_prices[ticker] = price
                
        # Check SL/TP
        to_close = self.portfolio.check_stop_loss_take_profit(current_prices)
        
        # Execute closing orders
        for ticker, reason, price in to_close:
            result = self.execution_engine.close_position_for_sl_tp(
                ticker=ticker,
                exit_price=price,
                reason=reason
            )
            
            if result and result.status.value == "FILLED":
                # Get the closed position to find P&L
                # The portfolio manager already moved it to closed_positions
                closed_pos = self.portfolio.closed_positions[-1]
                
                # Save to DB
                self.db.close_trade(
                    ticker=ticker,
                    exit_price=result.filled_price,
                    profit_loss=closed_pos.profit_loss,
                    profit_loss_pct=closed_pos.profit_loss_pct,
                    close_reason=reason
                )
                
                # Log execution
                self._save_execution_log(ticker, "SELL", "FILLED", f"Closed due to {reason}", result.order_id)
                
                # Notify
                self.telegram.send_position_closed(
                    ticker=ticker,
                    action="SELL",
                    entry_price=closed_pos.entry_price,
                    exit_price=result.filled_price,
                    pnl=closed_pos.profit_loss,
                    pnl_pct=closed_pos.profit_loss_pct,
                    reason=reason
                )

    def _save_trade_to_db(self, ticker: str, action: str, result: dict, signal: dict) -> None:
        """Helper to save a successful trade to database."""
        trade_data = {
            "ticker": ticker,
            "action": action,
            "entry_price": result.filled_price,
            "quantity": result.quantity,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "status": "OPEN",
        }
        self.db.save_trade(trade_data)

    def _save_execution_log(self, ticker: str, action: str, status: str, message: str, order_id: str = "") -> None:
        """Helper to save execution log."""
        log_data = {
            "ticker": ticker,
            "action": action,
            "status": status,
            "message": message,
            "order_id": order_id
        }
        self.db.save_execution_log(log_data)

    def _save_signal_to_db(self, signal) -> None:
        """Helper to save a signal to database."""
        signal_data = {
            "ticker": signal.ticker,
            "action": signal.action,
            "confidence": signal.confidence,
            "ai_probability": signal.ai_probability,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "reason": signal.reason,
            "executed": False, # Updated later if executed
        }
        return self.db.save_signal(signal_data)

    def market_scan_cycle(self) -> None:
        """
        Main trading loop:
        1. Fetch data
        2. Generate features
        3. Check SL/TP on existing positions
        4. Generate new signals
        5. Execute valid signals
        """
        if not self.is_running:
            return

        try:
            logger.info("="*50)
            logger.info(f"STARTING MARKET SCAN CYCLE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("="*50)

            # 1. Fetch data
            tickers = settings.ticker_list
            data_dict = self.data_fetcher.scan_all_tickers(tickers, days=100)
            
            if not data_dict:
                logger.error("Failed to fetch market data. Skipping cycle.")
                self.telegram.send_error("Failed to fetch market data during scan cycle.")
                return

            # 2. Enrich data with features
            enriched_data = {}
            for ticker, df in data_dict.items():
                features_df = self.feature_engine.calculate_all(df)
                if features_df is not None:
                    enriched_data[ticker] = features_df

            # 3. Check existing positions (SL/TP)
            self.check_open_positions()
            
            # 4. Generate new signals
            portfolio_state = self.portfolio.get_portfolio_state()
            signals = self.signal_engine.scan_and_generate(enriched_data, portfolio_state)
            
            # 5. Execute signals
            for signal in signals:
                # Save signal to DB
                signal_id = self._save_signal_to_db(signal)
                
                # Pre-trade notification
                self.telegram.send_trade_signal(signal, executed=False)
                
                # Execute
                result = self.execution_engine.execute_signal(signal)
                
                if result:
                    # Log execution
                    self._save_execution_log(
                        signal.ticker, 
                        signal.action, 
                        result.status.value, 
                        result.message, 
                        result.order_id
                    )
                    
                    if result.status.value == "FILLED":
                        # Mark signal as executed
                        if signal_id:
                            # Not implemented in DB manager yet, but we could update signal status
                            pass
                            
                        # Save trade to DB
                        self._save_trade_to_db(signal.ticker, signal.action, result, signal)
                        
                        # Post-trade notification
                        self.telegram.send_trade_signal(signal, executed=True)
            
            # Take portfolio snapshot at end of cycle
            snapshot = self.portfolio.take_equity_snapshot()
            self.db.save_portfolio_snapshot({
                "total_equity": snapshot.total_equity,
                "cash_balance": snapshot.cash_balance,
                "positions_value": snapshot.positions_value,
                "daily_pnl": snapshot.daily_pnl,
                "total_pnl": snapshot.total_pnl,
                "total_pnl_pct": snapshot.total_pnl_pct,
                "win_rate": self.portfolio.win_rate,
                "total_trades": self.portfolio.total_trades,
                "open_positions": self.portfolio.open_position_count,
                "max_drawdown": self.portfolio.max_drawdown,
            })
            
            logger.info("MARKET SCAN CYCLE COMPLETED")
            
        except Exception as e:
            logger.error(f"Error during market scan cycle: {e}", exc_info=True)
            self.telegram.send_error(f"Exception in main loop: {e}", is_critical=True)

    def daily_summary(self) -> None:
        """Generate and send daily summary."""
        if not self.is_running:
            return
            
        summary = self.portfolio.get_summary()
        self.telegram.send_daily_summary(summary)
        logger.info(f"Daily summary generated: Equity=Rp {summary['total_equity']:,.0f}, PnL=Rp {summary['daily_pnl']:,.0f}")

    def start(self) -> None:
        """Start the trading system."""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Schedule main loop based on config interval
        interval_mins = settings.SCAN_INTERVAL_MINUTES
        self.scheduler.add_job(
            self.market_scan_cycle, 
            'interval', 
            minutes=interval_mins,
            id='market_scan',
            replace_existing=True
        )
        
        # Schedule daily summary at 16:30 (after market close)
        self.scheduler.add_job(
            self.daily_summary,
            'cron',
            hour=16,
            minute=30,
            id='daily_summary',
            replace_existing=True
        )
        
        self.scheduler.start()
        
        mode = "PAPER TRADING" if settings.is_paper_trading else "LIVE TRADING"
        start_msg = f"🚀 AI Trading System Started in {mode} mode.\nMonitoring {len(settings.ticker_list)} stocks."
        logger.info(start_msg)
        self.telegram.send_message(start_msg)
        
        # Run first cycle immediately
        self.scheduler.add_job(self.market_scan_cycle, 'date', run_date=datetime.now())

    def stop(self) -> None:
        """Stop the trading system."""
        self.is_running = False
        self.scheduler.shutdown(wait=False)
        stop_msg = "🛑 AI Trading System Stopped."
        logger.info(stop_msg)
        self.telegram.send_message(stop_msg)


def handle_sigint(signum, frame):
    """Handle Ctrl+C gracefully."""
    logger.info("Interrupt signal received. Shutting down...")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)
    
    system = TradingSystem()
    try:
        system.start()
        # Keep main thread alive
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        system.stop()
        sys.exit(0)
