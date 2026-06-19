"""
AI Trading System IDX - Database Manager

Handles database connections, CRUD operations, and auto-table creation.
Supports SQLite (default) and PostgreSQL.

Usage:
    from database.db_manager import DatabaseManager
    db = DatabaseManager()
    db.save_trade(trade_data)
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sqlalchemy import create_engine, desc
from sqlalchemy.orm import Session, sessionmaker

from config import settings
from database.models import (
    Base,
    ExecutionLog,
    PerformanceMetric,
    PortfolioSnapshot,
    Signal,
    Trade,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """Manages all database operations for the trading system."""

    def __init__(self, database_url: Optional[str] = None):
        """
        Args:
            database_url: Database connection URL (default from config)
        """
        self.database_url = database_url or settings.DATABASE_URL

        # Create data directory for SQLite
        if self.database_url.startswith("sqlite"):
            db_path = self.database_url.replace("sqlite:///", "")
            db_dir = os.path.dirname(db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

        self.engine = create_engine(
            self.database_url,
            echo=False,
            pool_pre_ping=True,
        )
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Auto-create all tables
        Base.metadata.create_all(self.engine)

        logger.info(f"DatabaseManager initialized: {self.database_url}")

    def _get_session(self) -> Session:
        """Create a new database session."""
        return self.SessionLocal()

    # ---- Trade Operations ----

    def save_trade(self, trade_data: dict) -> Optional[int]:
        """Save a new trade record."""
        session = self._get_session()
        try:
            trade = Trade(**trade_data)
            session.add(trade)
            session.commit()
            trade_id = trade.id
            logger.debug(f"Trade saved: {trade}")
            return trade_id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save trade: {e}")
            return None
        finally:
            session.close()

    def update_trade(self, trade_id: int, update_data: dict) -> bool:
        """Update an existing trade."""
        session = self._get_session()
        try:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if trade:
                for key, value in update_data.items():
                    setattr(trade, key, value)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update trade {trade_id}: {e}")
            return False
        finally:
            session.close()

    def close_trade(
        self,
        ticker: str,
        exit_price: float,
        profit_loss: float,
        profit_loss_pct: float,
        close_reason: str,
    ) -> bool:
        """Close an open trade by ticker."""
        session = self._get_session()
        try:
            trade = (
                session.query(Trade)
                .filter(Trade.ticker == ticker, Trade.status == "OPEN")
                .first()
            )
            if trade:
                trade.exit_price = exit_price
                trade.profit_loss = profit_loss
                trade.profit_loss_pct = profit_loss_pct
                trade.status = "CLOSED"
                trade.close_reason = close_reason
                trade.exit_time = datetime.now()
                session.commit()
                logger.debug(f"Trade closed: {trade}")
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to close trade for {ticker}: {e}")
            return False
        finally:
            session.close()

    def get_open_trades(self) -> List[dict]:
        """Get all open trades."""
        session = self._get_session()
        try:
            trades = session.query(Trade).filter(Trade.status == "OPEN").all()
            return [
                {
                    "id": t.id,
                    "ticker": t.ticker,
                    "action": t.action,
                    "entry_price": t.entry_price,
                    "quantity": t.quantity,
                    "stop_loss": t.stop_loss,
                    "take_profit": t.take_profit,
                    "entry_time": t.entry_time,
                }
                for t in trades
            ]
        finally:
            session.close()

    def get_trade_history(self, limit: int = 100) -> List[dict]:
        """Get trade history (most recent first)."""
        session = self._get_session()
        try:
            trades = (
                session.query(Trade)
                .order_by(desc(Trade.created_at))
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": t.id,
                    "ticker": t.ticker,
                    "action": t.action,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "profit_loss": t.profit_loss,
                    "profit_loss_pct": t.profit_loss_pct,
                    "status": t.status,
                    "close_reason": t.close_reason,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                }
                for t in trades
            ]
        finally:
            session.close()

    # ---- Signal Operations ----

    def save_signal(self, signal_data: dict) -> Optional[int]:
        """Save a generated signal."""
        session = self._get_session()
        try:
            signal = Signal(**signal_data)
            session.add(signal)
            session.commit()
            return signal.id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save signal: {e}")
            return None
        finally:
            session.close()

    def get_recent_signals(self, limit: int = 50) -> List[dict]:
        """Get recent signals."""
        session = self._get_session()
        try:
            signals = (
                session.query(Signal)
                .order_by(desc(Signal.created_at))
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": s.id,
                    "ticker": s.ticker,
                    "action": s.action,
                    "confidence": s.confidence,
                    "ai_probability": s.ai_probability,
                    "entry_price": s.entry_price,
                    "stop_loss": s.stop_loss,
                    "take_profit": s.take_profit,
                    "reason": s.reason,
                    "executed": s.executed,
                    "created_at": s.created_at,
                }
                for s in signals
            ]
        finally:
            session.close()

    # ---- Portfolio Snapshot Operations ----

    def save_portfolio_snapshot(self, snapshot_data: dict) -> Optional[int]:
        """Save a portfolio snapshot."""
        session = self._get_session()
        try:
            snapshot = PortfolioSnapshot(**snapshot_data)
            session.add(snapshot)
            session.commit()
            return snapshot.id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save portfolio snapshot: {e}")
            return None
        finally:
            session.close()

    def get_equity_curve(self, days: int = 30) -> List[dict]:
        """Get equity curve data for the last N days."""
        session = self._get_session()
        try:
            since = datetime.now() - timedelta(days=days)
            snapshots = (
                session.query(PortfolioSnapshot)
                .filter(PortfolioSnapshot.snapshot_time >= since)
                .order_by(PortfolioSnapshot.snapshot_time)
                .all()
            )
            return [
                {
                    "timestamp": s.snapshot_time,
                    "total_equity": s.total_equity,
                    "cash_balance": s.cash_balance,
                    "total_pnl": s.total_pnl,
                    "total_pnl_pct": s.total_pnl_pct,
                }
                for s in snapshots
            ]
        finally:
            session.close()

    # ---- Execution Log Operations ----

    def save_execution_log(self, log_data: dict) -> Optional[int]:
        """Save an execution log entry."""
        session = self._get_session()
        try:
            log = ExecutionLog(**log_data)
            session.add(log)
            session.commit()
            return log.id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save execution log: {e}")
            return None
        finally:
            session.close()

    # ---- Performance Metrics ----

    def save_performance_metric(
        self, metric_name: str, metric_value: float, period: str = "daily"
    ) -> None:
        """Save a performance metric."""
        session = self._get_session()
        try:
            metric = PerformanceMetric(
                metric_name=metric_name,
                metric_value=metric_value,
                period=period,
            )
            session.add(metric)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save metric: {e}")
        finally:
            session.close()

    def get_latest_metrics(self) -> Dict[str, float]:
        """Get the most recent value for each metric."""
        session = self._get_session()
        try:
            from sqlalchemy import func

            # Subquery to get latest entry for each metric
            subquery = (
                session.query(
                    PerformanceMetric.metric_name,
                    func.max(PerformanceMetric.calculated_at).label("max_time"),
                )
                .group_by(PerformanceMetric.metric_name)
                .subquery()
            )

            results = (
                session.query(PerformanceMetric)
                .join(
                    subquery,
                    (PerformanceMetric.metric_name == subquery.c.metric_name)
                    & (PerformanceMetric.calculated_at == subquery.c.max_time),
                )
                .all()
            )

            return {r.metric_name: r.metric_value for r in results}
        finally:
            session.close()
