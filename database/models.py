"""
AI Trading System IDX - Database Models

SQLAlchemy ORM models for:
- Trades
- Signals
- Portfolio snapshots
- Execution logs
- Performance metrics

Usage:
    from database.models import Trade, Signal, PortfolioSnapshot
"""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Trade(Base):
    """Records all trading transactions."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    action = Column(String(4), nullable=False)  # BUY / SELL
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, default=0.0)
    quantity = Column(Integer, nullable=False)
    stop_loss = Column(Float, default=0.0)
    take_profit = Column(Float, default=0.0)
    profit_loss = Column(Float, default=0.0)
    profit_loss_pct = Column(Float, default=0.0)
    status = Column(String(10), default="OPEN")  # OPEN / CLOSED / CANCELLED
    close_reason = Column(String(20), default="")  # TAKE_PROFIT / STOP_LOSS / SIGNAL / MANUAL
    entry_time = Column(DateTime, default=datetime.now)
    exit_time = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return (
            f"<Trade(id={self.id}, ticker={self.ticker}, "
            f"action={self.action}, status={self.status})>"
        )


class Signal(Base):
    """Records all generated signals."""
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    action = Column(String(4), nullable=False)  # BUY / SELL / HOLD
    confidence = Column(Float, default=0.0)
    ai_probability = Column(Float, default=0.0)
    entry_price = Column(Float, default=0.0)
    stop_loss = Column(Float, default=0.0)
    take_profit = Column(Float, default=0.0)
    reason = Column(Text, default="")
    executed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return (
            f"<Signal(id={self.id}, ticker={self.ticker}, "
            f"action={self.action}, prob={self.ai_probability:.2f})>"
        )


class PortfolioSnapshot(Base):
    """Periodic snapshots of portfolio state for equity curve."""
    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    total_equity = Column(Float, nullable=False)
    cash_balance = Column(Float, nullable=False)
    positions_value = Column(Float, default=0.0)
    daily_pnl = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    total_pnl_pct = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    total_trades = Column(Integer, default=0)
    open_positions = Column(Integer, default=0)
    max_drawdown = Column(Float, default=0.0)
    snapshot_time = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return (
            f"<PortfolioSnapshot(equity={self.total_equity:,.0f}, "
            f"pnl={self.total_pnl_pct:+.1f}%)>"
        )


class ExecutionLog(Base):
    """Logs all execution attempts and results."""
    __tablename__ = "execution_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(Integer, nullable=True)
    ticker = Column(String(10), nullable=False)
    action = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False)
    message = Column(Text, default="")
    broker_response = Column(Text, default="")
    order_id = Column(String(50), default="")
    created_at = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return (
            f"<ExecutionLog(ticker={self.ticker}, "
            f"action={self.action}, status={self.status})>"
        )


class PerformanceMetric(Base):
    """Stores calculated performance metrics."""
    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(50), nullable=False)
    metric_value = Column(Float, nullable=False)
    period = Column(String(20), default="daily")  # daily / weekly / monthly / all_time
    calculated_at = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return (
            f"<PerformanceMetric(name={self.metric_name}, "
            f"value={self.metric_value})>"
        )
