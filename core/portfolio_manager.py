"""
AI Trading System IDX - Portfolio Manager

Tracks all positions, calculates equity, P&L, win rate, drawdown.
Generates equity curve data for dashboard visualization.

Usage:
    from core.portfolio_manager import PortfolioManager
    pm = PortfolioManager(initial_capital=10_000_000)
    pm.open_position(trade_signal)
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from config import settings
from core.risk_manager import PortfolioState
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    """Represents an open or closed trading position."""
    ticker: str
    entry_price: float
    quantity: int
    stop_loss: float
    take_profit: float
    entry_time: datetime
    direction: str = "BUY"
    status: str = "OPEN"  # OPEN, CLOSED, CANCELLED
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    highest_price: float = 0.0  # For trailing stop
    profit_loss: float = 0.0
    profit_loss_pct: float = 0.0

    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.quantity * self.entry_price

    def update_pnl(self, current_price: float) -> None:
        """Update unrealized P&L based on current price."""
        if self.direction == "BUY":
            self.profit_loss = (current_price - self.entry_price) * self.quantity
            self.profit_loss_pct = (current_price / self.entry_price - 1) * 100
        # Track highest price for trailing stop
        if current_price > self.highest_price:
            self.highest_price = current_price

    def close(self, exit_price: float) -> None:
        """Close the position and calculate final P&L."""
        self.exit_price = exit_price
        self.exit_time = datetime.now()
        self.status = "CLOSED"
        if self.direction == "BUY":
            self.profit_loss = (exit_price - self.entry_price) * self.quantity
            self.profit_loss_pct = (exit_price / self.entry_price - 1) * 100


@dataclass
class EquitySnapshot:
    """A snapshot of portfolio equity at a point in time."""
    timestamp: datetime
    total_equity: float
    cash_balance: float
    positions_value: float
    daily_pnl: float
    total_pnl: float
    total_pnl_pct: float


class PortfolioManager:
    """Manages trading portfolio, positions, and performance metrics."""

    def __init__(self, initial_capital: Optional[float] = None):
        """
        Args:
            initial_capital: Starting capital in IDR (default from config)
        """
        self.initial_capital = initial_capital or settings.INITIAL_CAPITAL
        self.cash_balance = self.initial_capital
        self.open_positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.equity_history: List[EquitySnapshot] = []

        # Daily tracking
        self._daily_start_equity = self.initial_capital
        self._daily_pnl = 0.0
        self._current_date = date.today()

        logger.info(
            f"PortfolioManager initialized: "
            f"capital=Rp {self.initial_capital:,.0f}"
        )

    @property
    def total_equity(self) -> float:
        """Total equity = cash + positions market value."""
        positions_value = sum(
            pos.quantity * pos.entry_price for pos in self.open_positions.values()
        )
        return self.cash_balance + positions_value

    @property
    def positions_value(self) -> float:
        """Total value of open positions."""
        return sum(
            pos.quantity * pos.entry_price for pos in self.open_positions.values()
        )

    @property
    def total_pnl(self) -> float:
        """Total profit/loss from initial capital."""
        return self.total_equity - self.initial_capital

    @property
    def total_pnl_pct(self) -> float:
        """Total P&L as percentage."""
        if self.initial_capital == 0:
            return 0.0
        return (self.total_pnl / self.initial_capital) * 100

    @property
    def win_rate(self) -> float:
        """Win rate from closed positions."""
        if not self.closed_positions:
            return 0.0
        wins = sum(1 for p in self.closed_positions if p.profit_loss > 0)
        return (wins / len(self.closed_positions)) * 100

    @property
    def total_trades(self) -> int:
        """Total number of completed trades."""
        return len(self.closed_positions)

    @property
    def open_position_count(self) -> int:
        """Number of currently open positions."""
        return len(self.open_positions)

    @property
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity history."""
        if not self.equity_history:
            return 0.0
        equities = [snap.total_equity for snap in self.equity_history]
        peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _reset_daily_tracking(self) -> None:
        """Reset daily P&L tracking at start of new day."""
        today = date.today()
        if self._current_date != today:
            self._daily_start_equity = self.total_equity
            self._daily_pnl = 0.0
            self._current_date = today

    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state for risk management."""
        self._reset_daily_tracking()
        return PortfolioState(
            total_equity=self.total_equity,
            cash_balance=self.cash_balance,
            open_positions=self.open_position_count,
            daily_pnl=self._daily_pnl,
            daily_pnl_date=self._current_date,
        )

    def open_position(
        self,
        ticker: str,
        entry_price: float,
        quantity: int,
        stop_loss: float,
        take_profit: float,
    ) -> Optional[Position]:
        """
        Open a new position.

        Args:
            ticker: Stock ticker
            entry_price: Entry price per share
            quantity: Number of shares
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Position object or None if insufficient funds
        """
        total_cost = entry_price * quantity

        if total_cost > self.cash_balance:
            logger.warning(
                f"Insufficient funds for {ticker}: "
                f"need Rp {total_cost:,.0f}, have Rp {self.cash_balance:,.0f}"
            )
            return None

        if ticker in self.open_positions:
            logger.warning(f"Position already open for {ticker}")
            return None

        position = Position(
            ticker=ticker,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now(),
            highest_price=entry_price,
        )

        self.cash_balance -= total_cost
        self.open_positions[ticker] = position

        logger.info(
            f"📈 POSITION OPENED [{ticker}]: "
            f"qty={quantity}, price={entry_price:,.0f}, "
            f"SL={stop_loss:,.0f}, TP={take_profit:,.0f}, "
            f"cost=Rp {total_cost:,.0f}"
        )

        return position

    def close_position(
        self, ticker: str, exit_price: float, reason: str = ""
    ) -> Optional[Position]:
        """
        Close an existing position.

        Args:
            ticker: Stock ticker
            exit_price: Exit price per share
            reason: Reason for closing (TP, SL, manual)

        Returns:
            Closed Position or None if not found
        """
        if ticker not in self.open_positions:
            logger.warning(f"No open position for {ticker}")
            return None

        position = self.open_positions.pop(ticker)
        position.close(exit_price)

        # Return capital + P&L
        proceeds = exit_price * position.quantity
        self.cash_balance += proceeds

        # Update daily P&L
        self._daily_pnl += position.profit_loss

        # Add to closed positions
        self.closed_positions.append(position)

        # Take snapshot
        self.take_equity_snapshot()

        emoji = "✅" if position.profit_loss >= 0 else "🛑"
        logger.info(
            f"{emoji} POSITION CLOSED [{ticker}] ({reason}): "
            f"entry={position.entry_price:,.0f}, exit={exit_price:,.0f}, "
            f"P&L=Rp {position.profit_loss:,.0f} ({position.profit_loss_pct:+.1f}%)"
        )

        return position

    def check_stop_loss_take_profit(
        self, current_prices: Dict[str, float]
    ) -> List[Tuple[str, str, float]]:
        """
        Check all open positions for SL/TP triggers.

        Args:
            current_prices: Dictionary of ticker -> current price

        Returns:
            List of (ticker, reason, exit_price) for positions to close
        """
        to_close = []

        for ticker, position in list(self.open_positions.items()):
            current_price = current_prices.get(ticker)
            if current_price is None:
                continue

            # Update unrealized P&L
            position.update_pnl(current_price)

            # Check take profit
            if current_price >= position.take_profit:
                to_close.append((ticker, "TAKE_PROFIT", current_price))
                logger.info(
                    f"🎯 TP HIT [{ticker}]: "
                    f"price={current_price:,.0f} >= TP={position.take_profit:,.0f}"
                )

            # Check stop loss
            elif current_price <= position.stop_loss:
                to_close.append((ticker, "STOP_LOSS", current_price))
                logger.info(
                    f"🛑 SL HIT [{ticker}]: "
                    f"price={current_price:,.0f} <= SL={position.stop_loss:,.0f}"
                )

            # Check trailing stop
            else:
                trailing_sl = position.highest_price - (
                    (position.highest_price - position.stop_loss) * 0.5
                )
                if (
                    position.highest_price > position.entry_price * 1.03
                    and current_price <= trailing_sl
                ):
                    to_close.append((ticker, "TRAILING_STOP", current_price))
                    logger.info(
                        f"📉 TRAILING SL [{ticker}]: "
                        f"price={current_price:,.0f} <= trailing SL={trailing_sl:,.0f}"
                    )

        return to_close

    def take_equity_snapshot(self) -> EquitySnapshot:
        """Record current equity state."""
        snapshot = EquitySnapshot(
            timestamp=datetime.now(),
            total_equity=self.total_equity,
            cash_balance=self.cash_balance,
            positions_value=self.positions_value,
            daily_pnl=self._daily_pnl,
            total_pnl=self.total_pnl,
            total_pnl_pct=self.total_pnl_pct,
        )
        self.equity_history.append(snapshot)
        return snapshot

    def get_summary(self) -> dict:
        """Get portfolio summary for dashboard/notifications."""
        return {
            "initial_capital": self.initial_capital,
            "total_equity": self.total_equity,
            "cash_balance": self.cash_balance,
            "positions_value": self.positions_value,
            "total_pnl": self.total_pnl,
            "total_pnl_pct": self.total_pnl_pct,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "open_positions": self.open_position_count,
            "max_drawdown": self.max_drawdown,
            "daily_pnl": self._daily_pnl,
        }

    def get_open_positions_list(self) -> List[dict]:
        """Get list of open positions as dictionaries."""
        return [
            {
                "ticker": pos.ticker,
                "entry_price": pos.entry_price,
                "quantity": pos.quantity,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "entry_time": pos.entry_time.isoformat(),
                "unrealized_pnl": pos.profit_loss,
                "unrealized_pnl_pct": pos.profit_loss_pct,
            }
            for pos in self.open_positions.values()
        ]

    def get_trade_history(self) -> List[dict]:
        """Get closed trade history as list of dictionaries."""
        return [
            {
                "ticker": pos.ticker,
                "entry_price": pos.entry_price,
                "exit_price": pos.exit_price,
                "quantity": pos.quantity,
                "profit_loss": pos.profit_loss,
                "profit_loss_pct": pos.profit_loss_pct,
                "entry_time": pos.entry_time.isoformat(),
                "exit_time": pos.exit_time.isoformat() if pos.exit_time else None,
                "status": pos.status,
            }
            for pos in self.closed_positions
        ]
