"""
AI Trading System IDX - Paper Trading Broker

Simulates order execution for paper trading mode.
Uses virtual capital (default: Rp 10.000.000).

Usage:
    from broker.paper_broker import PaperBroker
    broker = PaperBroker(initial_capital=10_000_000)
    result = broker.place_order(order)
"""

import uuid
from datetime import datetime
from typing import Dict, List

from broker.base import (
    BrokerAPI,
    BrokerPosition,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
)
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class PaperBroker(BrokerAPI):
    """Simulated broker for paper trading (no real money)."""

    def __init__(self, initial_capital: float = None):
        """
        Args:
            initial_capital: Starting virtual capital
        """
        self.initial_capital = initial_capital or settings.INITIAL_CAPITAL
        self.cash_balance = self.initial_capital
        self.positions: Dict[str, BrokerPosition] = {}
        self.order_history: List[OrderResult] = []
        self.pending_orders: Dict[str, Order] = {}

        # Simulation settings
        self.slippage_pct = 0.001  # 0.1% slippage simulation
        self.commission_pct = 0.0015  # 0.15% commission (typical IDX)

        logger.info(
            f"📄 PaperBroker initialized: "
            f"capital=Rp {self.initial_capital:,.0f}, "
            f"slippage={self.slippage_pct:.2%}, "
            f"commission={self.commission_pct:.2%}"
        )

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        return f"PAPER-{uuid.uuid4().hex[:8].upper()}"

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply realistic slippage to execution price."""
        if side == OrderSide.BUY:
            return price * (1 + self.slippage_pct)
        else:
            return price * (1 - self.slippage_pct)

    def _calculate_commission(self, total_value: float) -> float:
        """Calculate broker commission."""
        return total_value * self.commission_pct

    def place_order(self, order: Order) -> OrderResult:
        """
        Simulate order execution.

        Args:
            order: Order to execute

        Returns:
            OrderResult with simulated fill
        """
        order_id = self._generate_order_id()

        try:
            # Calculate execution price with slippage
            if order.order_type == OrderType.MARKET:
                filled_price = self._apply_slippage(order.price, order.side)
            else:
                # Limit order: fill at limit price (simplified)
                filled_price = order.price

            total_value = filled_price * order.quantity
            commission = self._calculate_commission(total_value)

            if order.side == OrderSide.BUY:
                # Check sufficient funds
                total_cost = total_value + commission
                if total_cost > self.cash_balance:
                    result = OrderResult(
                        order_id=order_id,
                        ticker=order.ticker,
                        side=order.side,
                        order_type=order.order_type,
                        quantity=order.quantity,
                        filled_price=0,
                        status=OrderStatus.REJECTED,
                        message=f"Insufficient funds: need Rp {total_cost:,.0f}, have Rp {self.cash_balance:,.0f}",
                    )
                    self.order_history.append(result)
                    logger.warning(f"❌ Order rejected [{order.ticker}]: {result.message}")
                    return result

                # Execute BUY
                self.cash_balance -= total_cost

                if order.ticker in self.positions:
                    # Average up existing position
                    pos = self.positions[order.ticker]
                    total_qty = pos.quantity + order.quantity
                    avg_price = (
                        (pos.avg_price * pos.quantity + filled_price * order.quantity)
                        / total_qty
                    )
                    pos.quantity = total_qty
                    pos.avg_price = avg_price
                    pos.current_price = filled_price
                else:
                    # New position
                    self.positions[order.ticker] = BrokerPosition(
                        ticker=order.ticker,
                        quantity=order.quantity,
                        avg_price=filled_price,
                        current_price=filled_price,
                        unrealized_pnl=0,
                    )

                logger.info(
                    f"📄 PAPER BUY [{order.ticker}]: "
                    f"qty={order.quantity}, price={filled_price:,.0f}, "
                    f"commission=Rp {commission:,.0f}, "
                    f"total=Rp {total_cost:,.0f}"
                )

            elif order.side == OrderSide.SELL:
                # Check position exists
                if order.ticker not in self.positions:
                    result = OrderResult(
                        order_id=order_id,
                        ticker=order.ticker,
                        side=order.side,
                        order_type=order.order_type,
                        quantity=order.quantity,
                        filled_price=0,
                        status=OrderStatus.REJECTED,
                        message=f"No position to sell for {order.ticker}",
                    )
                    self.order_history.append(result)
                    logger.warning(f"❌ Order rejected [{order.ticker}]: {result.message}")
                    return result

                pos = self.positions[order.ticker]
                if order.quantity > pos.quantity:
                    order.quantity = pos.quantity  # Sell all available

                # Execute SELL
                proceeds = total_value - commission
                self.cash_balance += proceeds

                pos.quantity -= order.quantity
                if pos.quantity <= 0:
                    del self.positions[order.ticker]

                logger.info(
                    f"📄 PAPER SELL [{order.ticker}]: "
                    f"qty={order.quantity}, price={filled_price:,.0f}, "
                    f"commission=Rp {commission:,.0f}, "
                    f"proceeds=Rp {proceeds:,.0f}"
                )

            # Create success result
            result = OrderResult(
                order_id=order_id,
                ticker=order.ticker,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                filled_price=filled_price,
                status=OrderStatus.FILLED,
                message="Paper trade executed successfully",
            )
            self.order_history.append(result)
            return result

        except Exception as e:
            result = OrderResult(
                order_id=order_id,
                ticker=order.ticker,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                filled_price=0,
                status=OrderStatus.FAILED,
                message=f"Execution error: {str(e)}",
            )
            self.order_history.append(result)
            logger.error(f"❌ Paper order failed [{order.ticker}]: {e}")
            return result

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order (paper mode - always succeeds)."""
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
            logger.info(f"Order {order_id} cancelled")
            return True
        return False

    def get_positions(self) -> List[BrokerPosition]:
        """Get all open positions."""
        return list(self.positions.values())

    def get_balance(self) -> float:
        """Get current cash balance."""
        return self.cash_balance

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Check order status."""
        for result in self.order_history:
            if result.order_id == order_id:
                return result.status
        return OrderStatus.PENDING

    def get_total_equity(self) -> float:
        """Get total equity (cash + positions)."""
        positions_value = sum(
            pos.quantity * pos.current_price for pos in self.positions.values()
        )
        return self.cash_balance + positions_value
