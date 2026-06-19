"""
AI Trading System IDX - Broker API Abstraction

Abstract base class for broker integrations.
All brokers (paper + live) must implement this interface.

Usage:
    class MyBroker(BrokerAPI):
        def place_order(self, order): ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


@dataclass
class Order:
    """Represents a trading order."""
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: float = 0.0  # For limit orders
    stop_loss: float = 0.0
    take_profit: float = 0.0


@dataclass
class OrderResult:
    """Result of an order execution."""
    order_id: str
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    filled_price: float
    status: OrderStatus
    message: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class BrokerPosition:
    """A position as reported by the broker."""
    ticker: str
    quantity: int
    avg_price: float
    current_price: float
    unrealized_pnl: float


class BrokerAPI(ABC):
    """
    Abstract broker API interface.
    All broker implementations must inherit from this class.
    """

    @abstractmethod
    def place_order(self, order: Order) -> OrderResult:
        """
        Place a trading order.

        Args:
            order: Order to execute

        Returns:
            OrderResult with execution details
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: ID of the order to cancel

        Returns:
            True if cancelled successfully
        """
        pass

    @abstractmethod
    def get_positions(self) -> List[BrokerPosition]:
        """
        Get all current positions.

        Returns:
            List of positions
        """
        pass

    @abstractmethod
    def get_balance(self) -> float:
        """
        Get current account balance (cash).

        Returns:
            Available cash balance
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Check status of an existing order.

        Args:
            order_id: Order ID to check

        Returns:
            Current order status
        """
        pass
