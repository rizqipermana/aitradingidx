"""
AI Trading System IDX - Live Broker Template

Template for live broker integration.
Extend this class and implement the methods for your specific broker.

IMPORTANT: This is a TEMPLATE. You must implement the actual API calls
for your broker (e.g., Mirae Asset, Mandiri Sekuritas, etc.).

Usage:
    class MiraeAssetBroker(LiveBroker):
        def _execute_api_call(self, endpoint, data):
            # Your broker-specific implementation
            ...
"""

import time
from typing import List, Optional

from broker.base import (
    BrokerAPI,
    BrokerPosition,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class LiveBroker(BrokerAPI):
    """
    Live broker integration template.

    Subclass this and implement _execute_api_call() for your broker.
    Includes retry logic and error handling.
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = "",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Args:
            api_key: Broker API key
            api_secret: Broker API secret
            base_url: Broker API base URL
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Delay between retries (seconds)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        logger.info(f"LiveBroker initialized (base_url={base_url})")
        logger.warning(
            "⚠️ LIVE BROKER MODE - Real money will be used! "
            "Ensure you have tested thoroughly with paper trading first."
        )

    def _execute_api_call(self, endpoint: str, data: dict) -> dict:
        """
        Execute an API call to the broker.
        Override this method in your broker-specific subclass.

        Args:
            endpoint: API endpoint
            data: Request payload

        Returns:
            API response as dictionary

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError(
            "You must implement _execute_api_call() for your specific broker. "
            "See the docstring for details."
        )

    def _retry_api_call(self, endpoint: str, data: dict) -> dict:
        """Execute API call with retry logic."""
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._execute_api_call(endpoint, data)
                return response
            except Exception as e:
                last_error = e
                logger.warning(
                    f"API call failed (attempt {attempt}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries:
                    sleep_time = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                    time.sleep(sleep_time)

        raise last_error

    def place_order(self, order: Order) -> OrderResult:
        """Place a live order through the broker API."""
        try:
            data = {
                "ticker": order.ticker,
                "side": order.side.value,
                "type": order.order_type.value,
                "quantity": order.quantity,
                "price": order.price,
            }

            response = self._retry_api_call("/orders", data)

            return OrderResult(
                order_id=response.get("order_id", ""),
                ticker=order.ticker,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                filled_price=response.get("filled_price", order.price),
                status=OrderStatus.FILLED,
                message=response.get("message", "Order executed"),
            )

        except Exception as e:
            logger.error(f"Live order failed [{order.ticker}]: {e}")
            return OrderResult(
                order_id="",
                ticker=order.ticker,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                filled_price=0,
                status=OrderStatus.FAILED,
                message=f"Execution error: {str(e)}",
            )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        try:
            self._retry_api_call(f"/orders/{order_id}/cancel", {})
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Cancel order failed: {e}")
            return False

    def get_positions(self) -> List[BrokerPosition]:
        """Get all current positions from broker."""
        try:
            response = self._retry_api_call("/positions", {})
            positions = []
            for pos_data in response.get("positions", []):
                positions.append(
                    BrokerPosition(
                        ticker=pos_data["ticker"],
                        quantity=pos_data["quantity"],
                        avg_price=pos_data["avg_price"],
                        current_price=pos_data["current_price"],
                        unrealized_pnl=pos_data.get("unrealized_pnl", 0),
                    )
                )
            return positions
        except Exception as e:
            logger.error(f"Get positions failed: {e}")
            return []

    def get_balance(self) -> float:
        """Get current account balance."""
        try:
            response = self._retry_api_call("/balance", {})
            return float(response.get("balance", 0))
        except Exception as e:
            logger.error(f"Get balance failed: {e}")
            return 0.0

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Check order status."""
        try:
            response = self._retry_api_call(f"/orders/{order_id}", {})
            status_str = response.get("status", "PENDING")
            return OrderStatus(status_str)
        except Exception as e:
            logger.error(f"Get order status failed: {e}")
            return OrderStatus.FAILED
