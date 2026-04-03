from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, floor
from typing import Dict, List, Optional, Tuple

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    # Fallbacks keep the file importable in local notebooks or tests where the
    # competition datamodel module is not present.
    @dataclass
    class Order:
        symbol: str
        price: int
        quantity: int

    @dataclass
    class OrderDepth:
        buy_orders: Dict[int, int] = field(default_factory=dict)
        sell_orders: Dict[int, int] = field(default_factory=dict)

    @dataclass
    class TradingState:
        order_depths: Dict[str, OrderDepth]
        position: Dict[str, int] = field(default_factory=dict)
        traderData: str = ""


PRODUCT = "EMERALDS"

# Core market-making parameters.
FAIR_VALUE = 10_000
POSITION_LIMIT = 20
BASE_HALF_SPREAD = 2
MIN_EDGE = 1
QUOTE_SIZE = 10
MAX_POSITION_UTILIZATION = 1.25

# Inventory and microstructure adjustments.
INVENTORY_SKEW_PER_UNIT = 0.12
IMBALANCE_ADJUSTMENT = 1.0
MAX_IMBALANCE_SHIFT = 1
JOIN_BEST_QUOTES = False
IMPROVE_BY_ONE_TICK = True

# Listens to order book imbalance, but only between the lower and the minimum of upper/value
def clamp(value: float, lower: float, upper: float) -> float:
    """takes in the value (imbalance x IMBALANCE_ADJUSTMENT), the lower (-MAX_IMBALANCE_SHIFT),
    and the upper (MAX_IMBALANCE_SHIFT)"""
    return max(lower, min(upper, value))    # x, but capped between upper and value


# Function to calculate the actual imbalance between bids and asks
# Takes in the bid volume and ask volume
def compute_order_book_imbalance(bid_volume: Optional[int], ask_volume: Optional[int]) -> float:
    """Return imbalance in [-1, 1]. 1 means all buyers, -1 means all sellers,
    and 0 means a balanced order book"""
    if bid_volume is None or ask_volume is None:    # incomplete info so we assume balance
        return 0.0

    total = bid_volume + ask_volume    # This is the denominator
    if total <= 0:
        return 0.0

    return clamp((bid_volume - ask_volume) / total, -1.0, 1.0)    # again uses clamp like we said


class Trader:
    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {PRODUCT: []}

        order_depth = state.order_depths.get(PRODUCT)
        if order_depth is None:
            return orders, 0, state.traderData

        position = state.position.get(PRODUCT, 0)
        bid_price, bid_volume, ask_price, ask_volume = self._best_bid_ask(order_depth)

        fair_value = self._fair_value(
            bid_price=bid_price,
            ask_price=ask_price,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
        )
        bid_quote, ask_quote = self._make_quotes(
            fair_value=fair_value,
            position=position,
            best_bid=bid_price,
            best_ask=ask_price,
        )

        buy_size, sell_size = self._quote_sizes(position)

        if buy_size > 0 and bid_quote is not None:
            orders[PRODUCT].append(Order(PRODUCT, bid_quote, buy_size))
        if sell_size > 0 and ask_quote is not None:
            orders[PRODUCT].append(Order(PRODUCT, ask_quote, -sell_size))

        trader_data = (
            f"fv={fair_value:.2f}|pos={position}|bid={bid_quote}|ask={ask_quote}"
        )
        return orders, 0, trader_data

    def _best_bid_ask(
        self, order_depth: OrderDepth
    ) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None

        bid_volume = (
            order_depth.buy_orders.get(best_bid) if best_bid is not None else None
        )
        ask_volume = (
            abs(order_depth.sell_orders.get(best_ask, 0))
            if best_ask is not None
            else None
        )
        return best_bid, bid_volume, best_ask, ask_volume

    def _fair_value(
        self,
        bid_price: Optional[int],
        ask_price: Optional[int],
        bid_volume: Optional[int],
        ask_volume: Optional[int],
    ) -> float:
        """
        Start from the known EMERALDS anchor and add only a small bounded
        microstructure adjustment. This keeps the strategy market-making
        oriented rather than predictive.
        """
        fair_value = float(FAIR_VALUE)

        imbalance = compute_order_book_imbalance(bid_volume, ask_volume)
        imbalance_shift = clamp(
            imbalance * IMBALANCE_ADJUSTMENT,
            -MAX_IMBALANCE_SHIFT,
            MAX_IMBALANCE_SHIFT,
        )
        fair_value += imbalance_shift

        if bid_price is not None and ask_price is not None and bid_price < ask_price:
            mid_price = (bid_price + ask_price) / 2
            fair_value = 0.8 * fair_value + 0.2 * mid_price

        return fair_value

    def _make_quotes(
        self,
        fair_value: float,
        position: int,
        best_bid: Optional[int],
        best_ask: Optional[int],
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Inventory shifts the reservation price:
        long inventory lowers quotes to encourage selling,
        short inventory raises quotes to encourage buying.
        """
        inventory_shift = position * INVENTORY_SKEW_PER_UNIT
        reservation_price = fair_value - inventory_shift

        bid_quote = floor(reservation_price - BASE_HALF_SPREAD)
        ask_quote = ceil(reservation_price + BASE_HALF_SPREAD)

        if best_bid is not None:
            if JOIN_BEST_QUOTES:
                candidate = best_bid + (1 if IMPROVE_BY_ONE_TICK else 0)
                bid_quote = max(bid_quote, candidate)
            else:
                candidate = best_bid + 2
                bid_quote = max(bid_quote, candidate)

        if best_ask is not None:
            if JOIN_BEST_QUOTES:
                candidate = best_ask - (1 if IMPROVE_BY_ONE_TICK else 0)
                ask_quote = min(ask_quote, candidate)
            else:
                candidate = best_ask - 2
                ask_quote = min(ask_quote, candidate)

        if best_ask is not None:
            bid_quote = min(bid_quote, best_ask - MIN_EDGE)
        if best_bid is not None:
            ask_quote = max(ask_quote, best_bid + MIN_EDGE)

        if bid_quote >= ask_quote:
            center = round(reservation_price)
            bid_quote = center - 1
            ask_quote = center + 1

        return bid_quote, ask_quote

    def _quote_sizes(self, position: int) -> Tuple[int, int]:
        usable_limit = int(POSITION_LIMIT * MAX_POSITION_UTILIZATION)
        buy_capacity = max(0, POSITION_LIMIT - position)
        sell_capacity = max(0, POSITION_LIMIT + position)

        buy_size = min(QUOTE_SIZE, buy_capacity)
        sell_size = min(QUOTE_SIZE, sell_capacity)

        # Fade the quote on the side that would worsen inventory once usage gets high.
        if position >= usable_limit:
            buy_size = 0
        elif position > 0:
            buy_size = min(buy_size, max(1, QUOTE_SIZE - position // 5))

        if position <= -usable_limit:
            sell_size = 0
        elif position < 0:
            sell_size = min(sell_size, max(1, QUOTE_SIZE - abs(position) // 5))

        return buy_size, sell_size
