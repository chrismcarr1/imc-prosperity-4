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


EMERALDS = "EMERALDS"
TOMATOES = "TOMATOES"
TOMATOES_ALIASES = (TOMATOES, "TOMATOE")
PRODUCTS = (EMERALDS, *TOMATOES_ALIASES)

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
QUOTE_ADJUSTMENT = 2

# Tomatoes market-making parameters.
TOMATOES_POSITION_LIMIT = 20
TOMATOES_QUOTE_SIZE = 5
TOMATOES_BASE_HALF_SPREAD = 3
TOMATOES_MIN_EDGE = 1
TOMATOES_INVENTORY_SKEW_PER_UNIT = 0.10
TOMATOES_FAIR_ALPHA = 0.18
TOMATOES_TREND_WEIGHT = 0.25
TOMATOES_MAX_TREND_SHIFT = 2.0
TOMATOES_IMBALANCE_ADJUSTMENT = 0.35
TOMATOES_MAX_IMBALANCE_SHIFT = 0.5
TOMATOES_HISTORY_LENGTH = 8

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
        orders: Dict[str, List[Order]] = {product: [] for product in PRODUCTS if product in state.order_depths}
        trader_state = self._decode_trader_data(state.traderData)

        order_depth = state.order_depths.get(EMERALDS)
        if order_depth is not None:
            position = state.position.get(EMERALDS, 0)
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
                orders.setdefault(EMERALDS, []).append(Order(EMERALDS, bid_quote, buy_size))
            if sell_size > 0 and ask_quote is not None:
                orders.setdefault(EMERALDS, []).append(Order(EMERALDS, ask_quote, -sell_size))

        tomato_symbol = self._first_available_product(state.order_depths, TOMATOES_ALIASES)
        tomato_depth = state.order_depths.get(tomato_symbol) if tomato_symbol else None
        if tomato_depth is not None:
            tomato_orders, tomato_data = self._trade_tomatoes(
                product=tomato_symbol,
                order_depth=tomato_depth,
                position=state.position.get(tomato_symbol, 0),
                trader_data=trader_state.get("tomatoes", ""),
            )
            orders.setdefault(tomato_symbol, []).extend(tomato_orders)
            trader_state["tomatoes"] = tomato_data

        trader_data = self._encode_trader_data(trader_state)
        return orders, 0, trader_data

    def _trade_tomatoes(self,product: str, order_depth: OrderDepth, position: int, trader_data: str):
        orders: List[Order] = []
        best_bid, bid_volume, best_ask, ask_volume = self._best_bid_ask(order_depth)
        history = self._parse_price_history(trader_data)

        if best_bid is not None and best_ask is not None and best_bid < best_ask:
            history.append((best_bid + best_ask) / 2)
        history = history[-TOMATOES_HISTORY_LENGTH:]

        if not history:
            return orders, ""

        fair_value = self._tomatoes_fair_value(
            history=history,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
        )
        bid_quote, ask_quote = self._make_tomatoes_quotes(
            fair_value=fair_value,
            position=position,
            best_bid=best_bid,
            best_ask=best_ask,
        )
        buy_size, sell_size = self._quote_sizes_with_params(
            position=position,
            position_limit=TOMATOES_POSITION_LIMIT,
            quote_size=TOMATOES_QUOTE_SIZE,
        )

        if buy_size > 0 and bid_quote is not None:
            orders.append(Order(product, bid_quote, buy_size))
        if sell_size > 0 and ask_quote is not None:
            orders.append(Order(product, ask_quote, -sell_size))

        history_str = ",".join(f"{price:.1f}" for price in history)
        return orders, history_str

    def _best_bid_ask(self, order_depth: OrderDepth):
        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None

        bid_volume = (order_depth.buy_orders.get(best_bid) if best_bid is not None else None)
        ask_volume = (abs(order_depth.sell_orders.get(best_ask)) if best_ask is not None else None)

        return best_bid, bid_volume, best_ask, ask_volume

    def _fair_value(self, bid_price, ask_price, bid_volume, ask_volume):

        fair_value = float(FAIR_VALUE)

        imbalance = compute_order_book_imbalance(bid_volume, ask_volume)
        imbalance_shift = clamp(imbalance * IMBALANCE_ADJUSTMENT, -MAX_IMBALANCE_SHIFT, MAX_IMBALANCE_SHIFT)

        fair_value += imbalance_shift

        if bid_price is not None and ask_price is not None:
            mid_price = (bid_price + ask_price) / 2
            fair_value = 0.8 * fair_value + 0.2 * mid_price

        return fair_value

    def _make_quotes(self, fair_value, position, best_bid, best_ask):
        """Inventory shifts the reservation price: long inventory lowers quotes to encourage selling, short 
        inventory raises quotes to encourage buying."""

        inventory_shift = position * INVENTORY_SKEW_PER_UNIT
        reservation_price = fair_value - inventory_shift

        bid_quote = floor(reservation_price - BASE_HALF_SPREAD)
        ask_quote = ceil(reservation_price + BASE_HALF_SPREAD)

        if best_bid is not None:
            candidate = best_bid + QUOTE_ADJUSTMENT
            bid_quote = max(bid_quote, candidate)

        if best_ask is not None:
            candidate = best_ask - QUOTE_ADJUSTMENT
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

    def _quote_sizes(self, position):
        return self._quote_sizes_with_params(position, POSITION_LIMIT, QUOTE_SIZE)

    def _quote_sizes_with_params(self, position, position_limit, quote_size):

        usable_limit = int(position_limit * MAX_POSITION_UTILIZATION)
        buy_capacity = max(0, position_limit - position)
        sell_capacity = max(0, position_limit + position)

        buy_size = min(quote_size, buy_capacity)
        sell_size = min(quote_size, sell_capacity)

        # Fade the quote on the side that would worsen inventory once usage gets high.
        if position >= usable_limit:
            buy_size = 0
        elif position > 0:
            buy_size = min(buy_size, max(1, quote_size - position // 5))

        if position <= -usable_limit:
            sell_size = 0
        elif position < 0:
            sell_size = min(sell_size, max(1, quote_size - abs(position) // 5))

        return buy_size, sell_size

    def _decode_trader_data(self, trader_data: str) -> Dict[str, str]:
        decoded: Dict[str, str] = {}
        if not trader_data:
            return decoded

        for segment in trader_data.split(";"):
            if "=" not in segment:
                continue
            key, value = segment.split("=", 1)
            if key:
                decoded[key] = value
        return decoded

    def _encode_trader_data(self, trader_state: Dict[str, str]) -> str:
        return ";".join(
            f"{key}={value}" for key, value in trader_state.items() if value
        )

    def _parse_price_history(self, trader_data: str) -> List[float]:
        if not trader_data:
            return []

        prices: List[float] = []
        for value in trader_data.split(","):
            try:
                prices.append(float(value))
            except ValueError:
                continue
        return prices

    def _tomatoes_fair_value(self, history, bid_volume, ask_volume):
        ema = history[0]
        for price in history[1:]:
            ema = (1 - TOMATOES_FAIR_ALPHA) * ema + TOMATOES_FAIR_ALPHA * price

        trend_shift = 0.0
        if len(history) >= 4:
            recent = sum(history[-3:]) / 3
            previous = sum(history[-6:-3]) / 3 if len(history) >= 6 else sum(history[:-3]) / max(1, len(history) - 3)
            trend_shift = clamp(
                (recent - previous) * TOMATOES_TREND_WEIGHT,
                -TOMATOES_MAX_TREND_SHIFT,
                TOMATOES_MAX_TREND_SHIFT,
            )

        imbalance = compute_order_book_imbalance(bid_volume, ask_volume)
        imbalance_shift = clamp(
            imbalance * TOMATOES_IMBALANCE_ADJUSTMENT,
            -TOMATOES_MAX_IMBALANCE_SHIFT,
            TOMATOES_MAX_IMBALANCE_SHIFT,
        )
        return ema + trend_shift + imbalance_shift

    def _make_tomatoes_quotes(self, fair_value, position, best_bid, best_ask):
        inventory_shift = position * TOMATOES_INVENTORY_SKEW_PER_UNIT
        reservation_price = fair_value - inventory_shift

        bid_quote = floor(reservation_price - TOMATOES_BASE_HALF_SPREAD)
        ask_quote = ceil(reservation_price + TOMATOES_BASE_HALF_SPREAD)

        if best_bid is not None:
            bid_quote = max(bid_quote, best_bid + 1)
        if best_ask is not None:
            ask_quote = min(ask_quote, best_ask - 1)

        if best_ask is not None:
            bid_quote = min(bid_quote, best_ask - TOMATOES_MIN_EDGE)
        if best_bid is not None:
            ask_quote = max(ask_quote, best_bid + TOMATOES_MIN_EDGE)

        if bid_quote >= ask_quote:
            center = round(reservation_price)
            bid_quote = center - 1
            ask_quote = center + 1
            if best_ask is not None:
                bid_quote = min(bid_quote, best_ask - TOMATOES_MIN_EDGE)
            if best_bid is not None:
                ask_quote = max(ask_quote, best_bid + TOMATOES_MIN_EDGE)

        return bid_quote, ask_quote

    def _first_available_product(self, order_depths: Dict[str, OrderDepth], candidates: Tuple[str, ...]):
        for product in candidates:
            if product in order_depths:
                return product
        return None