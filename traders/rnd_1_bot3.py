from dataclasses import dataclass, field
from math import ceil, floor
from typing import Dict, List, Optional, Tuple, Type

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


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp value into [lower, upper]."""
    return max(lower, min(upper, value))


def compute_order_book_imbalance(
    bid_volume: Optional[int], ask_volume: Optional[int]
) -> float:
    """Return imbalance in [-1, 1]."""
    if bid_volume is None or ask_volume is None:
        return 0.0

    total = bid_volume + ask_volume
    if total <= 0:
        return 0.0

    return clamp((bid_volume - ask_volume) / total, -1.0, 1.0)


class ProductTrader:
    def __init__(
        self,
        state: TradingState,
        product: str,
        trader_state: Dict[str, str],
    ) -> None:
        self.state = state
        self.product = product
        self.trader_state = trader_state
        self.order_depth = state.order_depths[product]
        self.position = state.position.get(product, 0)
        self.orders: List[Order] = []

    def best_bid_ask(
        self,
    ) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        best_bid = max(self.order_depth.buy_orders) if self.order_depth.buy_orders else None
        best_ask = min(self.order_depth.sell_orders) if self.order_depth.sell_orders else None

        bid_volume = self.order_depth.buy_orders.get(best_bid) if best_bid is not None else None
        ask_volume = (
            abs(self.order_depth.sell_orders.get(best_ask))
            if best_ask is not None
            else None
        )
        return best_bid, bid_volume, best_ask, ask_volume

    def add_buy_order(self, price: Optional[int], quantity: int) -> None:
        if quantity > 0 and price is not None:
            self.orders.append(Order(self.product, price, quantity))

    def add_sell_order(self, price: Optional[int], quantity: int) -> None:
        if quantity > 0 and price is not None:
            self.orders.append(Order(self.product, price, -quantity))

    def quote_sizes_with_params(
        self,
        position: int,
        position_limit: int,
        quote_size: int,
    ) -> Tuple[int, int]:
        usable_limit = int(position_limit * MAX_POSITION_UTILIZATION)
        buy_capacity = max(0, position_limit - position)
        sell_capacity = max(0, position_limit + position)

        buy_size = min(quote_size, buy_capacity)
        sell_size = min(quote_size, sell_capacity)

        if position >= usable_limit:
            buy_size = 0
        elif position > 0:
            buy_size = min(buy_size, max(1, quote_size - position // 5))

        if position <= -usable_limit:
            sell_size = 0
        elif position < 0:
            sell_size = min(sell_size, max(1, quote_size - abs(position) // 5))

        return buy_size, sell_size

    @staticmethod
    def parse_price_history(trader_data: str) -> List[float]:
        if not trader_data:
            return []

        prices: List[float] = []
        for value in trader_data.split(","):
            try:
                prices.append(float(value))
            except ValueError:
                continue
        return prices

    def get_orders(self) -> Tuple[List[Order], Optional[str], Optional[str]]:
        raise NotImplementedError


class StaticTrader(ProductTrader):
    trader_data_key = None

    def fair_value(
        self,
        bid_price: Optional[int],
        ask_price: Optional[int],
        bid_volume: Optional[int],
        ask_volume: Optional[int],
    ) -> float:
        fair_value = float(FAIR_VALUE)

        imbalance = compute_order_book_imbalance(bid_volume, ask_volume)
        imbalance_shift = clamp(
            imbalance * IMBALANCE_ADJUSTMENT,
            -MAX_IMBALANCE_SHIFT,
            MAX_IMBALANCE_SHIFT,
        )
        fair_value += imbalance_shift

        if bid_price is not None and ask_price is not None:
            mid_price = (bid_price + ask_price) / 2
            fair_value = 0.8 * fair_value + 0.2 * mid_price

        return fair_value

    def make_quotes(
        self,
        fair_value: float,
        position: int,
        best_bid: Optional[int],
        best_ask: Optional[int],
    ) -> Tuple[int, int]:
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

    def quote_sizes(self, position: int) -> Tuple[int, int]:
        return self.quote_sizes_with_params(position, POSITION_LIMIT, QUOTE_SIZE)

    def get_orders(self) -> Tuple[List[Order], Optional[str], Optional[str]]:
        bid_price, bid_volume, ask_price, ask_volume = self.best_bid_ask()

        fair_value = self.fair_value(
            bid_price=bid_price,
            ask_price=ask_price,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
        )
        bid_quote, ask_quote = self.make_quotes(
            fair_value=fair_value,
            position=self.position,
            best_bid=bid_price,
            best_ask=ask_price,
        )
        buy_size, sell_size = self.quote_sizes(self.position)

        self.add_buy_order(bid_quote, buy_size)
        self.add_sell_order(ask_quote, sell_size)
        return self.orders, None, None


class DynamicTrader(ProductTrader):
    trader_data_key = "tomatoes"

    def fair_value(
        self,
        history: List[float],
        bid_volume: Optional[int],
        ask_volume: Optional[int],
    ) -> float:
        ema = history[0]
        for price in history[1:]:
            ema = (1 - TOMATOES_FAIR_ALPHA) * ema + TOMATOES_FAIR_ALPHA * price

        trend_shift = 0.0
        if len(history) >= 4:
            recent = sum(history[-3:]) / 3
            previous = (
                sum(history[-6:-3]) / 3
                if len(history) >= 6
                else sum(history[:-3]) / max(1, len(history) - 3)
            )
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

    def make_quotes(
        self,
        fair_value: float,
        position: int,
        best_bid: Optional[int],
        best_ask: Optional[int],
    ) -> Tuple[int, int]:
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

    def get_orders(self) -> Tuple[List[Order], Optional[str], Optional[str]]:
        best_bid, bid_volume, best_ask, ask_volume = self.best_bid_ask()
        history = self.parse_price_history(self.trader_state.get(self.trader_data_key, ""))

        if best_bid is not None and best_ask is not None and best_bid < best_ask:
            history.append((best_bid + best_ask) / 2)
        history = history[-TOMATOES_HISTORY_LENGTH:]

        if not history:
            return self.orders, self.trader_data_key, ""

        fair_value = self.fair_value(
            history=history,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
        )
        bid_quote, ask_quote = self.make_quotes(
            fair_value=fair_value,
            position=self.position,
            best_bid=best_bid,
            best_ask=best_ask,
        )
        buy_size, sell_size = self.quote_sizes_with_params(
            position=self.position,
            position_limit=TOMATOES_POSITION_LIMIT,
            quote_size=TOMATOES_QUOTE_SIZE,
        )

        self.add_buy_order(bid_quote, buy_size)
        self.add_sell_order(ask_quote, sell_size)

        history_str = ",".join(f"{price:.1f}" for price in history)
        return self.orders, self.trader_data_key, history_str


class Trader:
    PRODUCT_TRADERS: Dict[str, Type[ProductTrader]] = {
        EMERALDS: StaticTrader,
        TOMATOES: DynamicTrader,
        "TOMATOE": DynamicTrader,
    }

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {
            product: [] for product in state.order_depths if product in self.PRODUCT_TRADERS
        }
        trader_state = self._decode_trader_data(state.traderData)

        for product, trader_cls in self._available_traders(state).items():
            product_trader = trader_cls(state, product, trader_state)
            orders, trader_data_key, trader_data_value = product_trader.get_orders()
            result.setdefault(product, []).extend(orders)
            if trader_data_key is not None:
                trader_state[trader_data_key] = trader_data_value or ""

        trader_data = self._encode_trader_data(trader_state)
        return result, 0, trader_data

    def _available_traders(
        self, state: TradingState
    ) -> Dict[str, Type[ProductTrader]]:
        available: Dict[str, Type[ProductTrader]] = {}

        if EMERALDS in state.order_depths:
            available[EMERALDS] = self.PRODUCT_TRADERS[EMERALDS]

        tomato_product = self._first_available_product(state.order_depths, TOMATOES_ALIASES)
        if tomato_product is not None:
            available[tomato_product] = self.PRODUCT_TRADERS[tomato_product]

        return available

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
        return ";".join(f"{key}={value}" for key, value in trader_state.items() if value)

    def _first_available_product(
        self,
        order_depths: Dict[str, OrderDepth],
        candidates: Tuple[str, ...],
    ) -> Optional[str]:
        for product in candidates:
            if product in order_depths:
                return product
        return None
