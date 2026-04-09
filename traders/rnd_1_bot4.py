from dataclasses import dataclass, field
import json
from math import ceil, floor
from typing import Dict, List, Optional

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
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

FAIR_VALUE = 10000
POSITION_LIMIT = 20
BASE_HALF_SPREAD = 9
MIN_EDGE = 2
QUOTE_SIZE = 9
INVENTORY_SKEW_PER_UNIT = 0.16
IMBALANCE_ADJUSTMENT = 4
MAX_IMBALANCE_SHIFT = 2
QUOTE_ADJUSTMENT = 2

TOMATOES_POSITION_LIMIT = 30
TOMATOES_QUOTE_SIZE = 10
TOMATOES_BASE_HALF_SPREAD = 11
TOMATOES_MIN_EDGE = 5
TOMATOES_INVENTORY_SKEW_PER_UNIT = 0.15
TOMATOES_FAIR_ALPHA = 0.18
TOMATOES_TREND_WEIGHT = 0.25
TOMATOES_MAX_TREND_SHIFT = 2.2
TOMATOES_IMBALANCE_ADJUSTMENT = 0.25
TOMATOES_MAX_IMBALANCE_SHIFT = 0.4
TOMATOES_HISTORY_LENGTH = 6


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def compute_order_book_imbalance(bid_volume: Optional[int], ask_volume: Optional[int]) -> float:
    if bid_volume is None or ask_volume is None:
        return 0.0

    total = bid_volume + ask_volume
    if total <= 0:
        return 0.0

    return clamp((bid_volume - ask_volume) / total, -1.0, 1.0)


class Product:
    def __init__(
        self,
        symbol: str,
        state: TradingState,
        trader_state: Dict[str, str],
        position_limit: int,
        quote_size: int,
    ) -> None:
        self.symbol = symbol
        self.state = state
        self.trader_state = trader_state
        self.position_limit = position_limit
        self.quote_size = quote_size
        self.position = state.position.get(symbol, 0)
        self.order_depth = state.order_depths.get(symbol)
        self.orders: List[Order] = []

    def best_bid_ask(self) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        if self.order_depth is None:
            return None, None, None, None

        best_bid = max(self.order_depth.buy_orders) if self.order_depth.buy_orders else None
        best_ask = min(self.order_depth.sell_orders) if self.order_depth.sell_orders else None
        bid_volume = self.order_depth.buy_orders.get(best_bid) if best_bid is not None else None
        ask_volume = abs(self.order_depth.sell_orders.get(best_ask)) if best_ask is not None else None
        return best_bid, bid_volume, best_ask, ask_volume

    def quote_sizes(self) -> tuple[int, int]:
        usable_limit = int(self.position_limit)
        buy_capacity = max(0, self.position_limit - self.position)
        sell_capacity = max(0, self.position_limit + self.position)

        buy_size = min(self.quote_size, buy_capacity)
        sell_size = min(self.quote_size, sell_capacity)

        if self.position >= usable_limit:
            buy_size = 0
        elif self.position > 0:
            buy_size = min(buy_size, max(1, self.quote_size - self.position // 5))

        if self.position <= -usable_limit:
            sell_size = 0
        elif self.position < 0:
            sell_size = min(sell_size, max(1, self.quote_size - abs(self.position) // 5))

        return buy_size, sell_size

    def build_orders(self) -> List[Order]:
        raise NotImplementedError


class StaticProduct(Product):
    def __init__(self, state: TradingState, trader_state: Dict[str, str]) -> None:
        super().__init__(EMERALDS, state, trader_state, POSITION_LIMIT, QUOTE_SIZE)

    def fair_value(
        self,
        bid_price: Optional[int],
        ask_price: Optional[int],
        bid_volume: Optional[int],
        ask_volume: Optional[int],
    ) -> float:
        fair_value = float(FAIR_VALUE)

        imbalance = compute_order_book_imbalance(bid_volume, ask_volume)
        fair_value += clamp(
            imbalance * IMBALANCE_ADJUSTMENT,
            -MAX_IMBALANCE_SHIFT,
            MAX_IMBALANCE_SHIFT,
        )

        if bid_price is not None and ask_price is not None:
            mid_price = (bid_price + ask_price) / 2
            fair_value = 0.8 * fair_value + 0.2 * mid_price

        return fair_value

    def make_quotes(
        self,
        fair_value: float,
        best_bid: Optional[int],
        best_ask: Optional[int],
    ) -> tuple[Optional[int], Optional[int]]:
        inventory_shift = self.position * INVENTORY_SKEW_PER_UNIT
        reservation_price = fair_value - inventory_shift

        bid_quote = floor(reservation_price - BASE_HALF_SPREAD)
        ask_quote = ceil(reservation_price + BASE_HALF_SPREAD)

        if best_bid is not None:
            bid_quote = max(bid_quote, best_bid + QUOTE_ADJUSTMENT)
        if best_ask is not None:
            ask_quote = min(ask_quote, best_ask - QUOTE_ADJUSTMENT)

        if best_ask is not None:
            bid_quote = min(bid_quote, best_ask - MIN_EDGE)
        if best_bid is not None:
            ask_quote = max(ask_quote, best_bid + MIN_EDGE)

        if bid_quote >= ask_quote:
            center = round(reservation_price)
            bid_quote = center - 1
            ask_quote = center + 1

        return bid_quote, ask_quote

    def build_orders(self) -> List[Order]:
        best_bid, bid_volume, best_ask, ask_volume = self.best_bid_ask()
        fair_value = self.fair_value(best_bid, best_ask, bid_volume, ask_volume)
        bid_quote, ask_quote = self.make_quotes(fair_value, best_bid, best_ask)
        buy_size, sell_size = self.quote_sizes()

        if buy_size > 0 and bid_quote is not None:
            self.orders.append(Order(self.symbol, bid_quote, buy_size))
        if sell_size > 0 and ask_quote is not None:
            self.orders.append(Order(self.symbol, ask_quote, -sell_size))

        return self.orders


class DynamicProduct(Product):
    def __init__(self, symbol: str, state: TradingState, trader_state: Dict[str, str]) -> None:
        super().__init__(symbol, state, trader_state, TOMATOES_POSITION_LIMIT, TOMATOES_QUOTE_SIZE)

    def parse_history(self) -> List[float]:
        raw_history = self.trader_state.get("tomatoes", "")
        if not raw_history:
            return []

        prices: List[float] = []
        for value in raw_history.split(","):
            try:
                prices.append(float(value))
            except ValueError:
                continue
        return prices

    def save_history(self, history: List[float]) -> None:
        self.trader_state["tomatoes"] = ",".join(f"{price:.1f}" for price in history)

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
            if len(history) >= 6:
                previous = sum(history[-6:-3]) / 3
            else:
                previous = sum(history[:-3]) / max(1, len(history) - 3)
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
        best_bid: Optional[int],
        best_ask: Optional[int],
    ) -> tuple[Optional[int], Optional[int]]:
        inventory_shift = self.position * TOMATOES_INVENTORY_SKEW_PER_UNIT
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

    def build_orders(self) -> List[Order]:
        best_bid, bid_volume, best_ask, ask_volume = self.best_bid_ask()
        history = self.parse_history()

        if best_bid is not None and best_ask is not None and best_bid < best_ask:
            history.append((best_bid + best_ask) / 2)
        history = history[-TOMATOES_HISTORY_LENGTH:]
        self.save_history(history)

        if not history:
            return self.orders

        fair_value = self.fair_value(history, bid_volume, ask_volume)
        bid_quote, ask_quote = self.make_quotes(fair_value, best_bid, best_ask)
        buy_size, sell_size = self.quote_sizes()

        if buy_size > 0 and bid_quote is not None:
            self.orders.append(Order(self.symbol, bid_quote, buy_size))
        if sell_size > 0 and ask_quote is not None:
            self.orders.append(Order(self.symbol, ask_quote, -sell_size))

        return self.orders


class Trader:
    @staticmethod
    def decode_trader_data(trader_data: str) -> Dict[str, str]:
        if not trader_data:
            return {}

        try:
            decoded = json.loads(trader_data)
            return decoded if isinstance(decoded, dict) else {}
        except json.JSONDecodeError:
            decoded: Dict[str, str] = {}
            for segment in trader_data.split(";"):
                if "=" not in segment:
                    continue
                key, value = segment.split("=", 1)
                if key:
                    decoded[key] = value
            return decoded

    @staticmethod
    def encode_trader_data(trader_state: Dict[str, str]) -> str:
        try:
            return json.dumps(trader_state)
        except (TypeError, ValueError):
            return ""

    @staticmethod
    def first_available_symbol(
        order_depths: Dict[str, OrderDepth],
        candidates: tuple[str, ...],
    ) -> Optional[str]:
        for symbol in candidates:
            if symbol in order_depths:
                return symbol
        return None

    def run(self, state: TradingState):
        trader_state = self.decode_trader_data(getattr(state, "traderData", ""))
        orders: Dict[str, List[Order]] = {}

        if EMERALDS in state.order_depths:
            orders[EMERALDS] = StaticProduct(state, trader_state).build_orders()

        tomato_symbol = self.first_available_symbol(state.order_depths, TOMATOES_ALIASES)
        if tomato_symbol is not None:
            orders[tomato_symbol] = DynamicProduct(tomato_symbol, state, trader_state).build_orders()

        return orders, 0, self.encode_trader_data(trader_state)
