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


ASH_COATED_OSMIUM = "ASH_COATED_OSMIUM"
INTARIAN_PEPPER_ROOT = "INTARIAN_PEPPER_ROOT"
INTARIAN_PEPPER_ROOT_ALIASES = (
    INTARIAN_PEPPER_ROOT,
    "INTARIAN_PEPPER",
    "PEPPER_ROOT",
)

ASH_FAIR_VALUE = 10_000
POSITION_LIMIT = 80

ASH_QUOTE_SIZE = 20
ASH_BASE_HALF_SPREAD = 11
ASH_MIN_EDGE = 2
ASH_INVENTORY_SKEW_PER_UNIT = 0.1
ASH_TAKE_EDGE = .4
ASH_IMBALANCE_ADJUSTMENT = 2.5
ASH_MAX_IMBALANCE_SHIFT = 4
ASH_SIGNAL_SIZE_BOOST = 12
ASH_SECOND_LEVEL_OFFSET = 1

ROOT_QUOTE_SIZE = 18
ROOT_BASE_HALF_SPREAD = 8.5
ROOT_MIN_EDGE = 6
ROOT_INVENTORY_SKEW_PER_UNIT = 0
ROOT_TAKE_EDGE = 1
ROOT_SIGNAL_SIZE_BOOST = 30
ROOT_SECOND_LEVEL_OFFSET = 2
ROOT_HISTORY_LENGTH = 45
ROOT_FAST_ALPHA = 0.3
ROOT_SLOW_ALPHA = 0.06
ROOT_MICRO_ALPHA = 0.15
ROOT_TREND_WEIGHT = 0.95
ROOT_MAX_TREND_SHIFT = 6.0
ROOT_REVERSION_WEIGHT = 0.6
ROOT_MAX_REVERSION_SHIFT = 3


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
        signal_size_boost: int,
        second_level_offset: int,
    ) -> None:
        self.symbol = symbol
        self.state = state
        self.trader_state = trader_state
        self.position_limit = position_limit
        self.quote_size = quote_size
        self.signal_size_boost = signal_size_boost
        self.second_level_offset = second_level_offset
        self.position = state.position.get(symbol, 0)
        self.order_depth = state.order_depths.get(symbol)
        self.orders: List[Order] = []
        self.pending_position = self.position
        self.front_ratio = 0.7

    def best_bid_ask(self) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        if self.order_depth is None:
            return None, None, None, None

        best_bid = max(self.order_depth.buy_orders) if self.order_depth.buy_orders else None
        best_ask = min(self.order_depth.sell_orders) if self.order_depth.sell_orders else None
        bid_volume = self.order_depth.buy_orders.get(best_bid) if best_bid is not None else None
        ask_volume = abs(self.order_depth.sell_orders.get(best_ask)) if best_ask is not None else None
        return best_bid, bid_volume, best_ask, ask_volume

    def ordered_sells(self) -> List[tuple[int, int]]:
        if self.order_depth is None:
            return []
        return [(price, abs(volume)) for price, volume in sorted(self.order_depth.sell_orders.items())]

    def ordered_buys(self) -> List[tuple[int, int]]:
        if self.order_depth is None:
            return []
        return [(price, abs(volume)) for price, volume in sorted(self.order_depth.buy_orders.items(), reverse=True)]

    def remaining_buy_capacity(self) -> int:
        return max(0, self.position_limit - self.pending_position)

    def remaining_sell_capacity(self) -> int:
        return max(0, self.position_limit + self.pending_position)

    def add_buy(self, price: int, volume: int) -> int:
        size = min(max(0, int(volume)), self.remaining_buy_capacity())
        if size > 0:
            self.orders.append(Order(self.symbol, int(price), size))
            self.pending_position += size
        return size

    def add_sell(self, price: int, volume: int) -> int:
        size = min(max(0, int(volume)), self.remaining_sell_capacity())
        if size > 0:
            self.orders.append(Order(self.symbol, int(price), -size))
            self.pending_position -= size
        return size

    def conviction_adjusted_sizes(self, signal_strength: float) -> tuple[int, int]:
        inventory_ratio = 0.0
        if self.position_limit > 0:
            inventory_ratio = abs(self.pending_position) / self.position_limit

        conviction = clamp(signal_strength, 0.0, 1.0)
        boost = round(conviction * self.signal_size_boost)
        base_size = self.quote_size + boost
        base_size = max(1, round(base_size * (1 - 0.45 * inventory_ratio)))

        buy_size = min(base_size, self.remaining_buy_capacity())
        sell_size = min(base_size, self.remaining_sell_capacity())

        if self.pending_position > 0:
            buy_size = min(buy_size, max(1, base_size - ceil(abs(self.pending_position) / 10)))
        elif self.pending_position < 0:
            sell_size = min(sell_size, max(1, base_size - ceil(abs(self.pending_position) / 10)))

        return max(0, buy_size), max(0, sell_size)

    def quote_two_levels(
        self,
        bid_quote: Optional[int],
        ask_quote: Optional[int],
        signal_strength: float,
    ) -> None:
        buy_size, sell_size = self.conviction_adjusted_sizes(signal_strength)
        front_buy = ceil(buy_size * self.front_ratio)
        front_sell = ceil(sell_size * self.front_ratio)
        back_buy = buy_size - front_buy
        back_sell = sell_size - front_sell

        if bid_quote is not None and front_buy > 0:
            self.add_buy(bid_quote, front_buy)
        if ask_quote is not None and front_sell > 0:
            self.add_sell(ask_quote, front_sell)

        if bid_quote is not None and back_buy > 0:
            self.add_buy(bid_quote - self.second_level_offset, back_buy)
        if ask_quote is not None and back_sell > 0:
            self.add_sell(ask_quote + self.second_level_offset, back_sell)

    def build_orders(self) -> List[Order]:
        raise NotImplementedError


class StableProduct(Product):
    def __init__(self, state: TradingState, trader_state: Dict[str, str]) -> None:
        super().__init__(
            ASH_COATED_OSMIUM,
            state,
            trader_state,
            POSITION_LIMIT,
            ASH_QUOTE_SIZE,
            ASH_SIGNAL_SIZE_BOOST,
            ASH_SECOND_LEVEL_OFFSET,
        )
        self.front_ratio = 0.78

    def fair_value(
        self,
        bid_price: Optional[int],
        ask_price: Optional[int],
        bid_volume: Optional[int],
        ask_volume: Optional[int],
    ) -> tuple[float, float]:
        fair_value = float(ASH_FAIR_VALUE)
        imbalance = compute_order_book_imbalance(bid_volume, ask_volume)
        imbalance_shift = clamp(
            imbalance * ASH_IMBALANCE_ADJUSTMENT,
            -ASH_MAX_IMBALANCE_SHIFT,
            ASH_MAX_IMBALANCE_SHIFT,
        )

        if bid_price is not None and ask_price is not None and bid_price < ask_price:
            mid_price = (bid_price + ask_price) / 2
            fair_value = 0.7 * fair_value + 0.3 * mid_price

        fair_value += imbalance_shift
        return fair_value, abs(imbalance)

    def take_liquidity(self, fair_value: float) -> None:
        reservation_price = fair_value - self.pending_position * ASH_INVENTORY_SKEW_PER_UNIT
        buy_threshold = reservation_price - ASH_TAKE_EDGE
        sell_threshold = reservation_price + ASH_TAKE_EDGE

        for ask_price, ask_volume in self.ordered_sells():
            if ask_price <= floor(buy_threshold):
                self.add_buy(ask_price, ask_volume)
            else:
                break

        for bid_price, bid_volume in self.ordered_buys():
            if bid_price >= ceil(sell_threshold):
                self.add_sell(bid_price, bid_volume)
            else:
                break

    def make_quotes(
        self,
        fair_value: float,
        best_bid: Optional[int],
        best_ask: Optional[int],
    ) -> tuple[Optional[int], Optional[int]]:
        reservation_price = fair_value - self.pending_position * ASH_INVENTORY_SKEW_PER_UNIT
        bid_quote = floor(reservation_price - ASH_BASE_HALF_SPREAD)
        ask_quote = ceil(reservation_price + ASH_BASE_HALF_SPREAD)

        if best_bid is not None:
            bid_quote = max(bid_quote, best_bid + 1)
        if best_ask is not None:
            ask_quote = min(ask_quote, best_ask - 1)

        if best_ask is not None:
            bid_quote = min(bid_quote, best_ask - ASH_MIN_EDGE)
        if best_bid is not None:
            ask_quote = max(ask_quote, best_bid + ASH_MIN_EDGE)

        if bid_quote >= ask_quote:
            center = round(reservation_price)
            bid_quote = center - ASH_MIN_EDGE
            ask_quote = center + ASH_MIN_EDGE

        return bid_quote, ask_quote

    def build_orders(self) -> List[Order]:
        best_bid, bid_volume, best_ask, ask_volume = self.best_bid_ask()
        fair_value, signal_strength = self.fair_value(best_bid, best_ask, bid_volume, ask_volume)
        self.take_liquidity(fair_value)

        best_bid, bid_volume, best_ask, ask_volume = self.best_bid_ask()
        fair_value, signal_strength = self.fair_value(best_bid, best_ask, bid_volume, ask_volume)
        bid_quote, ask_quote = self.make_quotes(fair_value, best_bid, best_ask)
        self.quote_two_levels(bid_quote, ask_quote, signal_strength)
        return self.orders


class DynamicProduct(Product):
    def __init__(self, symbol: str, state: TradingState, trader_state: Dict[str, str]) -> None:
        super().__init__(
            symbol,
            state,
            trader_state,
            POSITION_LIMIT,
            ROOT_QUOTE_SIZE,
            ROOT_SIGNAL_SIZE_BOOST,
            ROOT_SECOND_LEVEL_OFFSET,
        )

    def parse_history(self) -> List[float]:
        raw_history = self.trader_state.get("dynamic", "")
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
        self.trader_state["dynamic"] = ",".join(f"{price:.1f}" for price in history)

    def ema(self, history: List[float], alpha: float) -> float:
        value = history[0]
        for price in history[1:]:
            value = (1 - alpha) * value + alpha * price
        return value

    def fair_value(
        self,
        history: List[float],
        best_bid: Optional[int],
        best_ask: Optional[int],
        bid_volume: Optional[int],
        ask_volume: Optional[int],
    ) -> tuple[float, float]:
        slow_ema = self.ema(history, ROOT_SLOW_ALPHA)
        fast_ema = self.ema(history, ROOT_FAST_ALPHA)
        fair_value = 0.55 * fast_ema + 0.45 * slow_ema

        if best_bid is not None and best_ask is not None and best_bid < best_ask:
            micro_price = (
                best_bid * (ask_volume or 0) + best_ask * (bid_volume or 0)
            ) / max(1, (bid_volume or 0) + (ask_volume or 0))
            fair_value = (1 - ROOT_MICRO_ALPHA) * fair_value + ROOT_MICRO_ALPHA * micro_price

        trend_shift = clamp(
            (fast_ema - slow_ema) * ROOT_TREND_WEIGHT,
            -ROOT_MAX_TREND_SHIFT,
            ROOT_MAX_TREND_SHIFT,
        )

        anchor = sum(history[-12:]) / min(len(history), 12)
        reversion_shift = clamp(
            (anchor - fair_value) * ROOT_REVERSION_WEIGHT,
            -ROOT_MAX_REVERSION_SHIFT,
            ROOT_MAX_REVERSION_SHIFT,
        )

        signal_strength = max(
            abs(trend_shift) / max(1e-9, ROOT_MAX_TREND_SHIFT),
            abs(reversion_shift) / max(1e-9, ROOT_MAX_REVERSION_SHIFT),
        )
        return fair_value + trend_shift + reversion_shift, clamp(signal_strength, 0.0, 1.0)

    def take_liquidity(self, fair_value: float) -> None:
        reservation_price = fair_value - self.pending_position * ROOT_INVENTORY_SKEW_PER_UNIT
        buy_threshold = reservation_price - ROOT_TAKE_EDGE
        sell_threshold = reservation_price + ROOT_TAKE_EDGE

        for ask_price, ask_volume in self.ordered_sells():
            if ask_price <= floor(buy_threshold):
                self.add_buy(ask_price, ask_volume)
            else:
                break

        for bid_price, bid_volume in self.ordered_buys():
            if bid_price >= ceil(sell_threshold):
                self.add_sell(bid_price, bid_volume)
            else:
                break

    def make_quotes(
        self,
        fair_value: float,
        best_bid: Optional[int],
        best_ask: Optional[int],
        signal_strength: float,
    ) -> tuple[Optional[int], Optional[int]]:
        inventory_shift = self.pending_position * ROOT_INVENTORY_SKEW_PER_UNIT
        directional_push = signal_strength * 1.3
        reservation_price = fair_value - inventory_shift

        bid_quote = floor(reservation_price - ROOT_BASE_HALF_SPREAD + directional_push)
        ask_quote = ceil(reservation_price + ROOT_BASE_HALF_SPREAD - directional_push)

        if best_bid is not None:
            bid_quote = max(bid_quote, best_bid + 1)
        if best_ask is not None:
            ask_quote = min(ask_quote, best_ask - 1)

        if best_ask is not None:
            bid_quote = min(bid_quote, best_ask - ROOT_MIN_EDGE)
        if best_bid is not None:
            ask_quote = max(ask_quote, best_bid + ROOT_MIN_EDGE)

        if bid_quote >= ask_quote:
            center = round(reservation_price)
            bid_quote = center - ROOT_MIN_EDGE
            ask_quote = center + ROOT_MIN_EDGE

        return bid_quote, ask_quote

    def build_orders(self) -> List[Order]:
        best_bid, bid_volume, best_ask, ask_volume = self.best_bid_ask()
        history = self.parse_history()

        if best_bid is not None and best_ask is not None and best_bid < best_ask:
            history.append((best_bid + best_ask) / 2)
        history = history[-ROOT_HISTORY_LENGTH:]
        self.save_history(history)

        if not history:
            return self.orders

        fair_value, signal_strength = self.fair_value(history, best_bid, best_ask, bid_volume, ask_volume)
        self.take_liquidity(fair_value)

        best_bid, bid_volume, best_ask, ask_volume = self.best_bid_ask()
        fair_value, signal_strength = self.fair_value(history, best_bid, best_ask, bid_volume, ask_volume)
        bid_quote, ask_quote = self.make_quotes(fair_value, best_bid, best_ask, signal_strength)
        self.quote_two_levels(bid_quote, ask_quote, signal_strength)
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

        if ASH_COATED_OSMIUM in state.order_depths:
            orders[ASH_COATED_OSMIUM] = StableProduct(state, trader_state).build_orders()

        dynamic_symbol = self.first_available_symbol(state.order_depths, INTARIAN_PEPPER_ROOT_ALIASES)
        if dynamic_symbol is not None:
            orders[dynamic_symbol] = DynamicProduct(dynamic_symbol, state, trader_state).build_orders()

        return orders, 0, self.encode_trader_data(trader_state)