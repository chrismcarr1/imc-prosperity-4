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

ASH_QUOTE_SIZE = 24
ASH_BASE_HALF_SPREAD = 9
ASH_MIN_EDGE = 2
ASH_INVENTORY_SKEW_PER_UNIT = 0.08
ASH_TAKE_EDGE = 0.0
ASH_IMBALANCE_ADJUSTMENT = 3.6
ASH_MAX_IMBALANCE_SHIFT = 5.0
ASH_SIGNAL_SIZE_BOOST = 16
ASH_SECOND_LEVEL_OFFSET = 1

ROOT_QUOTE_SIZE = 24
ROOT_BASE_HALF_SPREAD = 5.2
ROOT_MIN_EDGE = 2
ROOT_INVENTORY_SKEW_PER_UNIT = 0.0
ROOT_TAKE_EDGE = 0.25
ROOT_SIGNAL_SIZE_BOOST = 42
ROOT_SECOND_LEVEL_OFFSET = 1
ROOT_HISTORY_LENGTH = 72
ROOT_FAST_ALPHA = 0.38
ROOT_MEDIUM_ALPHA = 0.18
ROOT_SLOW_ALPHA = 0.055
ROOT_MICRO_ALPHA = 0.22
ROOT_IMBALANCE_WEIGHT = 1.6
ROOT_MAX_IMBALANCE_SHIFT = 3.0
ROOT_TREND_WEIGHT = 1.35
ROOT_MAX_TREND_SHIFT = 8.5
ROOT_ACCEL_WEIGHT = 1.15
ROOT_MAX_ACCEL_SHIFT = 4.0
ROOT_REVERSION_WEIGHT = 0.7
ROOT_MAX_REVERSION_SHIFT = 4.0
ROOT_BREAKOUT_WEIGHT = 0.8
ROOT_MAX_BREAKOUT_SHIFT = 3.5


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
        trader_state: Dict[str, object],
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
        self.front_ratio = 0.74

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

    def inventory_ratio(self) -> float:
        if self.position_limit <= 0:
            return 0.0
        return abs(self.pending_position) / self.position_limit

    def quote_two_levels(
        self,
        bid_quote: Optional[int],
        ask_quote: Optional[int],
        bid_size: int,
        sell_size: int,
        front_ratio_buy: Optional[float] = None,
        front_ratio_sell: Optional[float] = None,
    ) -> None:
        use_front_buy = clamp(front_ratio_buy if front_ratio_buy is not None else self.front_ratio, 0.45, 1.0)
        use_front_sell = clamp(front_ratio_sell if front_ratio_sell is not None else self.front_ratio, 0.45, 1.0)

        front_buy = ceil(bid_size * use_front_buy)
        front_sell = ceil(sell_size * use_front_sell)
        back_buy = bid_size - front_buy
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
    def __init__(self, state: TradingState, trader_state: Dict[str, object]) -> None:
        super().__init__(
            ASH_COATED_OSMIUM,
            state,
            trader_state,
            POSITION_LIMIT,
            ASH_QUOTE_SIZE,
            ASH_SIGNAL_SIZE_BOOST,
            ASH_SECOND_LEVEL_OFFSET,
        )
        self.front_ratio = 0.82

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
            fair_value = 0.6 * fair_value + 0.4 * ((bid_price + ask_price) / 2)

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
        signal_strength: float,
    ) -> tuple[Optional[int], Optional[int], int, int]:
        reservation_price = fair_value - self.pending_position * ASH_INVENTORY_SKEW_PER_UNIT
        half_spread = ASH_BASE_HALF_SPREAD - 1.5 * signal_strength
        bid_quote = floor(reservation_price - half_spread)
        ask_quote = ceil(reservation_price + half_spread)

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

        inv = self.inventory_ratio()
        boost = round(signal_strength * self.signal_size_boost)
        base_size = max(4, round((self.quote_size + boost) * (1 - 0.35 * inv)))
        buy_size = min(base_size, self.remaining_buy_capacity())
        sell_size = min(base_size, self.remaining_sell_capacity())
        return bid_quote, ask_quote, buy_size, sell_size

    def build_orders(self) -> List[Order]:
        best_bid, bid_volume, best_ask, ask_volume = self.best_bid_ask()
        fair_value, signal_strength = self.fair_value(best_bid, best_ask, bid_volume, ask_volume)
        self.take_liquidity(fair_value)

        best_bid, bid_volume, best_ask, ask_volume = self.best_bid_ask()
        fair_value, signal_strength = self.fair_value(best_bid, best_ask, bid_volume, ask_volume)
        bid_quote, ask_quote, buy_size, sell_size = self.make_quotes(fair_value, best_bid, best_ask, signal_strength)
        self.quote_two_levels(bid_quote, ask_quote, buy_size, sell_size)
        return self.orders


class AggressivePepperProduct(Product):
    def __init__(self, symbol: str, state: TradingState, trader_state: Dict[str, object]) -> None:
        super().__init__(
            symbol,
            state,
            trader_state,
            POSITION_LIMIT,
            ROOT_QUOTE_SIZE,
            ROOT_SIGNAL_SIZE_BOOST,
            ROOT_SECOND_LEVEL_OFFSET,
        )

    def _load_series(self, key: str) -> List[float]:
        raw = self.trader_state.get(key, [])
        if isinstance(raw, list):
            result: List[float] = []
            for value in raw:
                try:
                    result.append(float(value))
                except (TypeError, ValueError):
                    continue
            return result

        if isinstance(raw, str):
            values: List[float] = []
            for item in raw.split(","):
                if not item:
                    continue
                try:
                    values.append(float(item))
                except ValueError:
                    continue
            return values

        return []

    def _save_series(self, key: str, values: List[float]) -> None:
        self.trader_state[key] = [round(value, 3) for value in values[-ROOT_HISTORY_LENGTH:]]

    def ema(self, history: List[float], alpha: float) -> float:
        value = history[0]
        for price in history[1:]:
            value = (1 - alpha) * value + alpha * price
        return value

    def mean_abs_diff(self, history: List[float], window: int) -> float:
        if len(history) < 2:
            return 0.0
        diffs = [abs(history[i] - history[i - 1]) for i in range(max(1, len(history) - window), len(history))]
        if not diffs:
            return 0.0
        return sum(diffs) / len(diffs)

    def collect_state(
        self,
        best_bid: Optional[int],
        bid_volume: Optional[int],
        best_ask: Optional[int],
        ask_volume: Optional[int],
    ) -> tuple[List[float], List[float]]:
        prices = self._load_series(f"{self.symbol}_prices")
        spreads = self._load_series(f"{self.symbol}_spreads")

        if best_bid is not None and best_ask is not None and best_bid < best_ask:
            prices.append((best_bid + best_ask) / 2)
            spreads.append(float(best_ask - best_bid))
        elif best_bid is not None:
            prices.append(float(best_bid))
        elif best_ask is not None:
            prices.append(float(best_ask))

        prices = prices[-ROOT_HISTORY_LENGTH:]
        spreads = spreads[-ROOT_HISTORY_LENGTH:]
        self._save_series(f"{self.symbol}_prices", prices)
        self._save_series(f"{self.symbol}_spreads", spreads)
        return prices, spreads

    def fair_value(
        self,
        history: List[float],
        spreads: List[float],
        best_bid: Optional[int],
        best_ask: Optional[int],
        bid_volume: Optional[int],
        ask_volume: Optional[int],
    ) -> tuple[float, float, float, float, float]:
        fast_ema = self.ema(history, ROOT_FAST_ALPHA)
        medium_ema = self.ema(history, ROOT_MEDIUM_ALPHA)
        slow_ema = self.ema(history, ROOT_SLOW_ALPHA)
        base_fair = 0.5 * fast_ema + 0.3 * medium_ema + 0.2 * slow_ema

        current_mid = history[-1]
        avg_spread = sum(spreads[-20:]) / max(1, min(len(spreads), 20)) if spreads else 12.0
        realized_vol = self.mean_abs_diff(history, 14)

        imbalance = compute_order_book_imbalance(bid_volume, ask_volume)
        imbalance_shift = clamp(
            imbalance * ROOT_IMBALANCE_WEIGHT,
            -ROOT_MAX_IMBALANCE_SHIFT,
            ROOT_MAX_IMBALANCE_SHIFT,
        )

        micro_shift = 0.0
        if best_bid is not None and best_ask is not None and best_bid < best_ask:
            micro_price = (
                best_bid * (ask_volume or 0) + best_ask * (bid_volume or 0)
            ) / max(1, (bid_volume or 0) + (ask_volume or 0))
            micro_shift = (micro_price - base_fair) * ROOT_MICRO_ALPHA

        trend_shift = clamp(
            (fast_ema - slow_ema) * ROOT_TREND_WEIGHT,
            -ROOT_MAX_TREND_SHIFT,
            ROOT_MAX_TREND_SHIFT,
        )
        accel_raw = 0.0
        if len(history) >= 8:
            accel_raw = (history[-1] - history[-4]) - (history[-4] - history[-7])
        accel_shift = clamp(
            accel_raw * ROOT_ACCEL_WEIGHT,
            -ROOT_MAX_ACCEL_SHIFT,
            ROOT_MAX_ACCEL_SHIFT,
        )

        anchor = sum(history[-10:]) / min(len(history), 10)
        reversion_shift = clamp(
            (anchor - current_mid) * ROOT_REVERSION_WEIGHT,
            -ROOT_MAX_REVERSION_SHIFT,
            ROOT_MAX_REVERSION_SHIFT,
        )

        breakout_shift = 0.0
        if len(history) >= 12:
            breakout = current_mid - max(history[-12:-1])
            breakdown = current_mid - min(history[-12:-1])
            if breakout > 0:
                breakout_shift = clamp(breakout * ROOT_BREAKOUT_WEIGHT, 0.0, ROOT_MAX_BREAKOUT_SHIFT)
            elif breakdown < 0:
                breakout_shift = clamp(breakdown * ROOT_BREAKOUT_WEIGHT, -ROOT_MAX_BREAKOUT_SHIFT, 0.0)

        fair_value = base_fair + micro_shift + imbalance_shift + trend_shift + accel_shift + reversion_shift + breakout_shift
        signal_strength = clamp(
            (
                abs(trend_shift) / ROOT_MAX_TREND_SHIFT
                + abs(accel_shift) / ROOT_MAX_ACCEL_SHIFT
                + abs(breakout_shift) / max(1e-9, ROOT_MAX_BREAKOUT_SHIFT)
            ) / 3,
            0.0,
            1.0,
        )
        alpha = fair_value - current_mid
        return fair_value, signal_strength, alpha, realized_vol, avg_spread

    def take_liquidity(
        self,
        fair_value: float,
        signal_strength: float,
        alpha: float,
        realized_vol: float,
    ) -> None:
        inv_ratio = self.inventory_ratio()
        buy_edge = max(0.0, ROOT_TAKE_EDGE - 0.55 * max(alpha, 0.0) + 0.12 * realized_vol)
        sell_edge = max(0.0, ROOT_TAKE_EDGE - 0.35 * max(-alpha, 0.0) + 0.12 * realized_vol)

        long_unwind_discount = 2.1 * inv_ratio + 1.5 * max(-alpha, 0.0)
        short_cover_bonus = 2.1 * inv_ratio + 1.5 * max(alpha, 0.0)

        buy_threshold = fair_value - buy_edge + short_cover_bonus * (1 if self.pending_position < 0 else 0)
        sell_threshold = fair_value + sell_edge - long_unwind_discount * (1 if self.pending_position > 0 else 0)

        take_buy_cap = max(0, round(self.quote_size + signal_strength * self.signal_size_boost * 0.8))
        take_sell_cap = max(0, round(self.quote_size + signal_strength * self.signal_size_boost * 0.8))

        bought = 0
        for ask_price, ask_volume in self.ordered_sells():
            if ask_price <= floor(buy_threshold) and bought < take_buy_cap:
                bought += self.add_buy(ask_price, min(ask_volume, take_buy_cap - bought))
            else:
                break

        sold = 0
        for bid_price, bid_volume in self.ordered_buys():
            if bid_price >= ceil(sell_threshold) and sold < take_sell_cap:
                sold += self.add_sell(bid_price, min(bid_volume, take_sell_cap - sold))
            else:
                break

    def make_quotes(
        self,
        fair_value: float,
        signal_strength: float,
        alpha: float,
        realized_vol: float,
        avg_spread: float,
        best_bid: Optional[int],
        best_ask: Optional[int],
    ) -> tuple[Optional[int], Optional[int], int, int, float, float]:
        inv_ratio = self.inventory_ratio()
        compression = clamp((avg_spread - 8.0) / 8.0, -0.5, 1.0)
        half_spread = clamp(
            ROOT_BASE_HALF_SPREAD + 0.35 * realized_vol - 1.2 * signal_strength - 0.6 * compression,
            3.0,
            8.0,
        )

        reservation_price = fair_value
        bid_quote = floor(reservation_price - half_spread + 0.25 * max(alpha, 0.0))
        ask_quote = ceil(reservation_price + half_spread + 0.25 * min(alpha, 0.0))

        if best_bid is not None:
            bid_quote = max(bid_quote, best_bid + 1)
        if best_ask is not None:
            ask_quote = min(ask_quote, best_ask - 1)

        if best_ask is not None:
            bid_quote = min(bid_quote, best_ask - ROOT_MIN_EDGE)
        if best_bid is not None:
            ask_quote = max(ask_quote, best_bid + ROOT_MIN_EDGE)

        if self.pending_position > 0:
            ask_quote -= ceil(1.5 + 4.0 * inv_ratio + 1.2 * max(-alpha, 0.0))
            if best_bid is not None:
                ask_quote = max(best_bid + 1, ask_quote)
            if best_ask is not None:
                ask_quote = min(best_ask, ask_quote)

        if self.pending_position < 0:
            bid_quote += ceil(1.5 + 4.0 * inv_ratio + 1.2 * max(alpha, 0.0))
            if best_ask is not None:
                bid_quote = min(best_ask - 1, bid_quote)
            if best_bid is not None:
                bid_quote = max(best_bid, bid_quote)

        if bid_quote >= ask_quote:
            center = round(reservation_price)
            bid_quote = center - ROOT_MIN_EDGE
            ask_quote = center + ROOT_MIN_EDGE

        base_size = max(5, round((self.quote_size + signal_strength * self.signal_size_boost) * (1 - 0.25 * inv_ratio)))
        buy_size = min(base_size, self.remaining_buy_capacity())
        sell_size = min(base_size, self.remaining_sell_capacity())

        if alpha > 0:
            buy_size = min(self.remaining_buy_capacity(), round(buy_size * (1.15 + 0.5 * signal_strength)))
            if self.pending_position > 0:
                sell_size = min(self.remaining_sell_capacity(), round(sell_size * (1.1 + 0.5 * inv_ratio)))
        elif alpha < 0:
            sell_size = min(self.remaining_sell_capacity(), round(sell_size * (1.15 + 0.5 * signal_strength)))
            if self.pending_position < 0:
                buy_size = min(self.remaining_buy_capacity(), round(buy_size * (1.1 + 0.5 * inv_ratio)))

        if self.pending_position > 0:
            sell_size = min(self.remaining_sell_capacity(), round(max(sell_size, base_size * (1.0 + 1.25 * inv_ratio))))
        elif self.pending_position < 0:
            buy_size = min(self.remaining_buy_capacity(), round(max(buy_size, base_size * (1.0 + 1.25 * inv_ratio))))

        buy_front = clamp(0.68 + 0.15 * max(alpha, 0.0), 0.58, 0.92)
        sell_front = clamp(0.68 + 0.15 * max(-alpha, 0.0), 0.58, 0.92)

        if self.pending_position > 0:
            sell_front = clamp(sell_front + 0.12 + 0.1 * inv_ratio, 0.58, 0.97)
        elif self.pending_position < 0:
            buy_front = clamp(buy_front + 0.12 + 0.1 * inv_ratio, 0.58, 0.97)

        return bid_quote, ask_quote, buy_size, sell_size, buy_front, sell_front

    def build_orders(self) -> List[Order]:
        best_bid, bid_volume, best_ask, ask_volume = self.best_bid_ask()
        history, spreads = self.collect_state(best_bid, bid_volume, best_ask, ask_volume)
        if not history:
            return self.orders

        fair_value, signal_strength, alpha, realized_vol, avg_spread = self.fair_value(
            history,
            spreads,
            best_bid,
            best_ask,
            bid_volume,
            ask_volume,
        )
        self.take_liquidity(fair_value, signal_strength, alpha, realized_vol)

        best_bid, bid_volume, best_ask, ask_volume = self.best_bid_ask()
        bid_quote, ask_quote, buy_size, sell_size, buy_front, sell_front = self.make_quotes(
            fair_value,
            signal_strength,
            alpha,
            realized_vol,
            avg_spread,
            best_bid,
            best_ask,
        )
        self.quote_two_levels(
            bid_quote,
            ask_quote,
            buy_size,
            sell_size,
            front_ratio_buy=buy_front,
            front_ratio_sell=sell_front,
        )
        return self.orders


class Trader:
    @staticmethod
    def decode_trader_data(trader_data: str) -> Dict[str, object]:
        if not trader_data:
            return {}

        try:
            decoded = json.loads(trader_data)
            return decoded if isinstance(decoded, dict) else {}
        except json.JSONDecodeError:
            decoded: Dict[str, object] = {}
            for segment in trader_data.split(";"):
                if "=" not in segment:
                    continue
                key, value = segment.split("=", 1)
                if key:
                    decoded[key] = value
            return decoded

    @staticmethod
    def encode_trader_data(trader_state: Dict[str, object]) -> str:
        try:
            return json.dumps(trader_state, separators=(",", ":"))
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
            orders[dynamic_symbol] = AggressivePepperProduct(dynamic_symbol, state, trader_state).build_orders()

        return orders, 0, self.encode_trader_data(trader_state)