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
        market_trades: Dict[str, list] = field(default_factory=dict)
        traderData: str = ""
        timestamp: int = 0


HYDROGEL_PACK = "HYDROGEL_PACK"
VELVETFRUIT_EXTRACT = "VELVETFRUIT_EXTRACT"
VEV_4000 = "VEV_4000"
VEV_4500 = "VEV_4500"
VEV_5000 = "VEV_5000"
VEV_5100 = "VEV_5100"
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
VEV_5400 = "VEV_5400"
VEV_5500 = "VEV_5500"
VEV_6000 = "VEV_6000"
VEV_6500 = "VEV_6500"
HYDRO_POSITION_LIMIT = 200
FRUIT_POSITION_LIMIT = 200
VOUCHER_POSITION_LIMIT = 300

HYDRO_MEAN_ANCHOR = 10000.0
HYDRO_NEUTRAL_LOWER = 9970.0
HYDRO_NEUTRAL_UPPER = 10030.0
HYDRO_HARD_BUY_LEVEL = 9950.0
HYDRO_HARD_SELL_LEVEL = 10050.0
HYDRO_HISTORY_LIMIT = 90
HYDRO_SOFT_FAIR_SKEW = 18.0
HYDRO_HARD_FAIR_SKEW = 45.0

BASE_QUOTE_SIZE = 6
MAX_QUOTE_SIZE = 16
HARD_QUOTE_SIZE = 28
HARD_TAKE_SIZE = 70
BASE_HALF_SPREAD = 8.0
MIN_EDGE = 2
TAKE_EDGE = 6.0
INVENTORY_SKEW = 0.16

FRUIT_HISTORY_LIMIT = 150
FRUIT_FAST_ALPHA = 0.20
FRUIT_SLOW_ALPHA = 0.040
FRUIT_MICRO_WEIGHT = 0.62
FRUIT_TREND_WEIGHT = 0.34
FRUIT_MAX_TREND_SHIFT = 2.6
FRUIT_IMBALANCE_WEIGHT = 0.55
FRUIT_MAX_IMBALANCE_SHIFT = 0.8
FRUIT_BASE_HALF_SPREAD = 2.35
FRUIT_MIN_EDGE = 1
FRUIT_TAKE_EDGE = 4.4
FRUIT_QUOTE_SIZE = 16
FRUIT_MAX_QUOTE_SIZE = 24
FRUIT_MAX_TAKE_SIZE = 14
FRUIT_SECOND_LEVEL_OFFSET = 1
FRUIT_INVENTORY_SKEW = 0.105
FRUIT_POSITION_UNWIND_TRIGGER = 115


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def top_of_book(order_depth: Optional[OrderDepth]) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    if order_depth is None:
        return None, None, None, None

    best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
    best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
    bid_volume = order_depth.buy_orders.get(best_bid) if best_bid is not None else None
    ask_volume = abs(order_depth.sell_orders.get(best_ask)) if best_ask is not None else None
    return best_bid, bid_volume, best_ask, ask_volume


def mid_price(order_depth: Optional[OrderDepth]) -> Optional[float]:
    best_bid, _, best_ask, _ = top_of_book(order_depth)
    if best_bid is None or best_ask is None or best_bid >= best_ask:
        return None
    return (best_bid + best_ask) / 2


def order_book_imbalance(bid_volume: Optional[int], ask_volume: Optional[int]) -> float:
    bid = max(0, int(bid_volume or 0))
    ask = max(0, int(ask_volume or 0))
    total = bid + ask
    if total <= 0:
        return 0.0
    return (bid - ask) / total


def parse_float_series(raw_values: str) -> List[float]:
    values: List[float] = []
    for raw_value in raw_values.split(","):
        if not raw_value:
            continue
        try:
            values.append(float(raw_value))
        except ValueError:
            continue
    return values


def save_float_series(values: List[float], precision: int = 2) -> str:
    return ",".join(f"{value:.{precision}f}" for value in values)


def ema(values: List[float], alpha: float) -> float:
    value = values[0]
    for price in values[1:]:
        value = (1.0 - alpha) * value + alpha * price
    return value


class HydroMeanReversionMarketMaker:
    def __init__(self, state: TradingState, trader_state: Dict[str, str]) -> None:
        self.state = state
        self.trader_state = trader_state
        self.symbol = HYDROGEL_PACK
        self.position = state.position.get(self.symbol, 0)
        self.pending_position = self.position
        self.order_depth = state.order_depths.get(self.symbol)
        self.orders: List[Order] = []

    def history_key(self) -> str:
        return "hist:hydro"

    def parse_history(self) -> List[float]:
        return parse_float_series(self.trader_state.get(self.history_key(), ""))

    def save_history(self, history: List[float]) -> None:
        self.trader_state[self.history_key()] = save_float_series(history[-HYDRO_HISTORY_LIMIT:], 2)

    def update_history(self) -> List[float]:
        history = self.parse_history()
        current_mid = mid_price(self.order_depth)
        if current_mid is not None:
            history.append(current_mid)
        history = history[-HYDRO_HISTORY_LIMIT:]
        self.save_history(history)
        return history

    def reversion_signal(self, current_mid: float) -> float:
        if current_mid < HYDRO_HARD_BUY_LEVEL:
            return 1.0
        if current_mid > HYDRO_HARD_SELL_LEVEL:
            return -1.0
        if current_mid < HYDRO_NEUTRAL_LOWER:
            distance = HYDRO_NEUTRAL_LOWER - current_mid
            width = HYDRO_NEUTRAL_LOWER - HYDRO_HARD_BUY_LEVEL
            return clamp(distance / width, 0.0, 1.0)
        if current_mid > HYDRO_NEUTRAL_UPPER:
            distance = current_mid - HYDRO_NEUTRAL_UPPER
            width = HYDRO_HARD_SELL_LEVEL - HYDRO_NEUTRAL_UPPER
            return -clamp(distance / width, 0.0, 1.0)
        return 0.0

    def is_hard_reversion_zone(self, current_mid: float, signal: float) -> bool:
        return (
            (signal > 0 and current_mid < HYDRO_HARD_BUY_LEVEL)
            or (signal < 0 and current_mid > HYDRO_HARD_SELL_LEVEL)
        )

    def fair_value(self, history: List[float], signal: float, current_mid: float) -> float:
        current_mid = history[-1] if history else mid_price(self.order_depth)
        if current_mid is None:
            return 0.0

        lookback = min(len(history), 24)
        recent_mean = sum(history[-lookback:]) / lookback if lookback else current_mid
        anchored_mean = 0.80 * HYDRO_MEAN_ANCHOR + 0.20 * recent_mean
        neutral_fair = 0.55 * current_mid + 0.45 * anchored_mean
        if signal == 0:
            return neutral_fair

        max_skew = HYDRO_HARD_FAIR_SKEW if self.is_hard_reversion_zone(current_mid, signal) else HYDRO_SOFT_FAIR_SKEW
        return neutral_fair + signal * max_skew

    def remaining_buy_capacity(self) -> int:
        return max(0, HYDRO_POSITION_LIMIT - self.pending_position)

    def remaining_sell_capacity(self) -> int:
        return max(0, HYDRO_POSITION_LIMIT + self.pending_position)

    def add_buy(self, price: int, volume: int) -> None:
        size = min(max(0, int(volume)), self.remaining_buy_capacity())
        if price >= 0 and size > 0:
            self.orders.append(Order(self.symbol, int(price), size))
            self.pending_position += size

    def add_sell(self, price: int, volume: int) -> None:
        size = min(max(0, int(volume)), self.remaining_sell_capacity())
        if price >= 0 and size > 0:
            self.orders.append(Order(self.symbol, int(price), -size))
            self.pending_position -= size

    def take_reversion_liquidity(self, fair_value: float, signal: float, current_mid: float) -> None:
        if self.order_depth is None or not self.is_hard_reversion_zone(current_mid, signal):
            return

        best_bid, bid_volume, best_ask, ask_volume = top_of_book(self.order_depth)
        if best_bid is None or best_ask is None:
            return

        reservation_price = fair_value - self.pending_position * INVENTORY_SKEW
        signal_strength = abs(signal)
        take_edge = max(MIN_EDGE, TAKE_EDGE - 4.0 * signal_strength)
        max_take = max(BASE_QUOTE_SIZE, round(HARD_TAKE_SIZE * signal_strength))

        if signal > 0:
            limit_price = floor(reservation_price - take_edge)
            remaining = min(max_take, self.remaining_buy_capacity())
            for price in sorted(self.order_depth.sell_orders):
                if remaining <= 0 or price > limit_price:
                    break
                volume = min(abs(self.order_depth.sell_orders[price]), remaining)
                self.add_buy(price, volume)
                remaining -= volume

        if signal < 0:
            limit_price = ceil(reservation_price + take_edge)
            remaining = min(max_take, self.remaining_sell_capacity())
            for price in sorted(self.order_depth.buy_orders, reverse=True):
                if remaining <= 0 or price < limit_price:
                    break
                volume = min(self.order_depth.buy_orders[price], remaining)
                self.add_sell(price, volume)
                remaining -= volume

    def make_liquidity(self, fair_value: float, signal: float, current_mid: float) -> None:
        best_bid, _, best_ask, _ = top_of_book(self.order_depth)
        if best_bid is None or best_ask is None:
            return

        signal_strength = abs(signal)
        hard_zone = self.is_hard_reversion_zone(current_mid, signal)
        reservation_price = fair_value - self.pending_position * INVENTORY_SKEW
        directional_push = signal * (2.0 + 3.0 * signal_strength)
        half_spread = BASE_HALF_SPREAD + abs(self.pending_position) / HYDRO_POSITION_LIMIT * 2.5

        bid_quote: Optional[int] = floor(reservation_price - half_spread + directional_push)
        ask_quote: Optional[int] = ceil(reservation_price + half_spread + directional_push)

        if signal > 0:
            bid_quote = max(bid_quote, best_bid + 1)
            ask_quote = None if hard_zone and self.pending_position <= 0 else max(ask_quote, best_ask)
        elif signal < 0:
            ask_quote = min(ask_quote, best_ask - 1)
            bid_quote = None if hard_zone and self.pending_position >= 0 else min(bid_quote, best_bid)
        else:
            bid_quote = max(bid_quote, best_bid)
            ask_quote = min(ask_quote, best_ask)

        if bid_quote is not None:
            bid_quote = min(bid_quote, best_ask - MIN_EDGE)
        if ask_quote is not None:
            ask_quote = max(ask_quote, best_bid + MIN_EDGE)

        if self.pending_position > HYDRO_POSITION_LIMIT - BASE_QUOTE_SIZE:
            bid_quote = None
        if self.pending_position < -HYDRO_POSITION_LIMIT + BASE_QUOTE_SIZE:
            ask_quote = None

        if bid_quote is not None and ask_quote is not None and bid_quote >= ask_quote:
            center = round(reservation_price)
            bid_quote = center - MIN_EDGE
            ask_quote = center + MIN_EDGE

        if hard_zone:
            quote_size = HARD_QUOTE_SIZE
        else:
            quote_size = max(1, min(MAX_QUOTE_SIZE, round(BASE_QUOTE_SIZE * (1.0 + signal_strength))))

        front_size = ceil(quote_size * 0.75)
        back_size = quote_size - front_size

        if bid_quote is not None:
            self.add_buy(bid_quote, front_size)
            if back_size > 0:
                self.add_buy(bid_quote - 2, back_size)
        if ask_quote is not None:
            self.add_sell(ask_quote, front_size)
            if back_size > 0:
                self.add_sell(ask_quote + 2, back_size)

    def build_orders(self) -> List[Order]:
        history = self.update_history()
        if not history:
            return self.orders

        current_mid = history[-1]
        signal = self.reversion_signal(current_mid)
        fair_value = self.fair_value(history, signal, current_mid)
        self.take_reversion_liquidity(fair_value, signal, current_mid)
        self.make_liquidity(fair_value, signal, current_mid)
        return self.orders


class FruitMomentumMarketMaker:
    def __init__(self, state: TradingState, trader_state: Dict[str, str]) -> None:
        self.state = state
        self.trader_state = trader_state
        self.symbol = VELVETFRUIT_EXTRACT
        self.position = state.position.get(self.symbol, 0)
        self.pending_position = self.position
        self.order_depth = state.order_depths.get(self.symbol)
        self.orders: List[Order] = []

    def history_key(self) -> str:
        return "hist:fruit_momentum"

    def parse_history(self) -> List[float]:
        return parse_float_series(self.trader_state.get(self.history_key(), ""))

    def save_history(self, history: List[float]) -> None:
        self.trader_state[self.history_key()] = save_float_series(history[-FRUIT_HISTORY_LIMIT:], 2)

    def update_history(self) -> List[float]:
        history = self.parse_history()
        current_mid = mid_price(self.order_depth)
        if current_mid is not None:
            history.append(current_mid)
        history = history[-FRUIT_HISTORY_LIMIT:]
        self.save_history(history)
        return history

    def remaining_buy_capacity(self) -> int:
        return max(0, FRUIT_POSITION_LIMIT - self.pending_position)

    def remaining_sell_capacity(self) -> int:
        return max(0, FRUIT_POSITION_LIMIT + self.pending_position)

    def add_buy(self, price: int, volume: int) -> None:
        size = min(max(0, int(volume)), self.remaining_buy_capacity())
        if price >= 0 and size > 0:
            self.orders.append(Order(self.symbol, int(price), size))
            self.pending_position += size

    def add_sell(self, price: int, volume: int) -> None:
        size = min(max(0, int(volume)), self.remaining_sell_capacity())
        if price >= 0 and size > 0:
            self.orders.append(Order(self.symbol, int(price), -size))
            self.pending_position -= size

    def fair_value(self, history: List[float]) -> tuple[float, float]:
        best_bid, bid_volume, best_ask, ask_volume = top_of_book(self.order_depth)
        current_mid = history[-1]
        fast_ema = ema(history, FRUIT_FAST_ALPHA)
        slow_ema = ema(history, FRUIT_SLOW_ALPHA)
        trend = fast_ema - slow_ema
        trend_shift = clamp(
            trend * FRUIT_TREND_WEIGHT,
            -FRUIT_MAX_TREND_SHIFT,
            FRUIT_MAX_TREND_SHIFT,
        )

        fair_value = 0.35 * current_mid + 0.30 * fast_ema + 0.35 * slow_ema
        if best_bid is not None and best_ask is not None and best_bid < best_ask:
            micro_price = (
                best_bid * (ask_volume or 0) + best_ask * (bid_volume or 0)
            ) / max(1, (bid_volume or 0) + (ask_volume or 0))
            fair_value = (1.0 - FRUIT_MICRO_WEIGHT) * fair_value + FRUIT_MICRO_WEIGHT * micro_price

        imbalance = order_book_imbalance(bid_volume, ask_volume)
        imbalance_shift = clamp(
            imbalance * FRUIT_IMBALANCE_WEIGHT,
            -FRUIT_MAX_IMBALANCE_SHIFT,
            FRUIT_MAX_IMBALANCE_SHIFT,
        )
        signal_strength = max(
            abs(trend_shift) / max(1e-9, FRUIT_MAX_TREND_SHIFT),
            abs(imbalance_shift) / max(1e-9, FRUIT_MAX_IMBALANCE_SHIFT),
            abs(self.pending_position) / FRUIT_POSITION_LIMIT * 0.55,
        )
        return fair_value + trend_shift + imbalance_shift, clamp(signal_strength, 0.0, 1.0)

    def take_liquidity(self, fair_value: float) -> None:
        if self.order_depth is None:
            return

        reservation_price = fair_value - self.pending_position * FRUIT_INVENTORY_SKEW
        buy_threshold = floor(reservation_price - FRUIT_TAKE_EDGE)
        sell_threshold = ceil(reservation_price + FRUIT_TAKE_EDGE)

        for ask_price in sorted(self.order_depth.sell_orders):
            if ask_price > buy_threshold:
                break
            ask_volume = abs(self.order_depth.sell_orders[ask_price])
            self.add_buy(ask_price, min(ask_volume, FRUIT_MAX_TAKE_SIZE))

        for bid_price in sorted(self.order_depth.buy_orders, reverse=True):
            if bid_price < sell_threshold:
                break
            bid_volume = self.order_depth.buy_orders[bid_price]
            self.add_sell(bid_price, min(bid_volume, FRUIT_MAX_TAKE_SIZE))

        best_bid, bid_volume, best_ask, ask_volume = top_of_book(self.order_depth)
        if best_ask is not None and self.pending_position < -FRUIT_POSITION_UNWIND_TRIGGER:
            if best_ask <= ceil(fair_value + FRUIT_MIN_EDGE):
                self.add_buy(best_ask, min(ask_volume or 0, FRUIT_MAX_TAKE_SIZE))
        if best_bid is not None and self.pending_position > FRUIT_POSITION_UNWIND_TRIGGER:
            if best_bid >= floor(fair_value - FRUIT_MIN_EDGE):
                self.add_sell(best_bid, min(bid_volume or 0, FRUIT_MAX_TAKE_SIZE))

    def quote_size(self, signal_strength: float) -> int:
        inventory_ratio = abs(self.pending_position) / FRUIT_POSITION_LIMIT
        size = round(FRUIT_QUOTE_SIZE * (1.0 + 0.45 * signal_strength - 0.35 * inventory_ratio))
        return max(4, min(FRUIT_MAX_QUOTE_SIZE, size))

    def make_liquidity(self, fair_value: float, signal_strength: float) -> None:
        best_bid, _, best_ask, _ = top_of_book(self.order_depth)
        if best_bid is None or best_ask is None:
            return

        reservation_price = fair_value - self.pending_position * FRUIT_INVENTORY_SKEW
        inventory_ratio = abs(self.pending_position) / FRUIT_POSITION_LIMIT
        half_spread = FRUIT_BASE_HALF_SPREAD + 1.1 * inventory_ratio

        bid_quote: Optional[int] = floor(reservation_price - half_spread)
        ask_quote: Optional[int] = ceil(reservation_price + half_spread)

        bid_quote = max(bid_quote, best_bid + 1)
        ask_quote = min(ask_quote, best_ask - 1)
        bid_quote = min(bid_quote, best_ask - FRUIT_MIN_EDGE)
        ask_quote = max(ask_quote, best_bid + FRUIT_MIN_EDGE)

        if self.pending_position > FRUIT_POSITION_LIMIT - FRUIT_QUOTE_SIZE:
            bid_quote = None
        if self.pending_position < -FRUIT_POSITION_LIMIT + FRUIT_QUOTE_SIZE:
            ask_quote = None

        if bid_quote is not None and ask_quote is not None and bid_quote >= ask_quote:
            center = round(reservation_price)
            bid_quote = center - FRUIT_MIN_EDGE
            ask_quote = center + FRUIT_MIN_EDGE

        size = self.quote_size(signal_strength)
        front_size = ceil(size * 0.72)
        back_size = size - front_size

        if bid_quote is not None:
            self.add_buy(bid_quote, front_size)
            if back_size > 0:
                self.add_buy(bid_quote - FRUIT_SECOND_LEVEL_OFFSET, back_size)
        if ask_quote is not None:
            self.add_sell(ask_quote, front_size)
            if back_size > 0:
                self.add_sell(ask_quote + FRUIT_SECOND_LEVEL_OFFSET, back_size)

    def build_orders(self) -> List[Order]:
        history = self.update_history()
        if not history:
            return self.orders

        fair_value, signal_strength = self.fair_value(history)
        self.take_liquidity(fair_value)
        fair_value, signal_strength = self.fair_value(history)
        self.make_liquidity(fair_value, signal_strength)
        return self.orders


@dataclass(frozen=True)
class VoucherBandConfig:
    symbol: str
    hard_buy: float
    hard_sell: float
    fair_anchor: float
    quote_size: int
    max_quote_size: int
    hard_take_size: int
    base_half_spread: float
    min_edge: int
    take_edge: float
    hard_edge: float
    inventory_skew: float
    second_level_offset: int = 1
    history_limit: int = 120
    fast_alpha: float = 0.18
    slow_alpha: float = 0.035
    micro_weight: float = 0.55
    middle_take_enabled: bool = True


class VoucherBandMarketMaker:
    config: VoucherBandConfig

    def __init__(self, state: TradingState, trader_state: Dict[str, str]) -> None:
        self.config = self.__class__.config
        self.state = state
        self.trader_state = trader_state
        self.symbol = self.config.symbol
        self.position = state.position.get(self.symbol, 0)
        self.pending_position = self.position
        self.order_depth = state.order_depths.get(self.symbol)
        self.orders: List[Order] = []

    def history_key(self) -> str:
        return f"hist:{self.symbol.lower()}_band"

    def parse_history(self) -> List[float]:
        return parse_float_series(self.trader_state.get(self.history_key(), ""))

    def save_history(self, history: List[float]) -> None:
        self.trader_state[self.history_key()] = save_float_series(history[-self.config.history_limit:], 2)

    def update_history(self) -> List[float]:
        history = self.parse_history()
        current_mid = mid_price(self.order_depth)
        if current_mid is not None:
            history.append(current_mid)
        history = history[-self.config.history_limit:]
        self.save_history(history)
        return history

    def remaining_buy_capacity(self) -> int:
        return max(0, VOUCHER_POSITION_LIMIT - self.pending_position)

    def remaining_sell_capacity(self) -> int:
        return max(0, VOUCHER_POSITION_LIMIT + self.pending_position)

    def add_buy(self, price: int, volume: int) -> None:
        size = min(max(0, int(volume)), self.remaining_buy_capacity())
        if price >= 0 and size > 0:
            self.orders.append(Order(self.symbol, int(price), size))
            self.pending_position += size

    def add_sell(self, price: int, volume: int) -> None:
        size = min(max(0, int(volume)), self.remaining_sell_capacity())
        if price >= 0 and size > 0:
            self.orders.append(Order(self.symbol, int(price), -size))
            self.pending_position -= size

    def regime_signal(self, current_mid: float) -> int:
        if current_mid < self.config.hard_buy:
            return 1
        if current_mid > self.config.hard_sell:
            return -1
        return 0

    def fair_value(self, history: List[float], current_mid: float, signal: int) -> float:
        best_bid, bid_volume, best_ask, ask_volume = top_of_book(self.order_depth)
        fast = ema(history, self.config.fast_alpha)
        slow = ema(history, self.config.slow_alpha)
        rolling_fair = 0.40 * current_mid + 0.30 * fast + 0.30 * slow

        if best_bid is not None and best_ask is not None and best_bid < best_ask:
            micro_price = (
                best_bid * (ask_volume or 0) + best_ask * (bid_volume or 0)
            ) / max(1, (bid_volume or 0) + (ask_volume or 0))
            rolling_fair = (1.0 - self.config.micro_weight) * rolling_fair + self.config.micro_weight * micro_price

        if signal > 0:
            return max(rolling_fair, self.config.fair_anchor)
        if signal < 0:
            return min(rolling_fair, self.config.fair_anchor)
        return 0.65 * rolling_fair + 0.35 * self.config.fair_anchor

    def hard_take_liquidity(self, fair_value: float, signal: int) -> None:
        if self.order_depth is None or signal == 0:
            return

        reservation_price = fair_value - self.pending_position * self.config.inventory_skew
        if signal > 0:
            limit_price = floor(reservation_price - self.config.hard_edge)
            remaining = min(self.config.hard_take_size, self.remaining_buy_capacity())
            for price in sorted(self.order_depth.sell_orders):
                if remaining <= 0 or price > limit_price:
                    break
                volume = min(abs(self.order_depth.sell_orders[price]), remaining)
                self.add_buy(price, volume)
                remaining -= volume

        if signal < 0:
            limit_price = ceil(reservation_price + self.config.hard_edge)
            remaining = min(self.config.hard_take_size, self.remaining_sell_capacity())
            for price in sorted(self.order_depth.buy_orders, reverse=True):
                if remaining <= 0 or price < limit_price:
                    break
                volume = min(self.order_depth.buy_orders[price], remaining)
                self.add_sell(price, volume)
                remaining -= volume

    def middle_take_liquidity(self, fair_value: float) -> None:
        if self.order_depth is None or not self.config.middle_take_enabled:
            return

        reservation_price = fair_value - self.pending_position * self.config.inventory_skew
        buy_threshold = floor(reservation_price - self.config.take_edge)
        sell_threshold = ceil(reservation_price + self.config.take_edge)

        for price in sorted(self.order_depth.sell_orders):
            if price > buy_threshold:
                break
            volume = min(abs(self.order_depth.sell_orders[price]), self.config.quote_size)
            self.add_buy(price, volume)

        for price in sorted(self.order_depth.buy_orders, reverse=True):
            if price < sell_threshold:
                break
            volume = min(self.order_depth.buy_orders[price], self.config.quote_size)
            self.add_sell(price, volume)

    def quote_size(self, signal: int) -> int:
        if signal != 0:
            return self.config.max_quote_size
        inventory_ratio = abs(self.pending_position) / VOUCHER_POSITION_LIMIT
        size = round(self.config.quote_size * (1.0 - 0.35 * inventory_ratio))
        return max(1, min(self.config.max_quote_size, size))

    def make_liquidity(self, fair_value: float, signal: int) -> None:
        best_bid, _, best_ask, _ = top_of_book(self.order_depth)
        if best_bid is None or best_ask is None:
            return

        reservation_price = fair_value - self.pending_position * self.config.inventory_skew
        inventory_ratio = abs(self.pending_position) / VOUCHER_POSITION_LIMIT
        directional_push = signal * 5.0
        half_spread = self.config.base_half_spread + 2.0 * inventory_ratio

        bid_quote: Optional[int] = floor(reservation_price - half_spread + directional_push)
        ask_quote: Optional[int] = ceil(reservation_price + half_spread + directional_push)

        if signal > 0:
            bid_quote = max(bid_quote, best_bid + 1)
            ask_quote = None if self.pending_position <= 0 else max(ask_quote, best_ask)
        elif signal < 0:
            ask_quote = min(ask_quote, best_ask - 1)
            bid_quote = None if self.pending_position >= 0 else min(bid_quote, best_bid)
        else:
            bid_quote = max(bid_quote, best_bid + 1)
            ask_quote = min(ask_quote, best_ask - 1)

        if bid_quote is not None:
            bid_quote = min(bid_quote, best_ask - self.config.min_edge)
        if ask_quote is not None:
            ask_quote = max(ask_quote, best_bid + self.config.min_edge)

        if self.pending_position > VOUCHER_POSITION_LIMIT - self.config.quote_size:
            bid_quote = None
        if self.pending_position < -VOUCHER_POSITION_LIMIT + self.config.quote_size:
            ask_quote = None

        if bid_quote is not None and ask_quote is not None and bid_quote >= ask_quote:
            center = round(reservation_price)
            bid_quote = center - self.config.min_edge
            ask_quote = center + self.config.min_edge

        size = self.quote_size(signal)
        front_size = ceil(size * 0.80)
        back_size = size - front_size

        if bid_quote is not None:
            self.add_buy(bid_quote, front_size)
            if back_size > 0:
                self.add_buy(bid_quote - self.config.second_level_offset, back_size)
        if ask_quote is not None:
            self.add_sell(ask_quote, front_size)
            if back_size > 0:
                self.add_sell(ask_quote + self.config.second_level_offset, back_size)

    def build_orders(self) -> List[Order]:
        history = self.update_history()
        if not history:
            return self.orders

        current_mid = history[-1]
        signal = self.regime_signal(current_mid)
        fair_value = self.fair_value(history, current_mid, signal)
        if signal == 0:
            self.middle_take_liquidity(fair_value)
        else:
            self.hard_take_liquidity(fair_value, signal)
        self.make_liquidity(fair_value, signal)
        return self.orders


class VEV4000BandMarketMaker(VoucherBandMarketMaker):
    config = VoucherBandConfig(VEV_4000, 1220.0, 1275.0, 1247.5, 24, 42, 90, 8.5, 2, 7.0, 2.0, 0.055, 2)


class VEV4500BandMarketMaker(VoucherBandMarketMaker):
    config = VoucherBandConfig(VEV_4500, 720.0, 775.0, 747.5, 24, 42, 90, 6.5, 2, 6.0, 2.0, 0.055, 2)


class VEV5000BandMarketMaker(VoucherBandMarketMaker):
    config = VoucherBandConfig(VEV_5000, 220.0, 278.0, 251.0, 30, 50, 90, 3.0, 1, 3.5, 1.0, 0.060, 1)


class VEV5100BandMarketMaker(VoucherBandMarketMaker):
    config = VoucherBandConfig(VEV_5100, 133.0, 185.0, 161.0, 30, 50, 90, 2.4, 1, 2.7, 1.0, 0.065, 1)


class VEV5200BandMarketMaker(VoucherBandMarketMaker):
    config = VoucherBandConfig(VEV_5200, 66.0, 109.0, 89.0, 24, 42, 60, 2.0, 1, 2.5, 1.0, 0.070, 1, middle_take_enabled=False)


class VEV5300BandMarketMaker(VoucherBandMarketMaker):
    config = VoucherBandConfig(VEV_5300, 25.0, 55.0, 41.0, 12, 28, 35, 1.5, 1, 2.0, 0.75, 0.075, 1, middle_take_enabled=False)


class VEV5400BandMarketMaker(VoucherBandMarketMaker):
    config = VoucherBandConfig(VEV_5400, 6.0, 19.0, 12.5, 2, 6, 5, 1.0, 1, 1.5, 0.75, 0.080, 1, middle_take_enabled=False)


class VEV5500BandMarketMaker(VoucherBandMarketMaker):
    config = VoucherBandConfig(VEV_5500, 1.5, 8.0, 4.7, 1, 4, 4, 1.0, 1, 1.5, 0.75, 0.090, 1, middle_take_enabled=False)


class VEV6000BandMarketMaker(VoucherBandMarketMaker):
    config = VoucherBandConfig(VEV_6000, 0.0, 1.0, 0.5, 44, 70, 0, 0.5, 1, 1.0, 0.5, 0.020, 1, middle_take_enabled=False)


class VEV6500BandMarketMaker(VoucherBandMarketMaker):
    config = VoucherBandConfig(VEV_6500, 0.0, 1.0, 0.5, 44, 70, 0, 0.5, 1, 1.0, 0.5, 0.020, 1, middle_take_enabled=False)


class Trader:
    @staticmethod
    def decode_trader_data(trader_data: str) -> Dict[str, str]:
        if not trader_data:
            return {}
        try:
            decoded = json.loads(trader_data)
            return decoded if isinstance(decoded, dict) else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def encode_trader_data(trader_state: Dict[str, str]) -> str:
        try:
            return json.dumps(trader_state, separators=(",", ":"))
        except (TypeError, ValueError):
            return ""

    def run(self, state: TradingState):
        trader_state = self.decode_trader_data(getattr(state, "traderData", ""))
        orders: Dict[str, List[Order]] = {}
        if HYDROGEL_PACK in state.order_depths:
            hydro_orders = HydroMeanReversionMarketMaker(state, trader_state).build_orders()
            if hydro_orders:
                orders[HYDROGEL_PACK] = hydro_orders
        if VELVETFRUIT_EXTRACT in state.order_depths:
            fruit_orders = FruitMomentumMarketMaker(state, trader_state).build_orders()
            if fruit_orders:
                orders[VELVETFRUIT_EXTRACT] = fruit_orders
        for voucher_strategy in (
            VEV4000BandMarketMaker,
            VEV4500BandMarketMaker,
            VEV5000BandMarketMaker,
            VEV5100BandMarketMaker,
            VEV5200BandMarketMaker,
            VEV5300BandMarketMaker,
            VEV5400BandMarketMaker,
            VEV5500BandMarketMaker,
            VEV6000BandMarketMaker,
            VEV6500BandMarketMaker,
        ):
            symbol = voucher_strategy.config.symbol
            if symbol not in state.order_depths:
                continue
            vev_orders = voucher_strategy(state, trader_state).build_orders()
            if vev_orders:
                orders[symbol] = vev_orders
        return orders, 0, self.encode_trader_data(trader_state)
