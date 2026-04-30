from dataclasses import dataclass, field
import json
from math import ceil, erf, floor, log, sqrt
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
        timestamp: int = 0


HYDROGEL_PACK = "HYDROGEL_PACK"
VELVETFRUIT_EXTRACT = "VELVETFRUIT_EXTRACT"

VOUCHER_STRIKES = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    "VEV_6000": 6000,
    "VEV_6500": 6500,
}

POSITION_LIMITS = {
    HYDROGEL_PACK: 200,
    VELVETFRUIT_EXTRACT: 200,
    **{symbol: 300 for symbol in VOUCHER_STRIKES},
}

HYDRO_POSITION_LIMIT = 60
HYDRO_QUOTE_SIZE = 10
HYDRO_SIGNAL_SIZE_BOOST = 4
HYDRO_SECOND_LEVEL_OFFSET = 2
HYDRO_BASE_HALF_SPREAD = 7.4
HYDRO_MIN_EDGE = 2
HYDRO_TAKE_EDGE = 9.0
HYDRO_INVENTORY_SKEW_PER_UNIT = 0.28
HYDRO_IMBALANCE_ADJUSTMENT = 1.6
HYDRO_MAX_IMBALANCE_SHIFT = 2.5
HYDRO_HISTORY_LENGTH = 100
HYDRO_FAST_ALPHA = 0.18
HYDRO_SLOW_ALPHA = 0.045
HYDRO_MICRO_WEIGHT = 0.62
HYDRO_TREND_WEIGHT = 0.14
HYDRO_MAX_TREND_SHIFT = 2.5
HYDRO_REVERSION_LOOKBACK = 20
HYDRO_REVERSION_WEIGHT = 0.20
HYDRO_MAX_REVERSION_SHIFT = 3.0
HYDRO_UNWIND_TRIGGER = 35
HYDRO_UNWIND_SIZE = 8
HYDRO_UNWIND_EDGE = 4.0

FRUIT_QUOTE_SIZE = 14
FRUIT_SIGNAL_SIZE_BOOST = 8
FRUIT_SECOND_LEVEL_OFFSET = 1
FRUIT_BASE_HALF_SPREAD = 3.2
FRUIT_MIN_EDGE = 1
FRUIT_TAKE_EDGE = 4.0
FRUIT_INVENTORY_SKEW_PER_UNIT = 0.14
FRUIT_HISTORY_LENGTH = 120
FRUIT_FAST_ALPHA = 0.18
FRUIT_SLOW_ALPHA = 0.035
FRUIT_MICRO_ALPHA = 0.55
FRUIT_TREND_WEIGHT = 0.18
FRUIT_MAX_TREND_SHIFT = 2.0
FRUIT_REVERSION_WEIGHT = 0.20
FRUIT_MAX_REVERSION_SHIFT = 2.0
FRUIT_IMBALANCE_WEIGHT = 0.35
FRUIT_MAX_IMBALANCE_SHIFT = 0.6

VOUCHER_QUOTE_SIZE = 28
VOUCHER_SIGNAL_SIZE_BOOST = 16
VOUCHER_SECOND_LEVEL_OFFSET = 1
VOUCHER_TAKE_EDGE = 2.75
VOUCHER_MIN_EDGE = 2
VOUCHER_BASE_HALF_SPREAD = 2.7
VOUCHER_INVENTORY_SKEW_PER_UNIT = 0.075
VOUCHER_MIN_THEO_SPREAD = 3.0
VOUCHER_DEFAULT_VOL = 0.292
VOUCHER_VOL_FLOOR = 0.01
VOUCHER_VOL_CAP = 1.00
VOUCHER_HISTORY_LENGTH = 120

VOUCHER_VOL_BY_STRIKE = {
    4000: 0.010,
    4500: 0.010,
    5000: 0.290,
    5100: 0.288,
    5200: 0.292,
    5300: 0.296,
    5400: 0.276,
    5500: 0.300,
    6000: 0.010,
    6500: 0.010,
}

VOUCHER_RISK_BY_STRIKE = {
    4000: {"quote_size": 28, "take_edge": 2.75, "edge_multiplier": 1.00, "front_ratio": 0.86},
    4500: {"quote_size": 28, "take_edge": 2.75, "edge_multiplier": 1.00, "front_ratio": 0.86},
    5000: {"quote_size": 28, "take_edge": 2.75, "edge_multiplier": 1.00, "front_ratio": 0.86},
    5100: {"quote_size": 28, "take_edge": 2.75, "edge_multiplier": 1.00, "front_ratio": 0.86},
    5200: {"quote_size": 28, "take_edge": 2.75, "edge_multiplier": 1.00, "front_ratio": 0.86},
    5300: {"quote_size": 28, "take_edge": 2.75, "edge_multiplier": 1.00, "front_ratio": 0.86},
    5400: {"quote_size": 28, "take_edge": 2.75, "edge_multiplier": 1.00, "front_ratio": 0.86},
    5500: {"quote_size": 28, "take_edge": 2.75, "edge_multiplier": 1.00, "front_ratio": 0.86},
    6000: {"quote_size": 28, "take_edge": 2.75, "edge_multiplier": 1.00, "front_ratio": 0.86},
    6500: {"quote_size": 28, "take_edge": 2.75, "edge_multiplier": 1.00, "front_ratio": 0.86},
}

ROUND_3_START_TTE_DAYS = 5.0
TRADING_DAY_TIMESTAMPS = 1_000_000


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def black_scholes_call(spot: float, strike: float, tte_days: float, volatility: float) -> float:
    intrinsic = max(0.0, spot - strike)
    if spot <= 0 or strike <= 0 or tte_days <= 0 or volatility <= 0:
        return intrinsic

    tte_years = max(tte_days / 365.0, 1e-6)
    sigma_sqrt_t = volatility * sqrt(tte_years)
    if sigma_sqrt_t <= 1e-9:
        return intrinsic

    d1 = (log(spot / strike) + 0.5 * volatility * volatility * tte_years) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return max(intrinsic, spot * normal_cdf(d1) - strike * normal_cdf(d2))


def compute_order_book_imbalance(bid_volume: Optional[int], ask_volume: Optional[int]) -> float:
    if bid_volume is None or ask_volume is None:
        return 0.0

    total = bid_volume + ask_volume
    if total <= 0:
        return 0.0

    return clamp((bid_volume - ask_volume) / total, -1.0, 1.0)


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
        self.front_ratio = 0.78

    def best_bid_ask(self) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        return top_of_book(self.order_depth)

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
        if price < 0:
            return 0
        size = min(max(0, int(volume)), self.remaining_buy_capacity())
        if size > 0:
            self.orders.append(Order(self.symbol, int(price), size))
            self.pending_position += size
        return size

    def add_sell(self, price: int, volume: int) -> int:
        if price < 0:
            return 0
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
        base_size = max(1, round(base_size * (1 - 0.34 * inventory_ratio)))

        buy_size = min(base_size, self.remaining_buy_capacity())
        sell_size = min(base_size, self.remaining_sell_capacity())

        if self.pending_position > 0:
            buy_size = min(buy_size, max(1, base_size - ceil(abs(self.pending_position) / 18)))
        elif self.pending_position < 0:
            sell_size = min(sell_size, max(1, base_size - ceil(abs(self.pending_position) / 18)))

        return max(0, buy_size), max(0, sell_size)

    def quote_two_levels(
        self,
        bid_quote: Optional[int],
        ask_quote: Optional[int],
        signal_strength: float,
    ) -> None:
        if self.quote_size <= 0:
            return

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


class HydroMarketMaker(Product):
    def __init__(self, state: TradingState, trader_state: Dict[str, str]) -> None:
        super().__init__(
            HYDROGEL_PACK,
            state,
            trader_state,
            HYDRO_POSITION_LIMIT,
            HYDRO_QUOTE_SIZE,
            HYDRO_SIGNAL_SIZE_BOOST,
            HYDRO_SECOND_LEVEL_OFFSET,
        )
        self.front_ratio = 0.72

    def history_key(self) -> str:
        return "hist:hydro_mm"

    def parse_history(self) -> List[float]:
        raw_history = self.trader_state.get(self.history_key(), "")
        history: List[float] = []
        for value in raw_history.split(","):
            if not value:
                continue
            try:
                history.append(float(value))
            except ValueError:
                continue
        return history

    def save_history(self, history: List[float]) -> None:
        self.trader_state[self.history_key()] = ",".join(f"{price:.2f}" for price in history[-HYDRO_HISTORY_LENGTH:])

    @staticmethod
    def ema(history: List[float], alpha: float) -> float:
        value = history[0]
        for price in history[1:]:
            value = (1 - alpha) * value + alpha * price
        return value

    def update_history(self) -> List[float]:
        history = self.parse_history()
        current_mid = mid_price(self.order_depth)
        if current_mid is not None:
            history.append(current_mid)
        history = history[-HYDRO_HISTORY_LENGTH:]
        self.save_history(history)
        return history

    def fair_value(
        self,
        history: List[float],
        best_bid: Optional[int],
        best_ask: Optional[int],
        bid_volume: Optional[int],
        ask_volume: Optional[int],
    ) -> tuple[float, float]:
        if not history:
            return 0.0, 0.0

        fast_ema = self.ema(history, HYDRO_FAST_ALPHA)
        slow_ema = self.ema(history, HYDRO_SLOW_ALPHA)
        fair_value = 0.45 * fast_ema + 0.55 * slow_ema
        current_mid = history[-1]

        if best_bid is not None and best_ask is not None and best_bid < best_ask:
            micro_price = (
                best_bid * (ask_volume or 0) + best_ask * (bid_volume or 0)
            ) / max(1, (bid_volume or 0) + (ask_volume or 0))
            fair_value = (1.0 - HYDRO_MICRO_WEIGHT) * fair_value + HYDRO_MICRO_WEIGHT * micro_price

        trend_shift = clamp(
            (fast_ema - slow_ema) * HYDRO_TREND_WEIGHT,
            -HYDRO_MAX_TREND_SHIFT,
            HYDRO_MAX_TREND_SHIFT,
        )
        recent_mean = sum(history[-HYDRO_REVERSION_LOOKBACK:]) / min(len(history), HYDRO_REVERSION_LOOKBACK)
        reversion_shift = clamp(
            (recent_mean - current_mid) * HYDRO_REVERSION_WEIGHT,
            -HYDRO_MAX_REVERSION_SHIFT,
            HYDRO_MAX_REVERSION_SHIFT,
        )

        imbalance = compute_order_book_imbalance(bid_volume, ask_volume)
        imbalance_shift = clamp(
            imbalance * HYDRO_IMBALANCE_ADJUSTMENT,
            -HYDRO_MAX_IMBALANCE_SHIFT,
            HYDRO_MAX_IMBALANCE_SHIFT,
        )

        position_pressure = abs(self.pending_position) / max(1, HYDRO_POSITION_LIMIT)
        signal_strength = max(
            abs(trend_shift) / max(1e-9, HYDRO_MAX_TREND_SHIFT),
            abs(reversion_shift) / max(1e-9, HYDRO_MAX_REVERSION_SHIFT),
            abs(imbalance) * 0.6,
            position_pressure,
        )
        return fair_value + trend_shift + reversion_shift + imbalance_shift, clamp(signal_strength, 0.0, 1.0)

    def take_liquidity(self, fair_value: float) -> None:
        reservation_price = fair_value - self.pending_position * HYDRO_INVENTORY_SKEW_PER_UNIT
        buy_threshold = reservation_price - HYDRO_TAKE_EDGE
        sell_threshold = reservation_price + HYDRO_TAKE_EDGE

        for ask_price, ask_volume in self.ordered_sells():
            reduces_short = self.pending_position < -HYDRO_UNWIND_TRIGGER and ask_price <= ceil(fair_value + HYDRO_UNWIND_EDGE)
            has_real_edge = ask_price <= floor(buy_threshold)
            if has_real_edge or reduces_short:
                max_size = HYDRO_UNWIND_SIZE if reduces_short and not has_real_edge else ask_volume
                self.add_buy(ask_price, min(ask_volume, max_size))
            else:
                break

        for bid_price, bid_volume in self.ordered_buys():
            reduces_long = self.pending_position > HYDRO_UNWIND_TRIGGER and bid_price >= floor(fair_value - HYDRO_UNWIND_EDGE)
            has_real_edge = bid_price >= ceil(sell_threshold)
            if has_real_edge or reduces_long:
                max_size = HYDRO_UNWIND_SIZE if reduces_long and not has_real_edge else bid_volume
                self.add_sell(bid_price, min(bid_volume, max_size))
            else:
                break

    def make_quotes(
        self,
        fair_value: float,
        best_bid: Optional[int],
        best_ask: Optional[int],
    ) -> tuple[Optional[int], Optional[int]]:
        reservation_price = fair_value - self.pending_position * HYDRO_INVENTORY_SKEW_PER_UNIT
        inventory_ratio = abs(self.pending_position) / max(1, HYDRO_POSITION_LIMIT)
        half_spread = HYDRO_BASE_HALF_SPREAD + 2.0 * inventory_ratio
        bid_quote: Optional[int] = floor(reservation_price - half_spread)
        ask_quote: Optional[int] = ceil(reservation_price + half_spread)

        if best_bid is not None and bid_quote is not None:
            bid_quote = min(max(bid_quote, best_bid + 1), floor(reservation_price - HYDRO_MIN_EDGE))
        if best_ask is not None and ask_quote is not None:
            ask_quote = max(min(ask_quote, best_ask - 1), ceil(reservation_price + HYDRO_MIN_EDGE))

        if best_ask is not None and bid_quote is not None:
            bid_quote = min(bid_quote, best_ask - HYDRO_MIN_EDGE)
        if best_bid is not None and ask_quote is not None:
            ask_quote = max(ask_quote, best_bid + HYDRO_MIN_EDGE)

        if self.pending_position > HYDRO_POSITION_LIMIT - HYDRO_QUOTE_SIZE:
            bid_quote = None
        elif self.pending_position < -HYDRO_POSITION_LIMIT + HYDRO_QUOTE_SIZE:
            ask_quote = None

        if bid_quote is not None and ask_quote is not None and bid_quote >= ask_quote:
            center = round(reservation_price)
            bid_quote = center - HYDRO_MIN_EDGE
            ask_quote = center + HYDRO_MIN_EDGE

        return bid_quote, ask_quote

    def build_orders(self) -> List[Order]:
        best_bid, bid_volume, best_ask, ask_volume = self.best_bid_ask()
        history = self.update_history()
        if not history:
            return self.orders

        fair_value, signal_strength = self.fair_value(history, best_bid, best_ask, bid_volume, ask_volume)
        self.take_liquidity(fair_value)

        best_bid, bid_volume, best_ask, ask_volume = self.best_bid_ask()
        fair_value, signal_strength = self.fair_value(history, best_bid, best_ask, bid_volume, ask_volume)
        bid_quote, ask_quote = self.make_quotes(fair_value, best_bid, best_ask)
        self.quote_two_levels(bid_quote, ask_quote, signal_strength)
        return self.orders


class FruitMarketMaker(Product):
    def __init__(self, state: TradingState, trader_state: Dict[str, str]) -> None:
        super().__init__(
            VELVETFRUIT_EXTRACT,
            state,
            trader_state,
            POSITION_LIMITS[VELVETFRUIT_EXTRACT],
            FRUIT_QUOTE_SIZE,
            FRUIT_SIGNAL_SIZE_BOOST,
            FRUIT_SECOND_LEVEL_OFFSET,
        )
        self.front_ratio = 0.72

    def history_key(self) -> str:
        return "hist:fruit_mm"

    def parse_history(self) -> List[float]:
        raw_history = self.trader_state.get(self.history_key(), "")
        history: List[float] = []
        for value in raw_history.split(","):
            if not value:
                continue
            try:
                history.append(float(value))
            except ValueError:
                continue
        return history

    def save_history(self, history: List[float]) -> None:
        self.trader_state[self.history_key()] = ",".join(f"{price:.2f}" for price in history[-FRUIT_HISTORY_LENGTH:])

    @staticmethod
    def ema(history: List[float], alpha: float) -> float:
        value = history[0]
        for price in history[1:]:
            value = (1 - alpha) * value + alpha * price
        return value

    def update_history(self) -> List[float]:
        history = self.parse_history()
        current_mid = mid_price(self.order_depth)
        if current_mid is not None:
            history.append(current_mid)
        history = history[-FRUIT_HISTORY_LENGTH:]
        self.save_history(history)
        return history

    def fair_value(
        self,
        history: List[float],
        best_bid: Optional[int],
        best_ask: Optional[int],
        bid_volume: Optional[int],
        ask_volume: Optional[int],
    ) -> tuple[float, float]:
        fast_ema = self.ema(history, FRUIT_FAST_ALPHA)
        slow_ema = self.ema(history, FRUIT_SLOW_ALPHA)
        fair_value = 0.45 * fast_ema + 0.55 * slow_ema

        if best_bid is not None and best_ask is not None and best_bid < best_ask:
            micro_price = (
                best_bid * (ask_volume or 0) + best_ask * (bid_volume or 0)
            ) / max(1, (bid_volume or 0) + (ask_volume or 0))
            fair_value = (1 - FRUIT_MICRO_ALPHA) * fair_value + FRUIT_MICRO_ALPHA * micro_price

        trend_shift = clamp(
            (fast_ema - slow_ema) * FRUIT_TREND_WEIGHT,
            -FRUIT_MAX_TREND_SHIFT,
            FRUIT_MAX_TREND_SHIFT,
        )
        anchor = sum(history[-18:]) / min(len(history), 18)
        reversion_shift = clamp(
            (anchor - fair_value) * FRUIT_REVERSION_WEIGHT,
            -FRUIT_MAX_REVERSION_SHIFT,
            FRUIT_MAX_REVERSION_SHIFT,
        )
        imbalance = compute_order_book_imbalance(bid_volume, ask_volume)
        imbalance_shift = clamp(
            imbalance * FRUIT_IMBALANCE_WEIGHT,
            -FRUIT_MAX_IMBALANCE_SHIFT,
            FRUIT_MAX_IMBALANCE_SHIFT,
        )

        signal_strength = max(
            abs(trend_shift) / max(1e-9, FRUIT_MAX_TREND_SHIFT),
            abs(reversion_shift) / max(1e-9, FRUIT_MAX_REVERSION_SHIFT),
            abs(imbalance) * 0.4,
        )
        return fair_value + trend_shift + reversion_shift + imbalance_shift, clamp(signal_strength, 0.0, 1.0)

    def take_liquidity(self, fair_value: float) -> None:
        reservation_price = fair_value - self.pending_position * FRUIT_INVENTORY_SKEW_PER_UNIT
        buy_threshold = reservation_price - FRUIT_TAKE_EDGE
        sell_threshold = reservation_price + FRUIT_TAKE_EDGE

        for ask_price, ask_volume in self.ordered_sells():
            if ask_price <= floor(buy_threshold):
                self.add_buy(ask_price, min(ask_volume, FRUIT_QUOTE_SIZE))
            else:
                break

        for bid_price, bid_volume in self.ordered_buys():
            if bid_price >= ceil(sell_threshold):
                self.add_sell(bid_price, min(bid_volume, FRUIT_QUOTE_SIZE))
            else:
                break

    def make_quotes(
        self,
        fair_value: float,
        best_bid: Optional[int],
        best_ask: Optional[int],
        signal_strength: float,
    ) -> tuple[Optional[int], Optional[int]]:
        reservation_price = fair_value - self.pending_position * FRUIT_INVENTORY_SKEW_PER_UNIT
        directional_push = signal_strength * 0.8
        bid_quote = floor(reservation_price - FRUIT_BASE_HALF_SPREAD + directional_push)
        ask_quote = ceil(reservation_price + FRUIT_BASE_HALF_SPREAD - directional_push)

        if best_bid is not None:
            bid_quote = max(bid_quote, best_bid + 1)
        if best_ask is not None:
            ask_quote = min(ask_quote, best_ask - 1)

        if best_ask is not None:
            bid_quote = min(bid_quote, best_ask - FRUIT_MIN_EDGE)
        if best_bid is not None:
            ask_quote = max(ask_quote, best_bid + FRUIT_MIN_EDGE)

        if bid_quote >= ask_quote:
            center = round(reservation_price)
            bid_quote = center - FRUIT_MIN_EDGE
            ask_quote = center + FRUIT_MIN_EDGE

        return bid_quote, ask_quote

    def build_orders(self) -> List[Order]:
        best_bid, bid_volume, best_ask, ask_volume = self.best_bid_ask()
        history = self.update_history()
        if not history:
            return self.orders

        fair_value, signal_strength = self.fair_value(history, best_bid, best_ask, bid_volume, ask_volume)
        self.take_liquidity(fair_value)

        best_bid, bid_volume, best_ask, ask_volume = self.best_bid_ask()
        fair_value, signal_strength = self.fair_value(history, best_bid, best_ask, bid_volume, ask_volume)
        bid_quote, ask_quote = self.make_quotes(fair_value, best_bid, best_ask, signal_strength)
        self.quote_two_levels(bid_quote, ask_quote, signal_strength)
        return self.orders


class VoucherBlackScholesProduct(Product):
    def __init__(
        self,
        symbol: str,
        strike: int,
        state: TradingState,
        trader_state: Dict[str, str],
        spot: float,
        volatility: float,
        tte_days: float,
    ) -> None:
        config = VOUCHER_RISK_BY_STRIKE.get(strike, {})
        super().__init__(
            symbol,
            state,
            trader_state,
            POSITION_LIMITS[symbol],
            int(config.get("quote_size", VOUCHER_QUOTE_SIZE)),
            VOUCHER_SIGNAL_SIZE_BOOST,
            VOUCHER_SECOND_LEVEL_OFFSET,
        )
        self.strike = strike
        self.spot = spot
        self.volatility = volatility
        self.tte_days = tte_days
        self.config = config
        self.front_ratio = float(config.get("front_ratio", 0.86))

    def theoretical_value(self) -> tuple[float, float]:
        model_value = black_scholes_call(
            self.spot,
            float(self.strike),
            self.tte_days,
            self.volatility,
        )
        option_mid = mid_price(self.order_depth)
        if option_mid is None:
            return model_value, 1.0

        blended_value = 0.95 * model_value + 0.05 * option_mid
        disagreement = abs(model_value - option_mid)
        signal_strength = clamp(disagreement / max(4.0, 0.025 * max(1.0, blended_value)), 0.0, 1.0)
        return blended_value, signal_strength

    def take_liquidity(self, fair_value: float) -> None:
        inventory_shift = self.pending_position * VOUCHER_INVENTORY_SKEW_PER_UNIT
        reservation_price = fair_value - inventory_shift
        take_edge = float(self.config.get("take_edge", VOUCHER_TAKE_EDGE))
        edge = max(take_edge, 0.013 * max(1.0, fair_value))
        buy_threshold = reservation_price - edge
        sell_threshold = reservation_price + edge

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
        reservation_price = max(0.0, fair_value - self.pending_position * VOUCHER_INVENTORY_SKEW_PER_UNIT)
        edge_multiplier = float(self.config.get("edge_multiplier", 1.0))
        half_spread = edge_multiplier * max(
            VOUCHER_BASE_HALF_SPREAD,
            VOUCHER_MIN_THEO_SPREAD,
            fair_value * 0.013,
        )
        bid_quote: Optional[int] = max(0, floor(reservation_price - half_spread))
        ask_quote: Optional[int] = max(1, ceil(reservation_price + half_spread))

        if fair_value < 1.0:
            bid_quote = None

        if best_bid is not None and bid_quote is not None:
            bid_quote = max(bid_quote, best_bid + 1)
        if best_ask is not None and ask_quote is not None:
            ask_quote = min(ask_quote, best_ask - 1)

        if best_ask is not None and bid_quote is not None:
            bid_quote = min(bid_quote, best_ask - VOUCHER_MIN_EDGE)
        if best_bid is not None and ask_quote is not None:
            ask_quote = max(ask_quote, best_bid + VOUCHER_MIN_EDGE)

        if bid_quote is not None:
            bid_quote = max(0, bid_quote)
        if ask_quote is not None:
            ask_quote = max(1, ask_quote)

        if bid_quote is not None and ask_quote is not None and bid_quote >= ask_quote:
            center = max(0, round(reservation_price))
            bid_quote = max(0, center - VOUCHER_MIN_EDGE)
            ask_quote = max(bid_quote + VOUCHER_MIN_EDGE, center + VOUCHER_MIN_EDGE)

        return bid_quote, ask_quote

    def build_orders(self) -> List[Order]:
        if self.spot <= 0 or self.tte_days <= 0:
            return self.orders

        fair_value, signal_strength = self.theoretical_value()
        self.take_liquidity(fair_value)

        best_bid, _, best_ask, _ = self.best_bid_ask()
        bid_quote, ask_quote = self.make_quotes(fair_value, best_bid, best_ask)
        if best_ask is not None and bid_quote is not None and bid_quote > best_ask - VOUCHER_MIN_EDGE:
            bid_quote = None
        if best_bid is not None and ask_quote is not None and ask_quote < best_bid + VOUCHER_MIN_EDGE:
            ask_quote = None
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
            return json.dumps(trader_state, separators=(",", ":"))
        except (TypeError, ValueError):
            return ""

    @staticmethod
    def parse_float_series(raw_history: str) -> List[float]:
        history: List[float] = []
        for value in raw_history.split(","):
            if not value:
                continue
            try:
                history.append(float(value))
            except ValueError:
                continue
        return history

    @staticmethod
    def estimate_volatility(history: List[float]) -> float:
        if len(history) < 8:
            return VOUCHER_DEFAULT_VOL

        returns: List[float] = []
        for previous, current in zip(history, history[1:]):
            if previous > 0 and current > 0:
                returns.append(log(current / previous))

        if len(returns) < 2:
            return VOUCHER_DEFAULT_VOL

        mean_return = sum(returns) / len(returns)
        variance = sum((value - mean_return) ** 2 for value in returns) / (len(returns) - 1)
        annualized_vol = sqrt(max(0.0, variance)) * sqrt(365.0)
        return clamp(annualized_vol, 0.24, 0.36)

    @staticmethod
    def option_volatility(strike: int, realized_volatility: float) -> float:
        calibrated_volatility = VOUCHER_VOL_BY_STRIKE.get(strike, VOUCHER_DEFAULT_VOL)
        if strike <= 4500 or strike >= 6000:
            return calibrated_volatility
        return clamp(0.85 * calibrated_volatility + 0.15 * realized_volatility, 0.24, 0.36)

    @staticmethod
    def round_3_tte_days(state: TradingState) -> float:
        timestamp = getattr(state, "timestamp", 0) or 0
        intraday_progress = clamp(timestamp / TRADING_DAY_TIMESTAMPS, 0.0, 1.0)
        return max(0.05, ROUND_3_START_TTE_DAYS - intraday_progress)

    def run(self, state: TradingState):
        trader_state = self.decode_trader_data(getattr(state, "traderData", ""))
        orders: Dict[str, List[Order]] = {}

        if HYDROGEL_PACK in state.order_depths:
            orders[HYDROGEL_PACK] = HydroMarketMaker(state, trader_state).build_orders()

        if VELVETFRUIT_EXTRACT in state.order_depths:
            orders[VELVETFRUIT_EXTRACT] = FruitMarketMaker(state, trader_state).build_orders()

        fruit_history = FruitMarketMaker(state, trader_state).parse_history()
        underlying_mid = mid_price(state.order_depths.get(VELVETFRUIT_EXTRACT))
        if underlying_mid is not None:
            underlying_fair = underlying_mid
        elif fruit_history:
            underlying_fair = fruit_history[-1]
        else:
            underlying_fair = 0.0

        realized_volatility = self.estimate_volatility(fruit_history)
        tte_days = self.round_3_tte_days(state)
        for symbol, strike in VOUCHER_STRIKES.items():
            if symbol in state.order_depths and underlying_fair > 0:
                orders[symbol] = VoucherBlackScholesProduct(
                    symbol,
                    strike,
                    state,
                    trader_state,
                    underlying_fair,
                    self.option_volatility(strike, realized_volatility),
                    tte_days,
                ).build_orders()

        return orders, 0, self.encode_trader_data(trader_state)
