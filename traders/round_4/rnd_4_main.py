from dataclasses import dataclass, field
import json
from math import ceil, erf, floor, log, sqrt
from typing import Dict, List, Optional

import numpy as np

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

TRADING_DAY_TIMESTAMPS = 1_000_000
ROUND_4_START_TTE_DAYS = 4.0

DELTA_CONFIG = {
    HYDROGEL_PACK: {
        "position_limit": 90,
        "quote_size": 12,
        "max_take_size": 16,
        "second_level_offset": 2,
        "history_length": 120,
        "fast_alpha": 0.18,
        "slow_alpha": 0.045,
        "micro_weight": 0.60,
        "trend_weight": 0.13,
        "max_trend_shift": 2.5,
        "reversion_lookback": 22,
        "reversion_weight": 0.22,
        "max_reversion_shift": 3.2,
        "imbalance_weight": 1.5,
        "max_imbalance_shift": 2.4,
        "base_half_spread": 7.4,
        "min_edge": 2,
        "take_edge": 8.5,
        "inventory_skew": 0.24,
        "counterparty_decay": 0.76,
        "counterparty_scale": 0.12,
        "max_counterparty_shift": 3.0,
    },
    VELVETFRUIT_EXTRACT: {
        "position_limit": 200,
        "quote_size": 16,
        "max_take_size": 18,
        "second_level_offset": 1,
        "history_length": 140,
        "fast_alpha": 0.20,
        "slow_alpha": 0.040,
        "micro_weight": 0.58,
        "trend_weight": 0.16,
        "max_trend_shift": 2.2,
        "reversion_lookback": 24,
        "reversion_weight": 0.19,
        "max_reversion_shift": 2.2,
        "imbalance_weight": 0.42,
        "max_imbalance_shift": 0.7,
        "base_half_spread": 3.0,
        "min_edge": 1,
        "take_edge": 3.6,
        "inventory_skew": 0.12,
        "counterparty_decay": 0.82,
        "counterparty_scale": 0.18,
        "max_counterparty_shift": 4.0,
    },
}

COUNTERPARTY_SCORES = {
    HYDROGEL_PACK: {
        "Mark 14": 0.34,
        "Mark 38": -0.30,
    },
    VELVETFRUIT_EXTRACT: {
        "Mark 67": 1.15,
        "Mark 49": -1.25,
        "Mark 55": 0.38,
        "Mark 14": -0.55,
        "Mark 22": -0.50,
        "Mark 01": -0.25,
    },
    "VEV_4000": {
        "Mark 38": 0.20,
        "Mark 14": -0.18,
    },
    "VEV_5200": {
        "Mark 22": -0.15,
        "Mark 14": 0.08,
    },
    "VEV_5300": {
        "Mark 22": 0.10,
        "Mark 14": -0.12,
        "Mark 01": -0.08,
    },
}

OPTION_VOL_FLOOR = 0.01
OPTION_VOL_CAP = 3.00
OPTION_MIN_EXTRINSIC = 0.05
OPTION_SMILE_MIN_POINTS = 4
OPTION_IV_THRESHOLD = 0.040
OPTION_PRICE_EDGE = 1.0
OPTION_PASSIVE_EDGE = 1.0
OPTION_MIN_SCORE = 0.98
OPTION_COUNTERPARTY_SCALE = 0.05
OPTION_UNDERLYING_SIGNAL_SCALE = 0.15
DEEP_VOUCHER_SYMBOL = "VEV_4000"
DEEP_VOUCHER_STRIKE = 4000
DEEP_VOUCHER_POSITION_LIMIT = 120
DEEP_VOUCHER_QUOTE_SIZE = 18
DEEP_VOUCHER_SECOND_LEVEL_OFFSET = 2
DEEP_VOUCHER_MIN_EDGE = 2
DEEP_VOUCHER_TAKE_EDGE_RATIO = 0.012
DEEP_VOUCHER_HALF_SPREAD_RATIO = 0.013
DEEP_VOUCHER_INVENTORY_SKEW = 0.080

OPTION_RISK = {
    4000: {"target": 2, "order": 1, "edge": 4.0, "skew": 0.12},
    4500: {"target": 1, "order": 1, "edge": 4.0, "skew": 0.12},
    5000: {"target": 1, "order": 1, "edge": 3.0, "skew": 0.14},
    5100: {"target": 1, "order": 1, "edge": 3.0, "skew": 0.14},
    5200: {"target": 1, "order": 1, "edge": 3.0, "skew": 0.14},
    5300: {"target": 1, "order": 1, "edge": 3.0, "skew": 0.16},
    5400: {"target": 1, "order": 1, "edge": 2.0, "skew": 0.20},
    5500: {"target": 1, "order": 1, "edge": 2.0, "skew": 0.22},
    6000: {"target": 1, "order": 1, "edge": 1.5, "skew": 0.25},
    6500: {"target": 1, "order": 1, "edge": 1.5, "skew": 0.25},
}


class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *args) -> None:
        self.logs += " ".join(map(str, args)) + "\n"

    def flush(self, state: TradingState, orders: Dict[str, List[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )
        max_item_length = max(0, (3750 - base_length) // 3)
        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(getattr(state, "traderData", ""), max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(getattr(state, "listings", {})),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(getattr(state, "own_trades", {})),
            self.compress_trades(getattr(state, "market_trades", {})),
            state.position,
            self.compress_observations(getattr(state, "observations", None)),
        ]

    def compress_listings(self, listings) -> list:
        return [[listing.symbol, listing.product, listing.denomination] for listing in listings.values()]

    def compress_order_depths(self, order_depths) -> dict:
        return {symbol: [depth.buy_orders, depth.sell_orders] for symbol, depth in order_depths.items()}

    def compress_trades(self, trades) -> list:
        return [
            [trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp]
            for symbol_trades in trades.values()
            for trade in symbol_trades
        ]

    def compress_observations(self, observations) -> list:
        if observations is None:
            return [{}, {}]
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: Dict[str, List[Order]]) -> list:
        return [[order.symbol, order.price, order.quantity] for symbol_orders in orders.values() for order in symbol_orders]

    def to_json(self, value) -> str:
        return json.dumps(value, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if max_length <= 3:
            return ""
        return value[: max_length - 3] + "..." if len(value) > max_length else value


logger = Logger()


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


def implied_call_volatility(spot: float, strike: float, tte_days: float, option_price: float) -> Optional[float]:
    intrinsic = max(0.0, spot - strike)
    if (
        spot <= 0
        or strike <= 0
        or tte_days <= 0
        or option_price <= intrinsic + OPTION_MIN_EXTRINSIC
        or option_price <= OPTION_MIN_EXTRINSIC
    ):
        return None

    low_price = black_scholes_call(spot, strike, tte_days, OPTION_VOL_FLOOR)
    high_price = black_scholes_call(spot, strike, tte_days, OPTION_VOL_CAP)
    if option_price < low_price or option_price > high_price:
        return None

    lower = OPTION_VOL_FLOOR
    upper = OPTION_VOL_CAP
    for _ in range(35):
        midpoint = (lower + upper) / 2
        if black_scholes_call(spot, strike, tte_days, midpoint) < option_price:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2


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
    if bid_volume is None or ask_volume is None:
        return 0.0
    total = bid_volume + ask_volume
    if total <= 0:
        return 0.0
    return clamp((bid_volume - ask_volume) / total, -1.0, 1.0)


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


def ema(history: List[float], alpha: float) -> float:
    value = history[0]
    for price in history[1:]:
        value = (1 - alpha) * value + alpha * price
    return value


def state_market_trades(state: TradingState, symbol: str):
    return getattr(state, "market_trades", {}).get(symbol, [])


def instantaneous_counterparty_signal(state: TradingState, symbol: str) -> float:
    scores = COUNTERPARTY_SCORES.get(symbol, {})
    signal = 0.0
    for trade in state_market_trades(state, symbol):
        quantity_scale = sqrt(max(1, getattr(trade, "quantity", 0)))
        buyer = getattr(trade, "buyer", None)
        seller = getattr(trade, "seller", None)
        if buyer in scores:
            signal += scores[buyer] * quantity_scale
        if seller in scores:
            signal -= scores[seller] * quantity_scale
    return signal


class Product:
    def __init__(
        self,
        symbol: str,
        state: TradingState,
        trader_state: Dict[str, str],
        position_limit: int,
    ) -> None:
        self.symbol = symbol
        self.state = state
        self.trader_state = trader_state
        self.position_limit = position_limit
        self.position = state.position.get(symbol, 0)
        self.pending_position = self.position
        self.order_depth = state.order_depths.get(symbol)
        self.orders: List[Order] = []

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
        size = min(max(0, int(volume)), self.remaining_buy_capacity())
        if price >= 0 and size > 0:
            self.orders.append(Order(self.symbol, int(price), size))
            self.pending_position += size
        return size

    def add_sell(self, price: int, volume: int) -> int:
        size = min(max(0, int(volume)), self.remaining_sell_capacity())
        if price >= 0 and size > 0:
            self.orders.append(Order(self.symbol, int(price), -size))
            self.pending_position -= size
        return size


class DeltaOneMarketMaker(Product):
    def __init__(self, symbol: str, state: TradingState, trader_state: Dict[str, str]) -> None:
        self.config = DELTA_CONFIG[symbol]
        super().__init__(
            symbol,
            state,
            trader_state,
            int(self.config["position_limit"]),
        )

    def history_key(self) -> str:
        return f"hist:{self.symbol}"

    def counterparty_key(self) -> str:
        return f"cp:{self.symbol}"

    def parse_history(self) -> List[float]:
        return parse_float_series(self.trader_state.get(self.history_key(), ""))

    def save_history(self, history: List[float]) -> None:
        history = history[-int(self.config["history_length"]):]
        self.trader_state[self.history_key()] = save_float_series(history, 2)

    def update_history(self) -> List[float]:
        history = self.parse_history()
        current_mid = mid_price(self.order_depth)
        if current_mid is not None:
            history.append(current_mid)
        self.save_history(history)
        return history[-int(self.config["history_length"]):]

    def update_counterparty_signal(self) -> float:
        previous = 0.0
        try:
            previous = float(self.trader_state.get(self.counterparty_key(), "0"))
        except ValueError:
            previous = 0.0

        raw_signal = instantaneous_counterparty_signal(self.state, self.symbol)
        decay = float(self.config["counterparty_decay"])
        signal = decay * previous + (1.0 - decay) * raw_signal
        self.trader_state[self.counterparty_key()] = f"{signal:.4f}"
        return signal

    def fair_value(self, history: List[float]) -> tuple[float, float]:
        best_bid, bid_volume, best_ask, ask_volume = self.best_bid_ask()
        if not history:
            return 0.0, 0.0

        fast_ema = ema(history, float(self.config["fast_alpha"]))
        slow_ema = ema(history, float(self.config["slow_alpha"]))
        current_mid = history[-1]
        fair_value = 0.45 * fast_ema + 0.55 * slow_ema

        if best_bid is not None and best_ask is not None and best_bid < best_ask:
            micro_price = (
                best_bid * (ask_volume or 0) + best_ask * (bid_volume or 0)
            ) / max(1, (bid_volume or 0) + (ask_volume or 0))
            micro_weight = float(self.config["micro_weight"])
            fair_value = (1.0 - micro_weight) * fair_value + micro_weight * micro_price

        trend_shift = clamp(
            (fast_ema - slow_ema) * float(self.config["trend_weight"]),
            -float(self.config["max_trend_shift"]),
            float(self.config["max_trend_shift"]),
        )
        lookback = int(self.config["reversion_lookback"])
        recent_mean = sum(history[-lookback:]) / min(len(history), lookback)
        reversion_shift = clamp(
            (recent_mean - current_mid) * float(self.config["reversion_weight"]),
            -float(self.config["max_reversion_shift"]),
            float(self.config["max_reversion_shift"]),
        )
        imbalance = order_book_imbalance(bid_volume, ask_volume)
        imbalance_shift = clamp(
            imbalance * float(self.config["imbalance_weight"]),
            -float(self.config["max_imbalance_shift"]),
            float(self.config["max_imbalance_shift"]),
        )
        counterparty_signal = self.update_counterparty_signal()
        counterparty_shift = clamp(
            counterparty_signal * float(self.config["counterparty_scale"]),
            -float(self.config["max_counterparty_shift"]),
            float(self.config["max_counterparty_shift"]),
        )

        signal_strength = max(
            abs(trend_shift) / max(1e-9, float(self.config["max_trend_shift"])),
            abs(reversion_shift) / max(1e-9, float(self.config["max_reversion_shift"])),
            abs(imbalance) * 0.45,
            abs(counterparty_shift) / max(1e-9, float(self.config["max_counterparty_shift"])),
            abs(self.pending_position) / max(1, self.position_limit),
        )
        fair = fair_value + trend_shift + reversion_shift + imbalance_shift + counterparty_shift
        return fair, clamp(signal_strength, 0.0, 1.0)

    def dynamic_size(self, signal_strength: float) -> int:
        base_size = int(self.config["quote_size"])
        inventory_ratio = abs(self.pending_position) / max(1, self.position_limit)
        boost = round(base_size * 0.45 * clamp(signal_strength, 0.0, 1.0))
        size = max(1, base_size + boost - ceil(inventory_ratio * base_size * 0.65))
        return size

    def take_liquidity(self, fair_value: float) -> None:
        reservation_price = fair_value - self.pending_position * float(self.config["inventory_skew"])
        buy_threshold = reservation_price - float(self.config["take_edge"])
        sell_threshold = reservation_price + float(self.config["take_edge"])
        max_take_size = int(self.config["max_take_size"])

        for ask_price, ask_volume in self.ordered_sells():
            exits_short = self.pending_position < -0.45 * self.position_limit and ask_price <= fair_value + float(self.config["min_edge"])
            has_edge = ask_price <= floor(buy_threshold)
            if has_edge or exits_short:
                self.add_buy(ask_price, min(ask_volume, max_take_size))
            else:
                break

        for bid_price, bid_volume in self.ordered_buys():
            exits_long = self.pending_position > 0.45 * self.position_limit and bid_price >= fair_value - float(self.config["min_edge"])
            has_edge = bid_price >= ceil(sell_threshold)
            if has_edge or exits_long:
                self.add_sell(bid_price, min(bid_volume, max_take_size))
            else:
                break

    def quote(self, fair_value: float, signal_strength: float) -> None:
        best_bid, _, best_ask, _ = self.best_bid_ask()
        reservation_price = fair_value - self.pending_position * float(self.config["inventory_skew"])
        inventory_ratio = abs(self.pending_position) / max(1, self.position_limit)
        directional_push = signal_strength * 0.75
        half_spread = float(self.config["base_half_spread"]) + inventory_ratio * 1.8
        min_edge = int(self.config["min_edge"])

        bid_quote: Optional[int] = floor(reservation_price - half_spread + directional_push)
        ask_quote: Optional[int] = ceil(reservation_price + half_spread - directional_push)

        if best_bid is not None:
            bid_quote = max(bid_quote, best_bid + 1)
            bid_quote = min(bid_quote, floor(reservation_price - min_edge))
        if best_ask is not None:
            ask_quote = min(ask_quote, best_ask - 1)
            ask_quote = max(ask_quote, ceil(reservation_price + min_edge))
        if best_ask is not None and bid_quote is not None:
            bid_quote = min(bid_quote, best_ask - min_edge)
        if best_bid is not None and ask_quote is not None:
            ask_quote = max(ask_quote, best_bid + min_edge)

        if self.pending_position > self.position_limit - int(self.config["quote_size"]):
            bid_quote = None
        elif self.pending_position < -self.position_limit + int(self.config["quote_size"]):
            ask_quote = None

        if bid_quote is not None and ask_quote is not None and bid_quote >= ask_quote:
            center = round(reservation_price)
            bid_quote = center - min_edge
            ask_quote = center + min_edge

        size = self.dynamic_size(signal_strength)
        front_size = ceil(size * 0.72)
        back_size = size - front_size
        if bid_quote is not None:
            self.add_buy(bid_quote, front_size)
            if back_size > 0:
                self.add_buy(bid_quote - int(self.config["second_level_offset"]), back_size)
        if ask_quote is not None:
            self.add_sell(ask_quote, front_size)
            if back_size > 0:
                self.add_sell(ask_quote + int(self.config["second_level_offset"]), back_size)

    def build_orders(self) -> List[Order]:
        history = self.update_history()
        if not history:
            return self.orders

        fair_value, signal_strength = self.fair_value(history)
        self.take_liquidity(fair_value)
        fair_value, signal_strength = self.fair_value(history)
        self.quote(fair_value, signal_strength)
        return self.orders


class DeepIntrinsicVoucherMarketMaker(Product):
    def __init__(self, state: TradingState, trader_state: Dict[str, str], spot: float) -> None:
        super().__init__(DEEP_VOUCHER_SYMBOL, state, trader_state, DEEP_VOUCHER_POSITION_LIMIT)
        self.spot = spot

    def fair_value(self) -> float:
        return max(0.0, self.spot - DEEP_VOUCHER_STRIKE)

    def take_liquidity(self, fair_value: float) -> None:
        reservation_price = fair_value - self.pending_position * DEEP_VOUCHER_INVENTORY_SKEW
        edge = max(2.0, fair_value * DEEP_VOUCHER_TAKE_EDGE_RATIO)
        buy_threshold = reservation_price - edge
        sell_threshold = reservation_price + edge

        for ask_price, ask_volume in self.ordered_sells():
            exits_short = self.pending_position < -0.55 * self.position_limit and ask_price <= fair_value + DEEP_VOUCHER_MIN_EDGE
            if ask_price <= floor(buy_threshold) or exits_short:
                self.add_buy(ask_price, min(ask_volume, DEEP_VOUCHER_QUOTE_SIZE))
            else:
                break

        for bid_price, bid_volume in self.ordered_buys():
            exits_long = self.pending_position > 0.55 * self.position_limit and bid_price >= fair_value - DEEP_VOUCHER_MIN_EDGE
            if bid_price >= ceil(sell_threshold) or exits_long:
                self.add_sell(bid_price, min(bid_volume, DEEP_VOUCHER_QUOTE_SIZE))
            else:
                break

    def quote(self, fair_value: float) -> None:
        best_bid, _, best_ask, _ = self.best_bid_ask()
        reservation_price = max(0.0, fair_value - self.pending_position * DEEP_VOUCHER_INVENTORY_SKEW)
        half_spread = max(3.0, fair_value * DEEP_VOUCHER_HALF_SPREAD_RATIO)
        bid_quote: Optional[int] = max(0, floor(reservation_price - half_spread))
        ask_quote: Optional[int] = max(1, ceil(reservation_price + half_spread))

        if best_bid is not None:
            bid_quote = max(bid_quote, best_bid + 1)
            bid_quote = min(bid_quote, floor(reservation_price - DEEP_VOUCHER_MIN_EDGE))
        if best_ask is not None:
            ask_quote = min(ask_quote, best_ask - 1)
            ask_quote = max(ask_quote, ceil(reservation_price + DEEP_VOUCHER_MIN_EDGE))

        if best_ask is not None:
            bid_quote = min(bid_quote, best_ask - DEEP_VOUCHER_MIN_EDGE)
        if best_bid is not None:
            ask_quote = max(ask_quote, best_bid + DEEP_VOUCHER_MIN_EDGE)

        if self.pending_position > self.position_limit - DEEP_VOUCHER_QUOTE_SIZE:
            bid_quote = None
        elif self.pending_position < -self.position_limit + DEEP_VOUCHER_QUOTE_SIZE:
            ask_quote = None

        if bid_quote is not None and ask_quote is not None and bid_quote >= ask_quote:
            center = round(reservation_price)
            bid_quote = max(0, center - DEEP_VOUCHER_MIN_EDGE)
            ask_quote = center + DEEP_VOUCHER_MIN_EDGE

        front_size = ceil(DEEP_VOUCHER_QUOTE_SIZE * 0.86)
        back_size = DEEP_VOUCHER_QUOTE_SIZE - front_size
        if bid_quote is not None:
            self.add_buy(bid_quote, front_size)
            if back_size > 0:
                self.add_buy(bid_quote - DEEP_VOUCHER_SECOND_LEVEL_OFFSET, back_size)
        if ask_quote is not None:
            self.add_sell(ask_quote, front_size)
            if back_size > 0:
                self.add_sell(ask_quote + DEEP_VOUCHER_SECOND_LEVEL_OFFSET, back_size)

    def build_orders(self) -> List[Order]:
        if self.spot <= 0 or self.order_depth is None:
            return self.orders
        fair_value = self.fair_value()
        self.take_liquidity(fair_value)
        self.quote(fair_value)
        return self.orders


class OptionSmileTrader:
    def __init__(
        self,
        state: TradingState,
        trader_state: Dict[str, str],
        spot: float,
        underlying_signal: float,
        tte_days: float,
    ) -> None:
        self.state = state
        self.trader_state = trader_state
        self.spot = spot
        self.underlying_signal = underlying_signal
        self.tte_days = tte_days
        self.orders: Dict[str, List[Order]] = {}

    def current_observations(self) -> List[Dict[str, float]]:
        observations: List[Dict[str, float]] = []
        for product, strike in VOUCHER_STRIKES.items():
            order_depth = self.state.order_depths.get(product)
            option_mid = mid_price(order_depth)
            if option_mid is None:
                continue

            implied_volatility = implied_call_volatility(self.spot, float(strike), self.tte_days, option_mid)
            if implied_volatility is None:
                continue

            observations.append(
                {
                    "product": product,
                    "strike": float(strike),
                    "mid": option_mid,
                    "moneyness": log(self.spot / float(strike)),
                    "implied_volatility": implied_volatility,
                }
            )
        return observations

    @staticmethod
    def fit_smile(observations: List[Dict[str, float]]) -> Optional[tuple[float, float, float]]:
        if len(observations) < OPTION_SMILE_MIN_POINTS:
            return None
        moneyness = np.array([obs["moneyness"] for obs in observations], dtype=float)
        implied_volatility = np.array([obs["implied_volatility"] for obs in observations], dtype=float)
        coefficients = np.polyfit(moneyness, implied_volatility, 2)
        return float(coefficients[0]), float(coefficients[1]), float(coefficients[2])

    def option_counterparty_signal(self, product: str) -> float:
        return instantaneous_counterparty_signal(self.state, product)

    def target_for_observation(self, observation: Dict[str, float], coefficients: tuple[float, float, float]) -> tuple[int, float, float]:
        a, b, c = coefficients
        product = str(observation["product"])
        strike = int(observation["strike"])
        moneyness = observation["moneyness"]
        implied_volatility = observation["implied_volatility"]
        option_mid = observation["mid"]
        fitted_iv = clamp(a * moneyness * moneyness + b * moneyness + c, OPTION_VOL_FLOOR, OPTION_VOL_CAP)

        shifted_spot = max(1.0, self.spot + self.underlying_signal * OPTION_UNDERLYING_SIGNAL_SCALE)
        fair_price = black_scholes_call(shifted_spot, float(strike), self.tte_days, fitted_iv)
        price_signal = fair_price - option_mid
        iv_deviation = implied_volatility - fitted_iv
        cp_signal = self.option_counterparty_signal(product)

        iv_score = clamp(-iv_deviation / OPTION_IV_THRESHOLD, -1.0, 1.0)
        price_score = clamp(price_signal / max(OPTION_PRICE_EDGE, OPTION_RISK[strike]["edge"]), -1.0, 1.0)
        cp_score = clamp(cp_signal * OPTION_COUNTERPARTY_SCALE, -0.35, 0.35)
        score = clamp(0.62 * iv_score + 0.32 * price_score + cp_score, -1.0, 1.0)

        if abs(iv_deviation) < OPTION_IV_THRESHOLD and abs(price_signal) < OPTION_RISK[strike]["edge"]:
            score *= 0.35
        if abs(score) < OPTION_MIN_SCORE:
            score = 0.0

        target = round(score * OPTION_RISK[strike]["target"])
        return int(target), fair_price, score

    def add_order(self, product: str, price: int, quantity: int, pending_positions: Dict[str, int]) -> None:
        if quantity == 0 or price < 0:
            return
        position_limit = POSITION_LIMITS[product]
        pending_position = pending_positions.get(product, self.state.position.get(product, 0))
        if quantity > 0:
            quantity = min(quantity, position_limit - pending_position)
        else:
            quantity = -min(-quantity, position_limit + pending_position)
        if quantity == 0:
            return
        self.orders.setdefault(product, []).append(Order(product, int(price), int(quantity)))
        pending_positions[product] = pending_position + quantity

    def trade_product(
        self,
        product: str,
        strike: int,
        fair_price: float,
        target_position: int,
        score: float,
        pending_positions: Dict[str, int],
    ) -> None:
        order_depth = self.state.order_depths.get(product)
        best_bid, bid_volume, best_ask, ask_volume = top_of_book(order_depth)
        if best_bid is None or best_ask is None:
            return

        current_position = pending_positions.get(product, self.state.position.get(product, 0))
        position_gap = target_position - current_position
        risk = OPTION_RISK[strike]
        max_order_size = int(risk["order"] + max(0.0, abs(current_position) / max(1, int(risk["target"]))) * risk["order"])
        max_order_size = max(1, min(max_order_size, int(risk["order"]) * 2))
        edge = float(risk["edge"])

        if position_gap > 0:
            exits_short = current_position < -0.55 * int(risk["target"]) and best_ask <= fair_price + edge
            has_edge = best_ask <= floor(fair_price - edge)
            has_strong_signal = score > 0.90 and best_ask <= fair_price
            if has_edge or exits_short or has_strong_signal:
                size = min(position_gap, max_order_size, ask_volume or 0)
                self.add_order(product, best_ask, size, pending_positions)
        elif position_gap < 0:
            exits_long = current_position > 0.55 * int(risk["target"]) and best_bid >= fair_price - edge
            has_edge = best_bid >= ceil(fair_price + edge)
            has_strong_signal = score < -0.90 and best_bid >= fair_price
            if has_edge or exits_long or has_strong_signal:
                size = min(-position_gap, max_order_size, bid_volume or 0)
                self.add_order(product, best_bid, -size, pending_positions)

        # Voucher smile signals are sparse and noisy. Cross only when there is
        # modeled edge; leave continuous passive making to the delta-one books.

    def build_orders(self) -> Dict[str, List[Order]]:
        observations = self.current_observations()
        coefficients = self.fit_smile(observations)
        if coefficients is None:
            return self.orders

        pending_positions = {product: self.state.position.get(product, 0) for product in VOUCHER_STRIKES}
        if getattr(self.state, "timestamp", 0) % 20_000 == 0:
            logger.print("option_smile", [round(value, 5) for value in coefficients], "spot", round(self.spot, 2))

        for observation in observations:
            product = str(observation["product"])
            strike = int(observation["strike"])
            target_position, fair_price, score = self.target_for_observation(observation, coefficients)
            self.trade_product(product, strike, fair_price, target_position, score, pending_positions)

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
    def round_4_tte_days(state: TradingState) -> float:
        timestamp = getattr(state, "timestamp", 0) or 0
        intraday_progress = clamp(timestamp / TRADING_DAY_TIMESTAMPS, 0.0, 1.0)
        return max(0.05, ROUND_4_START_TTE_DAYS - intraday_progress)

    def run(self, state: TradingState):
        trader_state = self.decode_trader_data(getattr(state, "traderData", ""))
        orders: Dict[str, List[Order]] = {}

        if HYDROGEL_PACK in state.order_depths:
            orders[HYDROGEL_PACK] = DeltaOneMarketMaker(HYDROGEL_PACK, state, trader_state).build_orders()

        if VELVETFRUIT_EXTRACT in state.order_depths:
            orders[VELVETFRUIT_EXTRACT] = DeltaOneMarketMaker(VELVETFRUIT_EXTRACT, state, trader_state).build_orders()

        spot = mid_price(state.order_depths.get(VELVETFRUIT_EXTRACT))
        if spot is None:
            fruit_history = parse_float_series(trader_state.get(f"hist:{VELVETFRUIT_EXTRACT}", ""))
            spot = fruit_history[-1] if fruit_history else 0.0

        underlying_signal = 0.0
        try:
            underlying_signal = float(trader_state.get(f"cp:{VELVETFRUIT_EXTRACT}", "0"))
        except ValueError:
            underlying_signal = 0.0

        if spot > 0:
            if DEEP_VOUCHER_SYMBOL in state.order_depths:
                deep_orders = DeepIntrinsicVoucherMarketMaker(state, trader_state, spot).build_orders()
                if deep_orders:
                    orders[DEEP_VOUCHER_SYMBOL] = deep_orders

            option_orders = OptionSmileTrader(
                state,
                trader_state,
                spot,
                underlying_signal,
                self.round_4_tte_days(state),
            ).build_orders()
            for symbol, symbol_orders in option_orders.items():
                if symbol_orders:
                    orders.setdefault(symbol, []).extend(symbol_orders)

        conversions = 0
        trader_data = self.encode_trader_data(trader_state)
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
