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

DELTA_ONE_CONFIG = {
    HYDROGEL_PACK: {
        "quote_size": 20,
        "signal_size_boost": 8,
        "second_level_offset": 2,
        "base_half_spread": 6.8,
        "min_edge": 2,
        "take_edge": 3.0,
        "inventory_skew": 0.085,
        "micro_weight": 0.34,
        "mean_reversion_weight": 0.24,
    },
    VELVETFRUIT_EXTRACT: {
        "quote_size": 30,
        "signal_size_boost": 16,
        "second_level_offset": 1,
        "base_half_spread": 1.8,
        "min_edge": 1,
        "take_edge": 1.25,
        "inventory_skew": 0.050,
        "micro_weight": 0.62,
        "mean_reversion_weight": 0.40,
    },
}
DELTA_ONE_HISTORY_LENGTH = 90
DELTA_ONE_FAST_ALPHA = 0.22
DELTA_ONE_SLOW_ALPHA = 0.045
DELTA_ONE_MAX_REVERSION_SHIFT = 4.0

VOUCHER_QUOTE_SIZE = 28
VOUCHER_SIGNAL_SIZE_BOOST = 16
VOUCHER_SECOND_LEVEL_OFFSET = 1
VOUCHER_TAKE_EDGE = 2.75
VOUCHER_MIN_EDGE = 2
VOUCHER_BASE_HALF_SPREAD = 2.7
VOUCHER_INVENTORY_SKEW_PER_UNIT = 0.075
VOUCHER_MIN_THEO_SPREAD = 3.0
VOUCHER_VOL_WINDOW = 20
VOUCHER_VOL_FLOOR = 0.01
VOUCHER_VOL_CAP = 1.20
VOUCHER_DEFAULT_VOL = 0.292
VOUCHER_SIGNAL_FAST_WINDOW = 8
VOUCHER_SIGNAL_SLOW_WINDOW = 50
VOUCHER_UNDERLYING_SIGNAL_THRESHOLD = 1.00
VOUCHER_MAX_UNDERLYING_SIGNAL = 3.0
VOUCHER_BASKET_QUOTE_FRACTION = 0.40
VOUCHER_RISK_BY_STRIKE = {
    4000: {"size_multiplier": 1.00, "edge_multiplier": 1.00, "signal_floor": 0.00, "front_ratio": 0.86},
    4500: {"size_multiplier": 1.00, "edge_multiplier": 1.00, "signal_floor": 0.00, "front_ratio": 0.86},
    5000: {"size_multiplier": 1.00, "edge_multiplier": 1.00, "signal_floor": 0.00, "front_ratio": 0.86},
    5100: {"size_multiplier": 1.00, "edge_multiplier": 1.00, "signal_floor": 0.00, "front_ratio": 0.86},
    5200: {"size_multiplier": 1.00, "edge_multiplier": 1.00, "signal_floor": 0.00, "front_ratio": 0.86},
    5300: {"size_multiplier": 1.00, "edge_multiplier": 1.00, "signal_floor": 0.00, "front_ratio": 0.86},
    5400: {"size_multiplier": 1.00, "edge_multiplier": 1.00, "signal_floor": 0.00, "front_ratio": 0.86},
    5500: {"size_multiplier": 1.00, "edge_multiplier": 1.00, "signal_floor": 0.00, "front_ratio": 0.86},
    6000: {"size_multiplier": 1.00, "edge_multiplier": 1.00, "signal_floor": 0.00, "front_ratio": 0.86},
    6500: {"size_multiplier": 1.00, "edge_multiplier": 1.00, "signal_floor": 0.00, "front_ratio": 0.86},
}
ROUND_3_START_TTE_DAYS = 5.0
TRADING_DAY_TIMESTAMPS = 1_000_000


class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *args, **kwargs) -> None:
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


def black_scholes_call_delta(spot: float, strike: float, tte_days: float, volatility: float) -> float:
    if spot <= 0 or strike <= 0 or tte_days <= 0 or volatility <= 0:
        return 1.0 if spot > strike else 0.0

    tte_years = max(tte_days / 365.0, 1e-6)
    sigma_sqrt_t = volatility * sqrt(tte_years)
    if sigma_sqrt_t <= 1e-9:
        return 1.0 if spot > strike else 0.0

    d1 = (log(spot / strike) + 0.5 * volatility * volatility * tte_years) / sigma_sqrt_t
    return normal_cdf(d1)


def implied_call_volatility(spot: float, strike: float, tte_days: float, option_price: float) -> Optional[float]:
    intrinsic = max(0.0, spot - strike)
    if spot <= 0 or strike <= 0 or tte_days <= 0 or option_price <= intrinsic + 0.05 or option_price <= 0.05:
        return None

    lower = VOUCHER_VOL_FLOOR
    upper = VOUCHER_VOL_CAP
    for _ in range(45):
        midpoint = (lower + upper) / 2
        if black_scholes_call(spot, strike, tte_days, midpoint) < option_price:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2


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


class DeltaOneProduct(Product):
    def __init__(self, symbol: str, state: TradingState, trader_state: Dict[str, str]) -> None:
        self.config = DELTA_ONE_CONFIG[symbol]
        super().__init__(
            symbol,
            state,
            trader_state,
            POSITION_LIMITS[symbol],
            int(self.config["quote_size"]),
            int(self.config["signal_size_boost"]),
            int(self.config["second_level_offset"]),
        )

    def history_key(self) -> str:
        return f"hist:{self.symbol}"

    def parse_history(self) -> List[float]:
        raw_history = self.trader_state.get(self.history_key(), "")
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
        self.trader_state[self.history_key()] = ",".join(f"{price:.2f}" for price in history)

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
        history = history[-DELTA_ONE_HISTORY_LENGTH:]
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
        slow_ema = self.ema(history, DELTA_ONE_SLOW_ALPHA)
        fast_ema = self.ema(history, DELTA_ONE_FAST_ALPHA)
        fair_value = 0.35 * fast_ema + 0.65 * slow_ema

        if best_bid is not None and best_ask is not None and best_bid < best_ask:
            micro_price = (
                best_bid * (ask_volume or 0) + best_ask * (bid_volume or 0)
            ) / max(1, (bid_volume or 0) + (ask_volume or 0))
            micro_weight = float(self.config["micro_weight"])
            fair_value = (1 - micro_weight) * fair_value + micro_weight * micro_price

        anchor = sum(history[-12:]) / min(len(history), 12)
        reversion_shift = clamp(
            (anchor - fair_value) * float(self.config["mean_reversion_weight"]),
            -DELTA_ONE_MAX_REVERSION_SHIFT,
            DELTA_ONE_MAX_REVERSION_SHIFT,
        )

        signal_strength = abs(reversion_shift) / max(1e-9, DELTA_ONE_MAX_REVERSION_SHIFT)
        return fair_value + reversion_shift, clamp(signal_strength, 0.0, 1.0)

    def take_liquidity(self, fair_value: float) -> None:
        reservation_price = fair_value - self.pending_position * self.config["inventory_skew"]
        buy_threshold = reservation_price - self.config["take_edge"]
        sell_threshold = reservation_price + self.config["take_edge"]

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
        inventory_shift = self.pending_position * self.config["inventory_skew"]
        directional_push = signal_strength * 1.5
        reservation_price = fair_value - inventory_shift

        bid_quote = floor(reservation_price - self.config["base_half_spread"] + directional_push)
        ask_quote = ceil(reservation_price + self.config["base_half_spread"] - directional_push)

        if best_bid is not None:
            bid_quote = min(max(bid_quote, best_bid + 1), floor(reservation_price - int(self.config["min_edge"])))
        if best_ask is not None:
            ask_quote = max(min(ask_quote, best_ask - 1), ceil(reservation_price + int(self.config["min_edge"])))

        if best_ask is not None:
            bid_quote = min(bid_quote, best_ask - int(self.config["min_edge"]))
        if best_bid is not None:
            ask_quote = max(ask_quote, best_bid + int(self.config["min_edge"]))

        if bid_quote >= ask_quote:
            center = round(reservation_price)
            bid_quote = center - int(self.config["min_edge"])
            ask_quote = center + int(self.config["min_edge"])

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


class VoucherProduct(Product):
    def __init__(
        self,
        symbol: str,
        strike: int,
        state: TradingState,
        trader_state: Dict[str, str],
        underlying_fair: float,
        volatility: float,
        tte_days: float,
        underlying_signal: float,
    ) -> None:
        self.strike = strike
        self.voucher_config = VOUCHER_RISK_BY_STRIKE.get(strike, {})
        quote_size = max(1, round(VOUCHER_QUOTE_SIZE * self.voucher_config.get("size_multiplier", 1.0)))
        signal_size_boost = max(0, round(VOUCHER_SIGNAL_SIZE_BOOST * self.voucher_config.get("size_multiplier", 1.0)))
        super().__init__(
            symbol,
            state,
            trader_state,
            POSITION_LIMITS[symbol],
            quote_size,
            signal_size_boost,
            VOUCHER_SECOND_LEVEL_OFFSET,
        )
        self.underlying_fair = underlying_fair
        self.volatility = volatility
        self.tte_days = tte_days
        self.underlying_signal = underlying_signal
        self.front_ratio = self.voucher_config.get("front_ratio", 0.86)

    def theoretical_value(self) -> tuple[float, float]:
        if self.strike <= 4500:
            model_value = max(0.0, self.underlying_fair - self.strike)
        elif self.strike >= 6000:
            model_value = 0.0
        else:
            model_value = black_scholes_call(
                self.underlying_fair,
                float(self.strike),
                self.tte_days,
                self.volatility,
            )

        delta = black_scholes_call_delta(self.underlying_fair, float(self.strike), self.tte_days, self.volatility)
        effective_delta = clamp(delta, 0.25, 1.0)
        model_value = max(0.0, model_value - self.underlying_signal * effective_delta)

        option_mid = mid_price(self.order_depth)
        if option_mid is None:
            return model_value, 1.0

        blended_value = 0.95 * model_value + 0.05 * option_mid
        disagreement = abs(model_value - option_mid)
        edge_multiplier = self.voucher_config.get("edge_multiplier", 1.0)
        signal_strength = clamp(disagreement / max(4.0 * edge_multiplier, 0.025 * edge_multiplier * max(1.0, blended_value)), 0.0, 1.0)
        return blended_value, signal_strength

    def take_liquidity(self, fair_value: float) -> None:
        if self.signal_strength < self.voucher_config.get("signal_floor", 0.0):
            return

        inventory_shift = self.pending_position * VOUCHER_INVENTORY_SKEW_PER_UNIT
        reservation_price = fair_value - inventory_shift
        edge_multiplier = self.voucher_config.get("edge_multiplier", 1.0)
        edge = max(VOUCHER_TAKE_EDGE * edge_multiplier, 0.013 * edge_multiplier * max(1.0, fair_value))
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
        if self.signal_strength < self.voucher_config.get("signal_floor", 0.0):
            return None, None

        reservation_price = max(0.0, fair_value - self.pending_position * VOUCHER_INVENTORY_SKEW_PER_UNIT)
        edge_multiplier = self.voucher_config.get("edge_multiplier", 1.0)
        half_spread = max(
            VOUCHER_BASE_HALF_SPREAD * edge_multiplier,
            VOUCHER_MIN_THEO_SPREAD * edge_multiplier,
            fair_value * 0.013 * edge_multiplier,
        )
        bid_quote = max(0, floor(reservation_price - half_spread))
        ask_quote = max(1, ceil(reservation_price + half_spread))

        cheap_option = fair_value < 1.00
        if cheap_option:
            bid_quote = None
            if self.pending_position <= 0 and self.signal_strength < 0.95:
                ask_quote = None

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

        if bid_quote is not None and bid_quote >= ask_quote:
            center = max(0, round(reservation_price))
            bid_quote = max(0, center - VOUCHER_MIN_EDGE)
            ask_quote = max(bid_quote + VOUCHER_MIN_EDGE, center + VOUCHER_MIN_EDGE)

        return bid_quote, ask_quote

    def quote_basket_signal(self) -> None:
        signal_scale = min(1.0, abs(self.underlying_signal) / max(1e-9, VOUCHER_MAX_UNDERLYING_SIGNAL))
        if signal_scale <= 0:
            return

        best_bid, bid_volume, best_ask, ask_volume = self.best_bid_ask()
        size = max(1, round(self.quote_size * VOUCHER_BASKET_QUOTE_FRACTION * signal_scale))

        if self.underlying_signal < 0 and best_ask is not None:
            if best_bid is None:
                bid_quote = max(0, best_ask - VOUCHER_MIN_EDGE)
            elif best_ask - best_bid <= 1:
                bid_quote = best_bid
            else:
                bid_quote = min(best_bid + 1, best_ask - 1)
            self.add_buy(bid_quote, size)
        elif self.underlying_signal > 0 and best_bid is not None:
            if best_ask is None:
                ask_quote = best_bid + VOUCHER_MIN_EDGE
            elif best_ask - best_bid <= 1:
                ask_quote = best_ask
            else:
                ask_quote = max(best_ask - 1, best_bid + 1)
            self.add_sell(ask_quote, size)

    def build_orders(self) -> List[Order]:
        if self.underlying_fair <= 0 or self.tte_days <= 0:
            return self.orders

        fair_value, signal_strength = self.theoretical_value()
        self.signal_strength = signal_strength
        self.quote_basket_signal()
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
    def parse_history(trader_state: Dict[str, str], symbol: str) -> List[float]:
        raw_history = trader_state.get(f"hist:{symbol}", "")
        return Trader.parse_float_series(raw_history)

    @staticmethod
    def parse_float_series(raw_values: str) -> List[float]:
        history: List[float] = []
        for value in raw_values.split(","):
            if not value:
                continue
            try:
                history.append(float(value))
            except ValueError:
                continue
        return history

    @staticmethod
    def save_float_series(trader_state: Dict[str, str], key: str, values: List[float], precision: int = 4) -> None:
        trader_state[key] = ",".join(f"{value:.{precision}f}" for value in values)

    @staticmethod
    def implied_volatility_key(symbol: str) -> str:
        return f"iv:{symbol}"

    def rolling_option_volatility(self, trader_state: Dict[str, str], symbol: str) -> float:
        history = self.parse_float_series(trader_state.get(self.implied_volatility_key(symbol), ""))
        if not history:
            return VOUCHER_DEFAULT_VOL
        return clamp(sum(history[-VOUCHER_VOL_WINDOW:]) / len(history[-VOUCHER_VOL_WINDOW:]), VOUCHER_VOL_FLOOR, VOUCHER_VOL_CAP)

    def update_rolling_option_volatility(
        self,
        trader_state: Dict[str, str],
        symbol: str,
        strike: int,
        state: TradingState,
        underlying_fair: float,
        tte_days: float,
    ) -> None:
        option_mid = mid_price(state.order_depths.get(symbol))
        if option_mid is None:
            return

        implied_volatility = implied_call_volatility(underlying_fair, float(strike), tte_days, option_mid)
        if implied_volatility is None:
            return

        key = self.implied_volatility_key(symbol)
        history = self.parse_float_series(trader_state.get(key, ""))
        history.append(clamp(implied_volatility, VOUCHER_VOL_FLOOR, VOUCHER_VOL_CAP))
        history = history[-VOUCHER_VOL_WINDOW:]
        self.save_float_series(trader_state, key, history)

    @staticmethod
    def underlying_option_signal(history: List[float]) -> float:
        if len(history) < VOUCHER_SIGNAL_SLOW_WINDOW:
            return 0.0

        fast_average = sum(history[-VOUCHER_SIGNAL_FAST_WINDOW:]) / VOUCHER_SIGNAL_FAST_WINDOW
        slow_average = sum(history[-VOUCHER_SIGNAL_SLOW_WINDOW:]) / VOUCHER_SIGNAL_SLOW_WINDOW
        signal = fast_average - slow_average
        if abs(signal) < VOUCHER_UNDERLYING_SIGNAL_THRESHOLD:
            return 0.0
        return clamp(signal, -VOUCHER_MAX_UNDERLYING_SIGNAL, VOUCHER_MAX_UNDERLYING_SIGNAL)

    @staticmethod
    def round_3_tte_days(state: TradingState) -> float:
        timestamp = getattr(state, "timestamp", 0) or 0
        intraday_progress = clamp(timestamp / TRADING_DAY_TIMESTAMPS, 0.0, 1.0)
        return max(0.05, ROUND_3_START_TTE_DAYS - intraday_progress)

    def run(self, state: TradingState):
        trader_state = self.decode_trader_data(getattr(state, "traderData", ""))
        orders: Dict[str, List[Order]] = {}

        for symbol in (HYDROGEL_PACK, VELVETFRUIT_EXTRACT):
            if symbol in state.order_depths:
                orders[symbol] = DeltaOneProduct(symbol, state, trader_state).build_orders()

        velvet_history = self.parse_history(trader_state, VELVETFRUIT_EXTRACT)
        underlying_mid = mid_price(state.order_depths.get(VELVETFRUIT_EXTRACT))
        if underlying_mid is not None:
            underlying_fair = underlying_mid
        elif velvet_history:
            underlying_fair = velvet_history[-1]
        else:
            underlying_fair = 0.0

        underlying_signal = self.underlying_option_signal(velvet_history)
        tte_days = self.round_3_tte_days(state)
        for symbol, strike in VOUCHER_STRIKES.items():
            if symbol in state.order_depths and underlying_fair > 0:
                rolling_volatility = self.rolling_option_volatility(trader_state, symbol)
                orders[symbol] = VoucherProduct(
                    symbol,
                    strike,
                    state,
                    trader_state,
                    underlying_fair,
                    rolling_volatility,
                    tte_days,
                    underlying_signal,
                ).build_orders()
                self.update_rolling_option_volatility(trader_state, symbol, strike, state, underlying_fair, tte_days)

        trader_data = self.encode_trader_data(trader_state)
        conversions = 0
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data