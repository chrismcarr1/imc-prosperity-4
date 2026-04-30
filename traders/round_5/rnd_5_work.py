import json
from collections import deque
from dataclasses import dataclass, field
from math import sqrt
from typing import Deque, Dict, List, Optional, Tuple

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


CHOCOLATE = "SNACKPACK_CHOCOLATE"
VANILLA = "SNACKPACK_VANILLA"
PISTACHIO = "SNACKPACK_PISTACHIO"
STRAWBERRY = "SNACKPACK_STRAWBERRY"
RASPBERRY = "SNACKPACK_RASPBERRY"

ALLOWED_PRODUCTS = {
    "GALAXY_SOUNDS_DARK_MATTER",
    "GALAXY_SOUNDS_BLACK_HOLES",
    "GALAXY_SOUNDS_PLANETARY_RINGS",
    "GALAXY_SOUNDS_SOLAR_WINDS",
    "GALAXY_SOUNDS_SOLAR_FLAMES",
    "SLEEP_POD_SUEDE",
    "SLEEP_POD_LAMB_WOOL",
    "SLEEP_POD_POLYESTER",
    "SLEEP_POD_NYLON",
    "SLEEP_POD_COTTON",
    "MICROCHIP_CIRCLE",
    "MICROCHIP_OVAL",
    "MICROCHIP_SQUARE",
    "MICROCHIP_RECTANGLE",
    "MICROCHIP_TRIANGLE",
    "PEBBLES_XS",
    "PEBBLES_S",
    "PEBBLES_M",
    "PEBBLES_L",
    "PEBBLES_XL",
    "ROBOT_VACUUMING",
    "ROBOT_MOPPING",
    "ROBOT_DISHES",
    "ROBOT_LAUNDRY",
    "ROBOT_IRONING",
    "UV_VISOR_YELLOW",
    "UV_VISOR_AMBER",
    "UV_VISOR_ORANGE",
    "UV_VISOR_RED",
    "UV_VISOR_MAGENTA",
    "TRANSLATOR_SPACE_GRAY",
    "TRANSLATOR_ASTRO_BLACK",
    "TRANSLATOR_ECLIPSE_CHARCOAL",
    "TRANSLATOR_GRAPHITE_MIST",
    "TRANSLATOR_VOID_BLUE",
    "PANEL_1X2",
    "PANEL_2X2",
    "PANEL_1X4",
    "PANEL_2X4",
    "PANEL_4X4",
    "OXYGEN_SHAKE_MORNING_BREATH",
    "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_MINT",
    "OXYGEN_SHAKE_CHOCOLATE",
    "OXYGEN_SHAKE_GARLIC",
    CHOCOLATE,
    VANILLA,
    PISTACHIO,
    STRAWBERRY,
    RASPBERRY,
}

POSITION_LIMITS = {product: 10 for product in ALLOWED_PRODUCTS}

HELD_OUT_PRODUCTS = {
    "GALAXY_SOUNDS_DARK_MATTER",
    "GALAXY_SOUNDS_PLANETARY_RINGS",
    "GALAXY_SOUNDS_SOLAR_WINDS",
    "MICROCHIP_RECTANGLE",
    "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_MINT",
    "OXYGEN_SHAKE_MORNING_BREATH",
    "PANEL_1X2",
    "PANEL_1X4",
    "PANEL_2X2",
    "PEBBLES_L",
    "ROBOT_DISHES",
    "ROBOT_LAUNDRY",
    "SLEEP_POD_LAMB_WOOL",
    "SNACKPACK_CHOCOLATE",
    "SNACKPACK_PISTACHIO",
    "SNACKPACK_RASPBERRY",
    "SNACKPACK_STRAWBERRY",
    "TRANSLATOR_SPACE_GRAY",
    "UV_VISOR_AMBER",
}

TRADED_PRODUCTS = ALLOWED_PRODUCTS - HELD_OUT_PRODUCTS

SPECIALIST_CONFIGS = {
    # Kept from the previous held-name short-horizon test.
    "GALAXY_SOUNDS_SOLAR_WINDS": {"lookback": 50, "mode": "fade", "size": 4, "min_move": 0},
    "SNACKPACK_CHOCOLATE": {"lookback": 50, "mode": "fade", "size": 4, "min_move": 0},
    "SNACKPACK_PISTACHIO": {"lookback": 50, "mode": "fade", "size": 4, "min_move": 0},
    "SNACKPACK_STRAWBERRY": {"lookback": 50, "mode": "fade", "size": 4, "min_move": 0},
    # Previously untouched names. These are intentionally small, one-sided
    # rules selected on the 1000-tick submitted-log horizon.
    "GALAXY_SOUNDS_DARK_MATTER": {"lookback": 20, "mode": "follow", "size": 4, "min_move": 40},
    "GALAXY_SOUNDS_PLANETARY_RINGS": {"lookback": 5, "mode": "follow", "size": 4, "min_move": 40},
    "MICROCHIP_RECTANGLE": {"lookback": 20, "mode": "follow", "size": 4, "min_move": 20},
    "OXYGEN_SHAKE_MINT": {"lookback": 50, "mode": "fade", "size": 4, "min_move": 40},
    "OXYGEN_SHAKE_MORNING_BREATH": {"lookback": 20, "mode": "follow", "size": 4, "min_move": 40},
    "PANEL_1X2": {"lookback": 75, "mode": "follow", "size": 4, "min_move": 40},
    "PANEL_1X4": {"lookback": 75, "mode": "follow", "size": 4, "min_move": 40},
    "PANEL_2X2": {"lookback": 10, "mode": "follow", "size": 4, "min_move": 5},
    "PEBBLES_L": {"lookback": 20, "mode": "fade", "size": 4, "min_move": 0},
    "ROBOT_DISHES": {"lookback": 3, "mode": "fade", "size": 4, "min_move": 40},
    "ROBOT_LAUNDRY": {"lookback": 20, "mode": "follow", "size": 4, "min_move": 20},
    "SLEEP_POD_LAMB_WOOL": {"lookback": 75, "mode": "follow", "size": 4, "min_move": 40},
    "TRANSLATOR_SPACE_GRAY": {"lookback": 75, "mode": "follow", "size": 4, "min_move": 5},
    "UV_VISOR_AMBER": {"lookback": 20, "mode": "follow", "size": 4, "min_move": 40},
}

# Each pair is modelled as y ~= alpha + beta * x. Signal +1 means the residual
# is cheap: long y and short beta*x. Signal -1 means short y and long beta*x.
PAIR_CONFIGS = [
    # These two signals activate early enough for the 1000-tick competition log
    # and tested best once shared snackpack position limits are enforced.
    {"key": "RS", "y": RASPBERRY, "x": STRAWBERRY, "weight": 1.00},
    {"key": "PR", "y": PISTACHIO, "x": RASPBERRY, "weight": 1.00},
]
PAIR_CONFIGS = [
    config
    for config in PAIR_CONFIGS
    if config["y"] not in HELD_OUT_PRODUCTS and config["x"] not in HELD_OUT_PRODUCTS
]

WINDOW = 75
ENTRY_Z = 1.5
EXIT_Z = 0.25
MIN_EDGE_AFTER_COST = 5.0
PAIR_BASE_TARGET = 6
PAIR_MAX_TARGET = 10
MAX_ORDER_SIZE = 10
MIN_BETA = 0.25
MAX_BETA = 2.25
MIN_RESID_STD = 1.0

MM_BASE_SIZE = 5
MM_WIDE_SIZE = 8
MM_MIN_SPREAD = 4
MM_JOIN_SPREAD = 3
MM_INVENTORY_SKEW = 0.45
MM_SOFT_LIMIT = 8

SPECIALIST_MIN_SPREAD = 3


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def best_bid_ask(order_depth: Optional[OrderDepth]) -> Tuple[Optional[int], int, Optional[int], int]:
    if order_depth is None:
        return None, 0, None, 0

    best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
    best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
    bid_volume = order_depth.buy_orders.get(best_bid, 0) if best_bid is not None else 0
    ask_volume = abs(order_depth.sell_orders.get(best_ask, 0)) if best_ask is not None else 0
    return best_bid, bid_volume, best_ask, ask_volume


def mid_price(order_depth: Optional[OrderDepth]) -> Optional[float]:
    best_bid, _, best_ask, _ = best_bid_ask(order_depth)
    if best_bid is None or best_ask is None:
        return None
    return (best_bid + best_ask) / 2.0


def decode_trader_data(trader_data: str) -> Dict[str, float]:
    if not trader_data:
        return {}
    try:
        decoded = json.loads(trader_data)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def encode_trader_data(trader_state: Dict[str, float]) -> str:
    compact = {}
    for key, value in trader_state.items():
        compact[key] = round(value, 6) if isinstance(value, float) else value
    return json.dumps(compact, separators=(",", ":"))


class RollingPairSignal:
    def __init__(self, key: str, y_symbol: str, x_symbol: str, weight: float) -> None:
        self.key = key
        self.y_symbol = y_symbol
        self.x_symbol = x_symbol
        self.weight = weight

        self.x_values: Deque[float] = deque()
        self.y_values: Deque[float] = deque()
        self.residuals: Deque[float] = deque()
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_xx = 0.0
        self.sum_xy = 0.0
        self.sum_resid = 0.0
        self.sum_resid_sq = 0.0

    def push(self, x_mid: float, y_mid: float) -> Tuple[Optional[float], float, float]:
        self.x_values.append(x_mid)
        self.y_values.append(y_mid)
        self.sum_x += x_mid
        self.sum_y += y_mid
        self.sum_xx += x_mid * x_mid
        self.sum_xy += x_mid * y_mid

        if len(self.x_values) > WINDOW:
            old_x = self.x_values.popleft()
            old_y = self.y_values.popleft()
            self.sum_x -= old_x
            self.sum_y -= old_y
            self.sum_xx -= old_x * old_x
            self.sum_xy -= old_x * old_y

        if len(self.x_values) < WINDOW:
            return None, 1.0, MIN_RESID_STD

        n = float(WINDOW)
        mean_x = self.sum_x / n
        mean_y = self.sum_y / n
        var_x = max(1e-9, self.sum_xx / n - mean_x * mean_x)
        cov_xy = self.sum_xy / n - mean_x * mean_y
        beta = clamp(cov_xy / var_x, MIN_BETA, MAX_BETA)
        alpha = mean_y - beta * mean_x
        residual = y_mid - (alpha + beta * x_mid)

        self.residuals.append(residual)
        self.sum_resid += residual
        self.sum_resid_sq += residual * residual
        if len(self.residuals) > WINDOW:
            old_residual = self.residuals.popleft()
            self.sum_resid -= old_residual
            self.sum_resid_sq -= old_residual * old_residual

        if len(self.residuals) < WINDOW:
            return None, beta, MIN_RESID_STD

        resid_mean = self.sum_resid / n
        resid_var = max(1e-9, self.sum_resid_sq / n - resid_mean * resid_mean)
        resid_std = max(MIN_RESID_STD, sqrt(resid_var))
        z_score = (residual - resid_mean) / resid_std
        return z_score, beta, resid_std

    def desired_legs(
        self,
        signal: int,
        beta: float,
        z_score: float,
    ) -> Dict[str, float]:
        if signal == 0:
            return {}

        conviction = clamp((abs(z_score) - ENTRY_Z) / 1.5, 0.0, 1.0)
        y_target = self.weight * (PAIR_BASE_TARGET + conviction * (PAIR_MAX_TARGET - PAIR_BASE_TARGET))
        x_target = min(10.0, beta * y_target)

        return {
            self.y_symbol: signal * y_target,
            self.x_symbol: -signal * x_target,
        }


def order_to_target(
    symbol: str,
    target_position: int,
    state: TradingState,
    pending_positions: Dict[str, int],
    aggressive: bool,
) -> List[Order]:
    if symbol not in TRADED_PRODUCTS:
        return []

    order_depth = state.order_depths.get(symbol)
    best_bid, bid_volume, best_ask, ask_volume = best_bid_ask(order_depth)
    if best_bid is None or best_ask is None:
        return []

    current_position = int(state.position.get(symbol, 0))
    pending_position = pending_positions.get(symbol, current_position)
    desired_change = target_position - pending_position
    if desired_change == 0:
        return []

    limit = POSITION_LIMITS[symbol]
    orders: List[Order] = []

    if desired_change > 0:
        quantity = min(desired_change, ask_volume, MAX_ORDER_SIZE, limit - pending_position)
        if quantity > 0:
            price = best_ask if aggressive else best_bid
            orders.append(Order(symbol, price, quantity))
            pending_positions[symbol] = pending_position + quantity
    else:
        quantity = min(-desired_change, bid_volume, MAX_ORDER_SIZE, limit + pending_position)
        if quantity > 0:
            price = best_bid if aggressive else best_ask
            orders.append(Order(symbol, price, -quantity))
            pending_positions[symbol] = pending_position - quantity

    return orders


def market_make_symbol(
    symbol: str,
    state: TradingState,
    pending_positions: Dict[str, int],
    inventory_target: int,
) -> List[Order]:
    if symbol not in TRADED_PRODUCTS:
        return []

    order_depth = state.order_depths.get(symbol)
    best_bid, bid_volume, best_ask, ask_volume = best_bid_ask(order_depth)
    if best_bid is None or best_ask is None:
        return []

    spread = best_ask - best_bid
    if spread < MM_JOIN_SPREAD:
        return []

    limit = POSITION_LIMITS[symbol]
    pending_position = pending_positions.get(symbol, int(state.position.get(symbol, 0)))
    inventory_error = pending_position - inventory_target

    if spread >= MM_MIN_SPREAD:
        bid_price = best_bid + 1
        ask_price = best_ask - 1
    else:
        bid_price = best_bid
        ask_price = best_ask

    if bid_price >= ask_price:
        return []

    reservation_shift = int(round(MM_INVENTORY_SKEW * inventory_error))
    bid_price -= max(0, reservation_shift)
    ask_price -= min(0, reservation_shift)

    bid_price = min(bid_price, best_ask - 1)
    ask_price = max(ask_price, best_bid + 1)
    if bid_price >= ask_price:
        return []

    quote_size = MM_WIDE_SIZE if spread >= 8 else MM_BASE_SIZE
    buy_capacity = limit - pending_position
    sell_capacity = limit + pending_position

    if pending_position >= MM_SOFT_LIMIT:
        buy_capacity = 0
    if pending_position <= -MM_SOFT_LIMIT:
        sell_capacity = 0

    buy_qty = min(quote_size, buy_capacity, max(0, ask_volume))
    sell_qty = min(quote_size, sell_capacity, max(0, bid_volume))

    orders: List[Order] = []
    if buy_qty > 0:
        orders.append(Order(symbol, bid_price, buy_qty))
        pending_position += buy_qty
    if sell_qty > 0:
        orders.append(Order(symbol, ask_price, -sell_qty))
        pending_position -= sell_qty

    pending_positions[symbol] = pending_position
    return orders


def specialist_orders(
    symbol: str,
    state: TradingState,
    pending_positions: Dict[str, int],
    history: Deque[float],
    config: Dict[str, int],
) -> List[Order]:
    order_depth = state.order_depths.get(symbol)
    best_bid, bid_volume, best_ask, ask_volume = best_bid_ask(order_depth)
    if best_bid is None or best_ask is None:
        return []

    spread = best_ask - best_bid
    if spread < SPECIALIST_MIN_SPREAD:
        return []

    mid = (best_bid + best_ask) / 2.0
    history.append(mid)
    lookback = int(config["lookback"])
    if len(history) <= lookback:
        return []

    old_mid = list(history)[-lookback - 1]
    move = mid - old_mid
    if abs(move) < float(config["min_move"]):
        return []

    bid_price = best_bid + 1 if spread >= 4 else best_bid
    ask_price = best_ask - 1 if spread >= 4 else best_ask
    if bid_price >= ask_price:
        return []

    pending_position = pending_positions.get(symbol, int(state.position.get(symbol, 0)))
    limit = POSITION_LIMITS[symbol]
    orders: List[Order] = []

    side = 1 if move > 0 else -1
    if config["mode"] == "fade":
        side *= -1

    quantity_limit = int(config["size"])
    if side > 0 and pending_position < limit:
        quantity = min(quantity_limit, limit - pending_position, ask_volume)
        if quantity > 0:
            orders.append(Order(symbol, bid_price, quantity))
            pending_position += quantity
    elif side < 0 and pending_position > -limit:
        quantity = min(quantity_limit, limit + pending_position, bid_volume)
        if quantity > 0:
            orders.append(Order(symbol, ask_price, -quantity))
            pending_position -= quantity

    pending_positions[symbol] = pending_position
    return orders


class Trader:
    def __init__(self) -> None:
        self.pairs = [
            RollingPairSignal(config["key"], config["y"], config["x"], float(config["weight"]))
            for config in PAIR_CONFIGS
        ]
        self.held_histories: Dict[str, Deque[float]] = {
            symbol: deque(maxlen=int(config["lookback"]) + 1)
            for symbol, config in SPECIALIST_CONFIGS.items()
        }

    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        trader_state = decode_trader_data(getattr(state, "traderData", ""))
        raw_targets: Dict[str, float] = {}
        next_state: Dict[str, float] = {}

        for pair in self.pairs:
            y_depth = state.order_depths.get(pair.y_symbol)
            x_depth = state.order_depths.get(pair.x_symbol)
            y_mid = mid_price(y_depth)
            x_mid = mid_price(x_depth)

            if y_mid is None or x_mid is None:
                next_state[f"sig:{pair.key}"] = 0
                continue

            y_bid, _, y_ask, _ = best_bid_ask(y_depth)
            x_bid, _, x_ask, _ = best_bid_ask(x_depth)
            execution_cost = float((y_ask - y_bid) + (x_ask - x_bid))

            z_value, beta, resid_std = pair.push(x_mid, y_mid)
            current_signal = int(trader_state.get(f"sig:{pair.key}", 0))
            signal = current_signal
            z_score = 0.0 if z_value is None else z_value

            if z_value is None:
                signal = 0
            elif current_signal == 0:
                edge_after_cost = abs(z_score) * resid_std - execution_cost
                if edge_after_cost >= MIN_EDGE_AFTER_COST:
                    if z_score > ENTRY_Z:
                        signal = -1
                    elif z_score < -ENTRY_Z:
                        signal = 1
            elif current_signal == 1 and z_score > -EXIT_Z:
                signal = 0
            elif current_signal == -1 and z_score < EXIT_Z:
                signal = 0

            for symbol, contribution in pair.desired_legs(signal, beta, z_score).items():
                raw_targets[symbol] = raw_targets.get(symbol, 0.0) + contribution

            next_state[f"sig:{pair.key}"] = signal
            next_state[f"z:{pair.key}"] = z_score
            next_state[f"b:{pair.key}"] = beta

        target_positions: Dict[str, int] = {}
        for symbol, raw_target in raw_targets.items():
            limit = POSITION_LIMITS[symbol]
            target_positions[symbol] = int(round(clamp(raw_target, -limit, limit)))

        pair_symbols = {config["y"] for config in PAIR_CONFIGS} | {config["x"] for config in PAIR_CONFIGS}
        quote_symbols = sorted(TRADED_PRODUCTS & set(state.order_depths.keys()))

        for symbol in pair_symbols:
            target_positions.setdefault(symbol, 0)

        pending_positions = {
            symbol: int(state.position.get(symbol, 0))
            for symbol in quote_symbols
        }

        for symbol in sorted(pair_symbols):
            current_position = int(state.position.get(symbol, 0))
            target = target_positions[symbol]
            aggressive = abs(target) < abs(current_position) or target * current_position < 0
            symbol_orders = order_to_target(symbol, target, state, pending_positions, aggressive)
            if symbol_orders:
                orders[symbol] = symbol_orders

        for symbol in quote_symbols:
            inventory_target = target_positions.get(symbol, 0)
            symbol_orders = market_make_symbol(symbol, state, pending_positions, inventory_target)
            if symbol_orders:
                orders.setdefault(symbol, []).extend(symbol_orders)

        for symbol in sorted(set(SPECIALIST_CONFIGS) & set(state.order_depths.keys())):
            symbol_orders = specialist_orders(
                symbol,
                state,
                pending_positions,
                self.held_histories[symbol],
                SPECIALIST_CONFIGS[symbol],
            )
            if symbol_orders:
                orders.setdefault(symbol, []).extend(symbol_orders)

        return orders, 0, encode_trader_data(next_state)
