import csv
import base64
import html
import io
import json
import math
import re
import statistics
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    matplotlib = None
    plt = None
    sns = None


ROOT = Path(__file__).resolve().parent
MARKOUT_HORIZONS = (1, 5, 10, 25, 50)
ROLLING_WINDOW = 25


@dataclass
class TradeRecord:
    timestamp: int
    symbol: str
    price: float
    quantity: int
    buyer: str
    seller: str

    @property
    def submission_side(self) -> int:
        if self.buyer == "SUBMISSION" and self.seller != "SUBMISSION":
            return 1
        if self.seller == "SUBMISSION" and self.buyer != "SUBMISSION":
            return -1
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "price": self.price,
            "quantity": self.quantity,
            "buyer": self.buyer,
            "seller": self.seller,
            "submission_side": self.submission_side,
        }


def latest_log_path(directory: Path) -> Path:
    logs = sorted(directory.glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not logs:
        raise FileNotFoundError(f"No .log files found in {directory}")
    return logs[0]


def load_sections(log_path: Path) -> tuple[str, str, str]:
    text = log_path.read_text(encoding="utf-8")
    activities_marker = "\nActivities log:\n"
    trades_marker = "\nTrade History:\n"

    activities_start = text.index(activities_marker)
    trades_start = text.index(trades_marker)

    sandbox_text = text[len("Sandbox logs:\n"):activities_start].strip()
    activities_text = text[activities_start + len(activities_marker):trades_start].strip()
    trade_text = text[trades_start + len(trades_marker):].strip()
    return sandbox_text, activities_text, trade_text


def parse_sandbox_logs(sandbox_text: str) -> list[dict[str, Any]]:
    if not sandbox_text:
        return []

    blocks = re.findall(r"\{[\s\S]*?\}", sandbox_text)
    parsed: list[dict[str, Any]] = []
    for block in blocks:
        try:
            parsed.append(json.loads(block))
        except json.JSONDecodeError:
            continue
    return parsed


def parse_activities(activities_text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(activities_text), delimiter=";")
    rows: list[dict[str, Any]] = []
    for raw in reader:
        row: dict[str, Any] = {}
        for key, value in raw.items():
            value = value.strip()
            if value == "":
                row[key] = None
            elif key in {"day", "timestamp"} or key.startswith("bid_price") or key.startswith("bid_volume") or key.startswith("ask_price") or key.startswith("ask_volume"):
                row[key] = int(float(value))
            elif key in {"mid_price", "profit_and_loss"}:
                row[key] = float(value)
            else:
                row[key] = value

        bid = row.get("bid_price_1")
        ask = row.get("ask_price_1")
        bid_v = abs(row.get("bid_volume_1") or 0)
        ask_v = abs(row.get("ask_volume_1") or 0)
        row["spread"] = float(ask - bid) if bid is not None and ask is not None else None
        denom = bid_v + ask_v
        row["imbalance"] = ((bid_v - ask_v) / denom) if denom else 0.0
        row["top_bid_depth"] = bid_v
        row["top_ask_depth"] = ask_v
        row["top3_bid_depth"] = sum(abs(row.get(f"bid_volume_{i}") or 0) for i in range(1, 4))
        row["top3_ask_depth"] = sum(abs(row.get(f"ask_volume_{i}") or 0) for i in range(1, 4))
        rows.append(row)

    return rows


def parse_trades(trade_text: str) -> list[TradeRecord]:
    cleaned = re.sub(r",(\s*[}\]])", r"\1", trade_text)
    payload = json.loads(cleaned)
    trades = [
        TradeRecord(
            timestamp=int(item["timestamp"]),
            symbol=item["symbol"],
            price=float(item["price"]),
            quantity=int(item["quantity"]),
            buyer=item.get("buyer", ""),
            seller=item.get("seller", ""),
        )
        for item in payload
    ]
    trades.sort(key=lambda trade: (trade.timestamp, trade.symbol, trade.price, trade.quantity))
    return trades


def rolling_std(values: list[float], window: int) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    for idx in range(len(values)):
        if idx + 1 < window:
            out.append(None)
            continue
        segment = values[idx + 1 - window: idx + 1]
        out.append(statistics.pstdev(segment) if len(segment) > 1 else 0.0)
    return out


def rolling_mean(values: list[float], window: int) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    running = 0.0
    for idx, value in enumerate(values):
        running += value
        if idx >= window:
            running -= values[idx - window]
        if idx + 1 < window:
            out.append(None)
        else:
            out.append(running / window)
    return out


def max_drawdown(series: list[float]) -> float:
    peak = float("-inf")
    worst = 0.0
    for value in series:
        peak = max(peak, value)
        worst = min(worst, value - peak)
    return worst


def mean(values: list[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def fmt_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "NA"
        return f"{value:,.{digits}f}"
    return f"{value:,}"


def series_lookup(rows: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
    timestamps = [int(row["timestamp"]) for row in rows]
    mids = [float(row["mid_price"]) for row in rows]
    return timestamps, mids


def value_at_or_before(times: list[int], values: list[float], timestamp: int) -> Optional[float]:
    idx = bisect_right(times, timestamp) - 1
    if idx < 0:
        return None
    return values[idx]


def future_value(times: list[int], values: list[float], timestamp: int, horizon_steps: int) -> Optional[float]:
    idx = bisect_right(times, timestamp) - 1
    target = idx + horizon_steps
    if idx < 0 or target >= len(values):
        return None
    return values[target]


def compute_trade_analytics(
    product: str,
    rows: list[dict[str, Any]],
    trades: list[TradeRecord],
) -> dict[str, Any]:
    product_trades = [trade for trade in trades if trade.symbol == product]
    submission_trades = [trade for trade in product_trades if trade.submission_side != 0]
    times, mids = series_lookup(rows)

    buy_timestamps: list[int] = []
    buy_prices: list[float] = []
    buy_sizes: list[int] = []
    sell_timestamps: list[int] = []
    sell_prices: list[float] = []
    sell_sizes: list[int] = []
    execution_edges: list[float] = []
    realized_spreads: list[float] = []
    markouts: dict[int, list[float]] = {horizon: [] for horizon in MARKOUT_HORIZONS}
    trade_notional = 0.0
    signed_volume = 0
    buy_volume = 0
    sell_volume = 0
    market_volume = sum(trade.quantity for trade in product_trades)
    submission_volume = 0
    trade_volume_by_timestamp: dict[int, int] = defaultdict(int)
    trade_count_by_timestamp: dict[int, int] = defaultdict(int)

    inventory_points: list[dict[str, Any]] = []
    inventory = 0

    for trade in submission_trades:
        side = trade.submission_side
        submission_volume += trade.quantity
        signed_qty = side * trade.quantity
        inventory += signed_qty
        inventory_points.append({"timestamp": trade.timestamp, "inventory": inventory, "signed_qty": signed_qty})
        trade_notional += trade.price * trade.quantity
        signed_volume += signed_qty
        trade_count_by_timestamp[trade.timestamp] += 1
        trade_volume_by_timestamp[trade.timestamp] += trade.quantity

        current_mid = value_at_or_before(times, mids, trade.timestamp)
        if current_mid is not None:
            edge = (current_mid - trade.price) if side > 0 else (trade.price - current_mid)
            execution_edges.append(edge)
            realized_spreads.append(edge * 2.0)

        for horizon in MARKOUT_HORIZONS:
            next_mid = future_value(times, mids, trade.timestamp, horizon)
            if next_mid is None:
                continue
            markout = (next_mid - trade.price) if side > 0 else (trade.price - next_mid)
            markouts[horizon].append(markout)

        if side > 0:
            buy_timestamps.append(trade.timestamp)
            buy_prices.append(trade.price)
            buy_sizes.append(trade.quantity)
            buy_volume += trade.quantity
        else:
            sell_timestamps.append(trade.timestamp)
            sell_prices.append(trade.price)
            sell_sizes.append(trade.quantity)
            sell_volume += trade.quantity

    row_timestamps = [int(row["timestamp"]) for row in rows]
    inventory_series: list[int] = []
    inventory_trade_idx = 0
    running_inventory = 0
    for timestamp in row_timestamps:
        while inventory_trade_idx < len(inventory_points) and inventory_points[inventory_trade_idx]["timestamp"] <= timestamp:
            running_inventory = inventory_points[inventory_trade_idx]["inventory"]
            inventory_trade_idx += 1
        inventory_series.append(running_inventory)

    return {
        "all_trades": [trade.to_dict() for trade in product_trades],
        "submission_trades": [trade.to_dict() for trade in submission_trades],
        "buy_markers": {"x": buy_timestamps, "y": buy_prices, "size": buy_sizes},
        "sell_markers": {"x": sell_timestamps, "y": sell_prices, "size": sell_sizes},
        "inventory_series": inventory_series,
        "inventory_points": inventory_points,
        "trade_volume_by_timestamp": dict(sorted(trade_volume_by_timestamp.items())),
        "trade_count_by_timestamp": dict(sorted(trade_count_by_timestamp.items())),
        "execution_edges": execution_edges,
        "realized_spreads": realized_spreads,
        "markouts": {str(h): values for h, values in markouts.items()},
        "summary": {
            "submission_trade_count": len(submission_trades),
            "market_trade_count": len(product_trades),
            "submission_volume": submission_volume,
            "market_volume": market_volume,
            "participation_rate": safe_div(submission_volume, market_volume),
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "net_volume": signed_volume,
            "avg_fill_price": safe_div(trade_notional, submission_volume),
            "avg_execution_edge": mean(execution_edges),
            "avg_realized_spread": mean(realized_spreads),
            "avg_markout": {str(h): mean(values) for h, values in markouts.items()},
            "inventory_final": inventory_series[-1] if inventory_series else 0,
            "inventory_max": max(inventory_series) if inventory_series else 0,
            "inventory_min": min(inventory_series) if inventory_series else 0,
            "avg_abs_inventory": mean([abs(x) for x in inventory_series]),
            "turnover": safe_div(submission_volume, max(1, max(abs(x) for x in inventory_series) if inventory_series else 1)),
        },
    }


def compute_product_report(product: str, rows: list[dict[str, Any]], trades: list[TradeRecord]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: row["timestamp"])
    timestamps = [int(row["timestamp"]) for row in rows]
    mids = [float(row["mid_price"]) for row in rows]
    pnls = [float(row["profit_and_loss"]) for row in rows]
    spreads = [float(row["spread"]) for row in rows if row["spread"] is not None]
    imbalances = [float(row["imbalance"]) for row in rows]
    top_bid_depth = [int(row["top_bid_depth"]) for row in rows]
    top_ask_depth = [int(row["top_ask_depth"]) for row in rows]
    top3_bid_depth = [int(row["top3_bid_depth"]) for row in rows]
    top3_ask_depth = [int(row["top3_ask_depth"]) for row in rows]
    returns = [0.0] + [mids[idx] - mids[idx - 1] for idx in range(1, len(mids))]
    abs_returns = [abs(value) for value in returns]
    rolling_vol = rolling_std(returns, ROLLING_WINDOW)
    rolling_abs_return = rolling_mean(abs_returns, ROLLING_WINDOW)
    trade_data = compute_trade_analytics(product, rows, trades)
    inventory_series = trade_data["inventory_series"]

    summary = {
        "rows": len(rows),
        "start_timestamp": timestamps[0] if timestamps else None,
        "end_timestamp": timestamps[-1] if timestamps else None,
        "final_pnl": pnls[-1] if pnls else None,
        "peak_pnl": max(pnls) if pnls else None,
        "trough_pnl": min(pnls) if pnls else None,
        "max_drawdown": max_drawdown(pnls) if pnls else None,
        "mid_min": min(mids) if mids else None,
        "mid_max": max(mids) if mids else None,
        "mid_mean": mean(mids),
        "spread_mean": mean(spreads),
        "spread_median": statistics.median(spreads) if spreads else None,
        "imbalance_mean": mean(imbalances),
        "realized_vol": statistics.pstdev(returns) if len(returns) > 1 else 0.0,
        "avg_abs_return": mean(abs_returns),
        "mid_return_correlation_proxy": safe_div(
            mean([returns[idx] * returns[idx - 1] for idx in range(1, len(returns))]) or 0.0,
            (statistics.pvariance(returns) if len(returns) > 1 else 0.0),
        ),
    }
    summary.update(trade_data["summary"])

    return {
        "summary": summary,
        "series": {
            "timestamps": timestamps,
            "mid": mids,
            "pnl": pnls,
            "spread": [row["spread"] for row in rows],
            "imbalance": imbalances,
            "top_bid_depth": top_bid_depth,
            "top_ask_depth": top_ask_depth,
            "top3_bid_depth": top3_bid_depth,
            "top3_ask_depth": top3_ask_depth,
            "returns": returns,
            "rolling_vol": rolling_vol,
            "rolling_abs_return": rolling_abs_return,
            "inventory": inventory_series,
        },
        "trades": trade_data,
    }


def compute_overall_report(products: dict[str, dict[str, Any]]) -> dict[str, Any]:
    all_timestamps = sorted({timestamp for product in products.values() for timestamp in product["series"]["timestamps"]})
    pnl_by_product: dict[str, dict[int, float]] = {}
    for name, report in products.items():
        pnl_by_product[name] = dict(zip(report["series"]["timestamps"], report["series"]["pnl"]))

    total_pnl = []
    for timestamp in all_timestamps:
        total_pnl.append(sum(series.get(timestamp, 0.0) for series in pnl_by_product.values()))

    final_pnls = {name: report["summary"]["final_pnl"] for name, report in products.items()}
    trade_counts = {name: report["summary"]["submission_trade_count"] for name, report in products.items()}
    submission_volume = {name: report["summary"]["submission_volume"] for name, report in products.items()}

    correlation = None
    names = list(products)
    if len(names) >= 2:
        left = products[names[0]]["series"]["returns"]
        right = products[names[1]]["series"]["returns"]
        pair_count = min(len(left), len(right))
        if pair_count > 2:
            left_slice = left[:pair_count]
            right_slice = right[:pair_count]
            left_mean = statistics.fmean(left_slice)
            right_mean = statistics.fmean(right_slice)
            numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_slice, right_slice))
            denominator = math.sqrt(
                sum((a - left_mean) ** 2 for a in left_slice) * sum((b - right_mean) ** 2 for b in right_slice)
            )
            correlation = numerator / denominator if denominator else None

    return {
        "timestamps": all_timestamps,
        "total_pnl": total_pnl,
        "final_pnls": final_pnls,
        "trade_counts": trade_counts,
        "submission_volume": submission_volume,
        "return_correlation": correlation,
    }


def stats_table(title: str, stats: dict[str, Any], digits: int = 2) -> str:
    rows = []
    for key, value in stats.items():
        label = key.replace("_", " ").title()
        if isinstance(value, dict):
            value_html = ", ".join(f"{sub_key}: {fmt_number(sub_value, digits)}" for sub_key, sub_value in value.items())
        else:
            value_html = fmt_number(value, digits)
        rows.append(f"<tr><th>{html.escape(label)}</th><td>{html.escape(value_html)}</td></tr>")
    return f"""
    <section class="card">
      <h3>{html.escape(title)}</h3>
      <table class="stats-table">
        {''.join(rows)}
      </table>
    </section>
    """


def _require_plotting_stack() -> None:
    if plt is None or sns is None:
        raise RuntimeError(
            "matplotlib/seaborn are required for offline report generation in build_visualizer.py. "
            "Install them in the interpreter used to run this script."
        )


def _plot_to_base64(fig: Any) -> str:
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _style_axes(ax: Any, title: str, xlabel: str = "Timestamp", ylabel: str = "") -> None:
    ax.set_title(title, color="#ecf1ff", fontsize=12, loc="left", pad=12)
    ax.set_facecolor("#111831")
    ax.figure.set_facecolor("#111831")
    ax.grid(True, color="#26345f", alpha=0.35, linewidth=0.8)
    ax.tick_params(colors="#9fb0d8", labelsize=8)
    ax.xaxis.label.set_color("#9fb0d8")
    ax.yaxis.label.set_color("#9fb0d8")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    for spine in ax.spines.values():
        spine.set_color("#26345f")


def _chart_card(title: str, image_b64: str, wide: bool = False) -> str:
    classes = "card chart-card wide-card" if wide else "card chart-card"
    return f"""
    <section class="{classes}">
      <h3>{html.escape(title)}</h3>
      <img alt="{html.escape(title)}" src="data:image/png;base64,{image_b64}" />
    </section>
    """


def _build_overall_chart_cards(report: dict[str, Any]) -> list[str]:
    overall = report["overall"]
    products = report["products"]
    cards: list[str] = []

    fig, ax = plt.subplots(figsize=(11.5, 4.3))
    palette = ["#4cc9f0", "#f72585", "#ffd166", "#52b788"]
    for idx, (product, payload) in enumerate(products.items()):
        ax.plot(payload["series"]["timestamps"], payload["series"]["pnl"], label=product, color=palette[idx % len(palette)], linewidth=1.8)
    ax.plot(overall["timestamps"], overall["total_pnl"], label="Total", color="#ffffff", linewidth=2.4)
    _style_axes(ax, "Cumulative PnL", ylabel="PnL")
    ax.legend(frameon=False, fontsize=8, labelcolor="#ecf1ff")
    cards.append(_chart_card("Cumulative PnL", _plot_to_base64(fig), wide=True))

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    names = list(overall["final_pnls"].keys())
    axes[0].bar(names, list(overall["final_pnls"].values()), color=palette[: len(names)])
    _style_axes(axes[0], "Final PnL by Product", xlabel="", ylabel="PnL")
    axes[1].bar(list(overall["trade_counts"].keys()), list(overall["trade_counts"].values()), color="#4cc9f0")
    _style_axes(axes[1], "Submission Trade Count", xlabel="", ylabel="Trades")
    axes[2].bar(list(overall["submission_volume"].keys()), list(overall["submission_volume"].values()), color="#ffd166")
    _style_axes(axes[2], "Submission Volume", xlabel="", ylabel="Volume")
    cards.append(_chart_card("Overall Cross-Section", _plot_to_base64(fig), wide=True))
    return cards


def _build_product_chart_cards(product: str, payload: dict[str, Any]) -> list[str]:
    series = payload["series"]
    trades = payload["trades"]
    submission_trades = trades["submission_trades"]
    cards: list[str] = []

    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.plot(series["timestamps"], series["mid"], color="#4cc9f0", linewidth=1.8, label="Mid")
    buy_x = trades["buy_markers"]["x"]
    if buy_x:
        ax.scatter(buy_x, trades["buy_markers"]["y"], s=[max(18, v * 12) for v in trades["buy_markers"]["size"]], color="#52b788", alpha=0.7, label="Buys")
    sell_x = trades["sell_markers"]["x"]
    if sell_x:
        ax.scatter(sell_x, trades["sell_markers"]["y"], s=[max(18, v * 12) for v in trades["sell_markers"]["size"]], color="#e63946", alpha=0.7, label="Sells")
    _style_axes(ax, f"{product} Mid Price and Fills", ylabel="Price")
    ax.legend(frameon=False, fontsize=8, labelcolor="#ecf1ff")
    cards.append(_chart_card(f"{product} Mid Price and Fills", _plot_to_base64(fig), wide=True))

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    axes[0].plot(series["timestamps"], series["pnl"], color="#ffd166", linewidth=1.8)
    _style_axes(axes[0], "PnL", ylabel="PnL")
    axes[1].plot(series["timestamps"], series["inventory"], color="#f72585", linewidth=1.8)
    _style_axes(axes[1], "Inventory", ylabel="Units")
    axes[2].plot(series["timestamps"], series["spread"], color="#ffd166", linewidth=1.5, label="Spread")
    twin = axes[2].twinx()
    twin.plot(series["timestamps"], series["imbalance"], color="#4cc9f0", linewidth=1.5, label="Imbalance")
    _style_axes(axes[2], "Spread and Imbalance", ylabel="Spread")
    twin.tick_params(colors="#9fb0d8", labelsize=8)
    twin.spines["right"].set_color("#26345f")
    twin.yaxis.label.set_color("#9fb0d8")
    twin.set_ylabel("Imbalance")
    cards.append(_chart_card(f"{product} PnL, Inventory, Spread/Imbalance", _plot_to_base64(fig), wide=True))

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    axes[0].plot(series["timestamps"], series["top_bid_depth"], color="#52b788", linewidth=1.4, label="Bid 1")
    axes[0].plot(series["timestamps"], series["top_ask_depth"], color="#e63946", linewidth=1.4, label="Ask 1")
    axes[0].plot(series["timestamps"], series["top3_bid_depth"], color="#2a9d8f", linewidth=1.2, linestyle="--", label="Bid top3")
    axes[0].plot(series["timestamps"], series["top3_ask_depth"], color="#ff6b6b", linewidth=1.2, linestyle="--", label="Ask top3")
    _style_axes(axes[0], "Visible Depth", ylabel="Volume")
    axes[0].legend(frameon=False, fontsize=7, labelcolor="#ecf1ff")
    axes[1].plot(series["timestamps"], series["rolling_vol"], color="#4cc9f0", linewidth=1.6, label="Rolling vol")
    axes[1].plot(series["timestamps"], series["rolling_abs_return"], color="#ffd166", linewidth=1.6, label="Rolling abs return")
    _style_axes(axes[1], "Realized Volatility", ylabel="Value")
    axes[1].legend(frameon=False, fontsize=7, labelcolor="#ecf1ff")
    fill_ts = [trade["timestamp"] for trade in submission_trades]
    fill_px = [trade["price"] for trade in submission_trades]
    fill_colors = ["#52b788" if trade["submission_side"] > 0 else "#e63946" for trade in submission_trades]
    fill_sizes = [max(18, trade["quantity"] * 12) for trade in submission_trades]
    axes[2].scatter(fill_ts, fill_px, c=fill_colors, s=fill_sizes, alpha=0.75)
    _style_axes(axes[2], "Fill Prices", ylabel="Price")
    cards.append(_chart_card(f"{product} Depth, Volatility, Fill Scatter", _plot_to_base64(fig), wide=True))

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    trade_times = list(trades["trade_volume_by_timestamp"].keys())
    trade_volumes = list(trades["trade_volume_by_timestamp"].values())
    axes[0].bar(trade_times, trade_volumes, color="#4cc9f0", width=180)
    twin = axes[0].twinx()
    twin.plot(series["timestamps"], series["inventory"], color="#f72585", linewidth=1.4)
    _style_axes(axes[0], "Submission Flow and Inventory", ylabel="Trade volume")
    twin.tick_params(colors="#9fb0d8", labelsize=8)
    twin.spines["right"].set_color("#26345f")
    twin.set_ylabel("Inventory", color="#9fb0d8")

    sns.histplot(trades["execution_edges"], bins=30, color="#ffd166", ax=axes[1], kde=False)
    sns.histplot(trades["realized_spreads"], bins=30, color="#4cc9f0", ax=axes[1], kde=False, alpha=0.55)
    _style_axes(axes[1], "Execution Edge Distribution", xlabel="Edge", ylabel="Count")

    markout_names = list(trades["markouts"].keys())
    markout_means = [mean(trades["markouts"][key]) or 0.0 for key in markout_names]
    axes[2].bar([f"{key} steps" for key in markout_names], markout_means, color="#52b788")
    _style_axes(axes[2], "Average Markout", xlabel="", ylabel="Markout")
    axes[2].tick_params(axis="x", rotation=30)
    cards.append(_chart_card(f"{product} Flow, Edge Distribution, Markouts", _plot_to_base64(fig), wide=True))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
    axes[0].bar(list(trades["trade_count_by_timestamp"].keys()), list(trades["trade_count_by_timestamp"].values()), color="#f72585", width=180)
    _style_axes(axes[0], "Trade Count by Timestamp", ylabel="Trades")
    sns.histplot([trade["quantity"] for trade in submission_trades], bins=20, color="#4cc9f0", ax=axes[1], kde=False)
    _style_axes(axes[1], "Submission Trade Size Distribution", xlabel="Quantity", ylabel="Count")
    cards.append(_chart_card(f"{product} Trade Intensity and Size Distribution", _plot_to_base64(fig), wide=True))

    return cards


def generate_html(report: dict[str, Any]) -> str:
    _require_plotting_stack()
    sns.set_theme(style="darkgrid")

    product_tables = [stats_table(f"{product} Summary", product_report["summary"]) for product, product_report in report["products"].items()]
    summary_cards = [stats_table("Run Summary", report["meta"]), stats_table("Overall Metrics", report["overall"]["summary"]), *product_tables]
    overall_charts = _build_overall_chart_cards(report)

    product_sections: list[str] = []
    for product, product_report in report["products"].items():
        cards = "\n".join(_build_product_chart_cards(product, product_report))
        product_sections.append(
            f"""
            <div class="section-head">
              <h2>{html.escape(product)}</h2>
              <div>
                <span class="pill">Execution analytics</span>
                <span class="pill">Microstructure</span>
                <span class="pill">Inventory</span>
              </div>
            </div>
            <div class="grid plots">{cards}</div>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Backtest Visualizer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #111831;
      --panel-2: #172142;
      --text: #ecf1ff;
      --muted: #9fb0d8;
      --accent: #4cc9f0;
      --accent-2: #f72585;
      --accent-3: #ffd166;
      --good: #52b788;
      --bad: #e63946;
      --border: #26345f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "IBM Plex Sans", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(76, 201, 240, 0.15), transparent 30%),
        radial-gradient(circle at top right, rgba(247, 37, 133, 0.14), transparent 24%),
        linear-gradient(180deg, #08101f, #0b1020 30%, #0b1020);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1680px;
      margin: 0 auto;
      padding: 28px 24px 80px;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 34px; letter-spacing: 0.02em; }}
    h2 {{ margin-top: 28px; font-size: 24px; }}
    p, li {{ color: var(--muted); }}
    .lead {{ max-width: 980px; line-height: 1.5; margin-bottom: 20px; }}
    .grid {{ display: grid; gap: 18px; }}
    .cards {{ grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); margin: 24px 0 28px; }}
    .plots {{ grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); }}
    .card {{
      background: linear-gradient(180deg, rgba(23,33,66,0.95), rgba(17,24,49,0.95));
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 22px 40px rgba(0,0,0,0.22);
    }}
    .stats-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    .stats-table th, .stats-table td {{
      padding: 8px 0;
      border-bottom: 1px solid rgba(159,176,216,0.14);
      text-align: left;
      vertical-align: top;
    }}
    .stats-table th {{ color: var(--muted); font-weight: 600; padding-right: 12px; width: 52%; }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 16px;
      margin: 20px 0 12px;
    }}
    .pill {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(76, 201, 240, 0.12);
      color: var(--accent);
      font-size: 12px;
      margin-right: 8px;
      border: 1px solid rgba(76, 201, 240, 0.22);
    }}
    .chart-card img {{
      width: 100%;
      display: block;
      border-radius: 12px;
      background: #111831;
    }}
    .wide-card {{
      grid-column: 1 / -1;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Backtest Visualizer</h1>
    <p class="lead">
      Auto-generated research dashboard for the latest Prosperity backtest log. This report pulls the newest
      <code>.log</code> under <code>backtests</code>, parses market activity and trade history, reconstructs submission flow,
      and surfaces market microstructure, PnL, inventory, execution quality, and markout analytics.
    </p>
    <div class="grid cards">{''.join(summary_cards)}</div>
    <div class="section-head">
      <h2>Overall</h2>
      <div><span class="pill">Latest log</span><span class="pill">{html.escape(report["meta"]["log_name"])}</span></div>
    </div>
    <div class="grid plots">{''.join(overall_charts)}</div>
    {''.join(product_sections)}
  </div>
</body>
</html>"""


def build_report(log_path: Path) -> dict[str, Any]:
    sandbox_text, activities_text, trade_text = load_sections(log_path)
    sandbox_logs = parse_sandbox_logs(sandbox_text)
    activities = parse_activities(activities_text)
    trades = parse_trades(trade_text)

    by_product: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in activities:
        by_product[str(row["product"])].append(row)

    product_reports = {
        product: compute_product_report(product, rows, trades)
        for product, rows in sorted(by_product.items())
    }
    overall = compute_overall_report(product_reports)

    meta_summary = {
        "log_name": log_path.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "products": ", ".join(product_reports),
        "activity_rows": len(activities),
        "trade_rows": len(trades),
        "sandbox_entries": len(sandbox_logs),
        "sandbox_non_empty": sum(
            1 for entry in sandbox_logs if (entry.get("sandboxLog") or "").strip() or (entry.get("lambdaLog") or "").strip()
        ),
    }

    overall_summary = {
        "final_total_pnl": overall["total_pnl"][-1] if overall["total_pnl"] else None,
        "peak_total_pnl": max(overall["total_pnl"]) if overall["total_pnl"] else None,
        "max_total_drawdown": max_drawdown(overall["total_pnl"]) if overall["total_pnl"] else None,
        "return_correlation": overall["return_correlation"],
    }

    return {
        "meta": meta_summary,
        "overall": {**overall, "summary": overall_summary},
        "products": product_reports,
    }


def write_outputs(report: dict[str, Any], directory: Path, log_path: Path) -> tuple[Path, Path]:
    html_path = directory / f"{log_path.stem}_visualizer.html"
    latest_html_path = directory / "latest_visualizer.html"
    data_path = directory / f"{log_path.stem}_visualizer_data.json"
    latest_data_path = directory / "latest_visualizer_data.json"

    html_text = generate_html(report)
    html_path.write_text(html_text, encoding="utf-8")
    latest_html_path.write_text(html_text, encoding="utf-8")

    json_text = json.dumps(report, separators=(",", ":"))
    data_path.write_text(json_text, encoding="utf-8")
    latest_data_path.write_text(json_text, encoding="utf-8")
    return latest_html_path, latest_data_path


def main() -> None:
    log_path = latest_log_path(ROOT)
    report = build_report(log_path)
    html_path, data_path = write_outputs(report, ROOT, log_path)
    print(f"Latest log: {log_path.name}")
    print(f"HTML report: {html_path}")
    print(f"Data dump: {data_path}")


if __name__ == "__main__":
    main()
