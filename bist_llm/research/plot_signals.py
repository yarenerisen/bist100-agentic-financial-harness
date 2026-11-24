"""Visualises a strategy's signals and trades on the price chart, plus its equity curve vs buy-and-hold and XU100.
Usage (from bist_llm/): python -m research.plot_signals ASELS lower_bollinger_touch [--start 2024-01-01 --end 2026-09-10 --hold 5]
Writes docs/figures/<TICKER>_<strategy>.png"""
import argparse, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools import core, backtest   # noqa: E402

INK, MUTED, GRID = "#1f2328", "#6e7781", "#e6e8eb"
PRICE, BAND, BUY, SELL, STRAT, BH, XU = "#24292f", "#8c959f", "#1a7f37", "#cf222e", "#0969da", "#8250df", "#bc4c00"


def plot(ticker, strategy, start="2024-01-01", end="2026-09-10", hold=5, out_dir=None):
    end = str(core._clamp(end).date()); s0 = pd.Timestamp(start)
    ind = core.calculate_indicators(ticker, start, end)
    sig, full = backtest._signals(strategy, ticker, {}, end, [ticker])
    trades, dret, pos = backtest._simulate(sig, full, s0, hold, 0.2, 0.05)
    done = [t for t in trades if "return_net_pct" in t]
    xu = core.get_ohlcv("XU100", "2023-06-01", end).close
    xu = xu[xu.index >= s0]
    bh = ind.close / ind.close.iloc[0]
    eq = (1 + dret.fillna(0)).cumprod()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for ax in (ax1, ax2):
        ax.grid(color=GRID, linewidth=0.8); ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
    ax1.fill_between(ind.index, ind.bb_lo, ind.bb_up, color=BAND, alpha=0.15, linewidth=0, label="Bollinger (20, 2)")
    ax1.plot(ind.index, ind.close, color=PRICE, linewidth=1.2, label=f"{ticker} close (adjusted)")
    ent = pd.to_datetime([t["entry_date"] for t in done]); ext = pd.to_datetime([t["exit_date"] for t in done])
    ax1.scatter(ent, full.open.reindex(ent), marker="^", s=38, color=BUY, zorder=3, label=f"entry (next open), {len(done)} trades")
    ax1.scatter(ext, full.close.reindex(ext), marker="v", s=38, color=SELL, zorder=3, label=f"exit after {hold} days")
    ax1.set_title(f"{ticker} — {strategy.replace('_', ' ')} (educational backtest, not advice)", loc="left", color=INK, fontsize=12)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax2.plot(eq.index, (eq - 1) * 100, color=STRAT, linewidth=1.6, label="strategy (after costs)")
    ax2.plot(bh.index, (bh - 1) * 100, color=BH, linewidth=1.2, label=f"buy and hold {ticker}")
    ax2.plot(xu.index, (xu / xu.iloc[0] - 1) * 100, color=XU, linewidth=1.2, label="XU100")
    ends = sorted([(float(sr.iloc[-1] * 100), sr.index[-1], c) for sr, c in ((eq - 1, STRAT), (bh - 1, BH), (xu / xu.iloc[0] - 1, XU))])
    lo, hi = ax2.get_ylim(); gap = (hi - lo) * 0.06; placed = []
    for y, x, color in ends:                       # push end labels apart so they never overlap
        yy = max(y, placed[-1] + gap) if placed else y
        placed.append(yy)
        ax2.annotate(f"{y:.1f}%", (x, y), xytext=(x, yy), textcoords="data", color=color, fontsize=9, va="center",
                     ha="left", annotation_clip=False)
    ax2.axhline(0, color=MUTED, linewidth=0.8)
    ax2.set_ylabel("cumulative return, %", color=MUTED, fontsize=9)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.text(0.01, 0.005, f"Signal at close t → enter next open → exit close t+{hold}; 0.2% commission + 0.05% slippage per side. Data to {end}.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out_dir = Path(out_dir or ROOT.parent / "docs" / "figures"); out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{ticker}_{strategy}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker"); ap.add_argument("strategy")
    ap.add_argument("--start", default="2024-01-01"); ap.add_argument("--end", default="2026-09-10"); ap.add_argument("--hold", type=int, default=5)
    a = ap.parse_args()
    print(plot(a.ticker.upper(), a.strategy, a.start, a.end, a.hold))
