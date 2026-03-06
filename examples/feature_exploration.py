"""Feature exploration for the SPY pipeline.

Produces four figures:

Figure 1 — Pipeline chart (last 500 bars)
    Panel 1: Close price + Bollinger Bands, dots coloured by triple-barrier label
    Panel 2: RSI with 70/30 reference lines
    Panel 3: MACD line, signal, histogram
    Panel 4: Volume ratio with 1.0 reference line

Figure 2 — Existing feature analysis (all data)
    Top row: Correlation matrix heatmap + ADF stationarity table
    Bottom row: Per-feature distribution, split by label class (+1 / 0 / -1)

Figure 3 — Candidate feature chart (last 500 bars)
    7 panels: ATR%, BB %B, RSI momentum, vol regime, ret 1d / 5d / 20d

Figure 4 — Candidate feature analysis (all data)
    Top row: Correlation matrix + ADF stationarity table
    Bottom rows: Per-feature distribution, split by label class

Run with:
    uv run python examples/feature_exploration.py
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading.data import load_ohlcv
from trading.features import (
    adf_stationarity,
    atr,
    bollinger_bands,
    correlation_matrix,
    ewm_volatility,
    macd,
    rsi,
    volume_ratio,
)
from trading.labeling import triple_barrier_labels

# ── Parameters ────────────────────────────────────────────────────────────────
CHART_BARS = 500        # number of recent bars shown in Figure 1
PT_SL = (2.0, 1.0)     # profit-take / stop-loss multipliers
MAX_HOLD = 20           # triple-barrier max holding period
VOL_SPAN = 20           # ewm_volatility span
BB_PERIOD = 20          # Bollinger period
RSI_PERIOD = 14
MACD_PARAMS = (12, 26, 9)
VR_PERIOD = 20

LABEL_COLORS = {1: "#2ecc71", 0: "#95a5a6", -1: "#e74c3c"}
LABEL_NAMES  = {1: "PT (+1)", 0: "Time (0)", -1: "SL (−1)"}

# ── Load & compute ─────────────────────────────────────────────────────────────
print("Loading data…")
df = load_ohlcv()
close: pd.Series = df["close"]      # type: ignore[assignment]
high:  pd.Series = df["high"]       # type: ignore[assignment]
low:   pd.Series = df["low"]        # type: ignore[assignment]
volume: pd.Series = df["volume"]    # type: ignore[assignment]

print("Computing features…")
vol   = ewm_volatility(close, span=VOL_SPAN)
rsi_s = rsi(close, period=RSI_PERIOD)
bb    = bollinger_bands(close, period=BB_PERIOD)
macd_df = macd(close, *MACD_PARAMS)
atr_s = atr(high, low, close)
vr    = volume_ratio(volume, period=VR_PERIOD)

print("Computing labels…")
labels = triple_barrier_labels(close, vol, pt_sl=PT_SL, max_hold=MAX_HOLD)

# Build aligned feature DataFrame (drop NaN rows for analysis)
features = pd.DataFrame({
    "rsi":          rsi_s,
    "atr":          atr_s,
    "bb_width":     (bb["upper"] - bb["lower"]) / bb["middle"],
    "macd":         macd_df["macd"],
    "volume_ratio": vr,
    "ewm_vol":      vol,
}).join(labels).dropna()

label_col: pd.Series = features["label"]       # type: ignore[assignment]
feat_cols = [c for c in features.columns if c != "label"]

# Slice for chart
chart = pd.DataFrame({
    "close": close,
    "upper": bb["upper"],
    "middle": bb["middle"],
    "lower": bb["lower"],
    "rsi":   rsi_s,
    "macd_line":   macd_df["macd"],
    "macd_signal": macd_df["signal"],
    "macd_hist":   macd_df["histogram"],
    "vr":    vr,
    "label": labels,
}).iloc[-CHART_BARS:]

# ── Figure 1: Pipeline chart ───────────────────────────────────────────────────
print("Plotting Figure 1…")
fig1, axes = plt.subplots(
    4, 1, figsize=(16, 14),
    gridspec_kw={"height_ratios": [4, 1.5, 1.5, 1.5]},
    sharex=True,
)
fig1.suptitle(
    f"SPY — Triple-Barrier Labels (PT×{PT_SL[0]}, SL×{PT_SL[1]}, hold={MAX_HOLD}d)",
    fontsize=14, fontweight="bold",
)

# Panel 1: Price + BB + labels
ax = axes[0]
ax.fill_between(chart.index, chart["upper"], chart["lower"],
                alpha=0.15, color="steelblue", label="BB band")
ax.plot(chart.index, chart["close"],  color="black",     linewidth=0.8, label="Close")
ax.plot(chart.index, chart["upper"],  color="steelblue", linewidth=0.6, linestyle="--")
ax.plot(chart.index, chart["middle"], color="steelblue", linewidth=0.6, alpha=0.5)
ax.plot(chart.index, chart["lower"],  color="steelblue", linewidth=0.6, linestyle="--")

for lbl, color in LABEL_COLORS.items():
    mask = chart["label"] == lbl
    ax.scatter(chart.index[mask], chart["close"][mask],
               color=color, s=12, zorder=5, label=LABEL_NAMES[lbl], alpha=0.7)

ax.set_ylabel("Price (USD)")
ax.legend(loc="upper left", fontsize=8, ncol=4)
ax.grid(alpha=0.3)

# Panel 2: RSI
ax = axes[1]
ax.plot(chart.index, chart["rsi"], color="purple", linewidth=0.8)
ax.axhline(70, color="red",   linestyle="--", linewidth=0.7, alpha=0.7)
ax.axhline(30, color="green", linestyle="--", linewidth=0.7, alpha=0.7)
ax.axhline(50, color="gray",  linestyle=":",  linewidth=0.5, alpha=0.5)
ax.set_ylabel("RSI")
ax.set_ylim(0, 100)
ax.grid(alpha=0.3)

# Panel 3: MACD
ax = axes[2]
hist = chart["macd_hist"]
colors = np.where(hist >= 0, "#2ecc71", "#e74c3c")
ax.bar(chart.index, hist, color=colors, width=1.0, alpha=0.7)
ax.plot(chart.index, chart["macd_line"],   color="blue",   linewidth=0.8, label="MACD")
ax.plot(chart.index, chart["macd_signal"], color="orange", linewidth=0.8, label="Signal")
ax.axhline(0, color="black", linewidth=0.4)
ax.set_ylabel("MACD")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)

# Panel 4: Volume ratio
ax = axes[3]
ax.plot(chart.index, chart["vr"], color="teal", linewidth=0.8)
ax.axhline(1.0, color="black", linestyle="--", linewidth=0.6, alpha=0.6)
ax.set_ylabel("Vol Ratio")
ax.set_xlabel("Date")
ax.grid(alpha=0.3)

fig1.tight_layout()
fig1.savefig("examples/pipeline_chart.png", dpi=150, bbox_inches="tight")
print("  → saved examples/pipeline_chart.png")

# ── Figure 2: Feature analysis ────────────────────────────────────────────────
print("Plotting Figure 2…")
n_feats = len(feat_cols)
fig2 = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, n_feats, figure=fig2, hspace=0.45, wspace=0.35)

fig2.suptitle("Feature Analysis — SPY (all data, NaN dropped)", fontsize=14, fontweight="bold")

# Top-left: correlation matrix
ax_corr = fig2.add_subplot(gs[0, :3])
corr = correlation_matrix(features[feat_cols])
im = ax_corr.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
ax_corr.set_xticks(range(n_feats))
ax_corr.set_yticks(range(n_feats))
ax_corr.set_xticklabels(feat_cols, rotation=45, ha="right", fontsize=8)
ax_corr.set_yticklabels(feat_cols, fontsize=8)
for i in range(n_feats):
    for j in range(n_feats):
        ax_corr.text(j, i, f"{corr.values[i, j]:.2f}",
                     ha="center", va="center", fontsize=7,
                     color="white" if abs(corr.values[i, j]) > 0.5 else "black")
plt.colorbar(im, ax=ax_corr, shrink=0.8)
ax_corr.set_title("Pearson Correlation Matrix", fontsize=10)

# Top-right: ADF stationarity table
ax_adf = fig2.add_subplot(gs[0, 3:])
ax_adf.axis("off")
adf_rows = []
for col in feat_cols:
    s: pd.Series = features[col]    # type: ignore[assignment]
    res = adf_stationarity(s)
    adf_rows.append([
        col,
        f"{res['stat']:.3f}",
        f"{res['p_value']:.4f}",
        "YES" if res["is_stationary"] else "NO",
    ])
table = ax_adf.table(
    cellText=adf_rows,
    colLabels=["Feature", "ADF stat", "p-value", "Stationary?"],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.6)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif col == 3 and row > 0:
        val = adf_rows[row - 1][3]
        cell.set_facecolor("#d5f5e3" if val == "YES" else "#fadbd8")
ax_adf.set_title("ADF Stationarity (H₀: unit root)", fontsize=10)

# Bottom row: per-feature distributions split by label
label_style = {
     1: dict(color=LABEL_COLORS[ 1], alpha=0.6, label=LABEL_NAMES[ 1]),
     0: dict(color=LABEL_COLORS[ 0], alpha=0.6, label=LABEL_NAMES[ 0]),
    -1: dict(color=LABEL_COLORS[-1], alpha=0.6, label=LABEL_NAMES[-1]),
}
for idx, col in enumerate(feat_cols):
    ax_f = fig2.add_subplot(gs[1, idx])
    for lbl in [1, 0, -1]:
        subset = features.loc[label_col == lbl, col].dropna()
        ax_f.hist(subset, bins=40, density=True, **label_style[lbl])
    ax_f.set_title(col, fontsize=9)
    ax_f.set_xlabel("")
    ax_f.tick_params(labelsize=7)
    ax_f.grid(alpha=0.3)
    if idx == 0:
        ax_f.legend(fontsize=7)

fig2.savefig("examples/feature_analysis.png", dpi=150, bbox_inches="tight")
print("  → saved examples/feature_analysis.png")

# ── Candidate features ────────────────────────────────────────────────────────
print("Computing candidate features…")

atr_pct      = atr_s / close * 100
bb_pct_b     = (close - bb["lower"]) / (bb["upper"] - bb["lower"])
rsi_mom      = rsi_s.diff(5)
vol_regime   = vol / vol.rolling(252).median()
ret_1d       = close.pct_change(1) * 100
ret_5d       = close.pct_change(5) * 100
ret_20d      = close.pct_change(20) * 100

cand_names = ["atr_pct", "bb_pct_b", "rsi_mom", "vol_regime",
              "ret_1d", "ret_5d", "ret_20d"]

candidates = pd.DataFrame({
    "atr_pct":    atr_pct,
    "bb_pct_b":   bb_pct_b,
    "rsi_mom":    rsi_mom,
    "vol_regime": vol_regime,
    "ret_1d":     ret_1d,
    "ret_5d":     ret_5d,
    "ret_20d":    ret_20d,
}).join(labels).dropna()

cand_label: pd.Series = candidates["label"]    # type: ignore[assignment]

# Chart slice
cchart = candidates.iloc[-CHART_BARS:]

# ── Figure 3: Candidate feature chart ─────────────────────────────────────────
print("Plotting Figure 3…")

cand_panel_cfg = [
    ("atr_pct",    "ATR % of price",     "steelblue",  {"ref": None}),
    ("bb_pct_b",   "BB %B",              "darkorange",  {"ref": [0.0, 0.5, 1.0]}),
    ("rsi_mom",    "RSI momentum (Δ5d)", "purple",      {"ref": [0.0]}),
    ("vol_regime", "Vol regime",         "firebrick",   {"ref": [1.0]}),
    ("ret_1d",     "Return 1d (%)",      "#27ae60",     {"ref": [0.0]}),
    ("ret_5d",     "Return 5d (%)",      "#16a085",     {"ref": [0.0]}),
    ("ret_20d",    "Return 20d (%)",     "#8e44ad",     {"ref": [0.0]}),
]

fig3, axes3 = plt.subplots(
    len(cand_panel_cfg), 1,
    figsize=(16, 16),
    sharex=True,
    gridspec_kw={"hspace": 0.05},
)
fig3.suptitle("SPY — Candidate Features (last 500 bars)", fontsize=14, fontweight="bold")

for ax, (col, title, color, opts) in zip(axes3, cand_panel_cfg):
    ax.plot(cchart.index, cchart[col], color=color, linewidth=0.8)
    if opts["ref"]:
        for r in opts["ref"]:
            ax.axhline(r, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    # Shade by label
    for lbl, lcolor in LABEL_COLORS.items():
        mask = cchart["label"] == lbl
        ax.scatter(cchart.index[mask], cchart[col][mask],
                   color=lcolor, s=6, zorder=4, alpha=0.5)
    ax.set_ylabel(title, fontsize=8)
    ax.grid(alpha=0.3)

axes3[-1].set_xlabel("Date")
fig3.tight_layout()
fig3.savefig("examples/candidate_chart.png", dpi=150, bbox_inches="tight")
print("  → saved examples/candidate_chart.png")

# ── Figure 4: Candidate feature analysis ──────────────────────────────────────
print("Plotting Figure 4…")

n_cand = len(cand_names)
fig4 = plt.figure(figsize=(20, 11))
gs4 = gridspec.GridSpec(3, n_cand, figure=fig4, hspace=0.55, wspace=0.35)
fig4.suptitle("Candidate Feature Analysis — SPY (all data, NaN dropped)",
              fontsize=14, fontweight="bold")

# Correlation matrix (top-left block, spans first 4 cols)
ax_cc = fig4.add_subplot(gs4[0, :4])
corr_c = correlation_matrix(candidates[cand_names])
im_c = ax_cc.imshow(corr_c.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
ax_cc.set_xticks(range(n_cand))
ax_cc.set_yticks(range(n_cand))
ax_cc.set_xticklabels(cand_names, rotation=45, ha="right", fontsize=7)
ax_cc.set_yticklabels(cand_names, fontsize=7)
for i in range(n_cand):
    for j in range(n_cand):
        ax_cc.text(j, i, f"{corr_c.values[i, j]:.2f}",
                   ha="center", va="center", fontsize=6,
                   color="white" if abs(corr_c.values[i, j]) > 0.5 else "black")
plt.colorbar(im_c, ax=ax_cc, shrink=0.8)
ax_cc.set_title("Pearson Correlation Matrix", fontsize=10)

# ADF table (top-right block)
ax_cadf = fig4.add_subplot(gs4[0, 4:])
ax_cadf.axis("off")
cadf_rows = []
for col in cand_names:
    s2: pd.Series = candidates[col]    # type: ignore[assignment]
    res = adf_stationarity(s2)
    cadf_rows.append([
        col,
        f"{res['stat']:.3f}",
        f"{res['p_value']:.4f}",
        "YES" if res["is_stationary"] else "NO",
    ])
ctable = ax_cadf.table(
    cellText=cadf_rows,
    colLabels=["Feature", "ADF stat", "p-value", "Stationary?"],
    loc="center",
    cellLoc="center",
)
ctable.auto_set_font_size(False)
ctable.set_fontsize(8)
ctable.scale(1.0, 1.5)
for (row, col_idx), cell in ctable.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif col_idx == 3 and row > 0:
        val = cadf_rows[row - 1][3]
        cell.set_facecolor("#d5f5e3" if val == "YES" else "#fadbd8")
ax_cadf.set_title("ADF Stationarity (H₀: unit root)", fontsize=10)

# Distribution panels — rows 1 and 2
label_style = {
     1: dict(color=LABEL_COLORS[ 1], alpha=0.6, label=LABEL_NAMES[ 1]),
     0: dict(color=LABEL_COLORS[ 0], alpha=0.6, label=LABEL_NAMES[ 0]),
    -1: dict(color=LABEL_COLORS[-1], alpha=0.6, label=LABEL_NAMES[-1]),
}
for idx, col in enumerate(cand_names):
    row_idx = 1 + idx // n_cand      # rows 1–2 (all 7 fit in one row here)
    col_idx = idx % n_cand
    ax_d = fig4.add_subplot(gs4[row_idx, col_idx])
    for lbl in [1, 0, -1]:
        subset = candidates.loc[cand_label == lbl, col].dropna()
        ax_d.hist(subset, bins=40, density=True, **label_style[lbl])
    ax_d.set_title(col, fontsize=8)
    ax_d.tick_params(labelsize=7)
    ax_d.grid(alpha=0.3)
    if idx == 0:
        ax_d.legend(fontsize=6)

fig4.savefig("examples/candidate_analysis.png", dpi=150, bbox_inches="tight")
print("  → saved examples/candidate_analysis.png")

# ── Console summary ────────────────────────────────────────────────────────────
total = label_col.value_counts().sort_index()
print("\nLabel distribution:")
for lbl, count in total.items():
    pct = 100 * count / len(label_col)
    print(f"  {LABEL_NAMES[lbl]:12s}: {count:5d}  ({pct:.1f}%)")

print(f"\nExisting feature matrix: {features.shape[0]} rows × {len(feat_cols)} features")
print(f"Candidate feature matrix: {candidates.shape[0]} rows × {len(cand_names)} features")
print("Done.")
