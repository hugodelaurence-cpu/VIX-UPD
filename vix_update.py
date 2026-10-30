import pandas as pd
import numpy as np
import yfinance as yf

# =====================================================
# DOWNLOAD DATA
# =====================================================

START = "1987-01-01"
END   = "2026-10-30"

print("Downloading S&P 500...")
spx = yf.download("^GSPC", start=START, end=END, auto_adjust=True)

print("Downloading VIX...")
vix = yf.download("^VIX", start="1990-01-01", end=END)

print("Downloading VXO...")
vxo = yf.download("^VXO", start=START, end=END)

# =====================================================
# BUILD VOL SERIES
# =====================================================

vol = pd.DataFrame(index=spx.index)

vol["VIX"] = vix["Close"]
vol["VXO"] = vxo["Close"]

# Use VXO before VIX exists
vol["VOL"] = vol["VIX"].combine_first(vol["VXO"])

# =====================================================
# SPX RETURNS
# =====================================================

df = pd.DataFrame(index=spx.index)

df["SPX"] = spx["Close"]

df["Return"] = (
    df["SPX"].pct_change() * 100
)

# Prior day's VIX
df["VIX"] = vol["VOL"].shift(1)

df = df.dropna()

# =====================================================
# RETURN BUCKETS
# =====================================================

return_bins = [
    -999,
    -20,
    -15,
    -10,
    -7,
    -5,
    -4,
    -3,
    -2,
    -1,
     0,
    999
]

return_labels = [
    "<-20",
    "-20 to -15",
    "-15 to -10",
    "-10 to -7",
    "-7 to -5",
    "-5 to -4",
    "-4 to -3",
    "-3 to -2",
    "-2 to -1",
    "-1 to 0",
    ">0"
]

df["ReturnBucket"] = pd.cut(
    df["Return"],
    bins=return_bins,
    labels=return_labels,
    include_lowest=True
)

# =====================================================
# VIX BUCKETS
# =====================================================

vix_bins = [
    0,
    10,
    12,
    15,
    20,
    25,
    30,
    40,
    50,
    999
]

vix_labels = [
    "<10",
    "10 to 12",
    "12 to 15",
    "15 to 20",
    "20 to 25",
    "25 to 30",
    "30 to 40",
    "40 to 50",
    "50+"
]

df["VIXBucket"] = pd.cut(
    df["VIX"],
    bins=vix_bins,
    labels=vix_labels,
    include_lowest=True
)

# =====================================================
# EMPIRICAL PROBABILITY MATRIX
# =====================================================

counts = pd.crosstab(
    df["ReturnBucket"],
    df["VIXBucket"]
)

probs = counts.div(counts.sum(axis=0), axis=1) * 100

probs = probs.round(2)

print("\n")
print("="*80)
print("CONDITIONAL PROBABILITY MATRIX")
print("="*80)
print(probs)

# Save
probs.to_csv(
    "conditional_probability_matrix_1987_2026.csv"
)

# =====================================================
# WORST DAILY DECLINES
# =====================================================

worst = (
    df[df["Return"] <= -5]
    [["Return","VIX"]]
    .sort_values("Return")
)

print("\n")
print("="*80)
print("WORST DAILY DECLINES")
print("="*80)
print(worst)

worst.to_csv(
    "worst_sp500_declines_1987_2026.csv"
)

# =====================================================
# ARTICLE-STYLE TABLE IMAGE
# =====================================================

import matplotlib.pyplot as plt

fig, ax = plt.subplots(
    figsize=(12,6)
)

ax.axis("off")

table = ax.table(
    cellText=probs.values,
    rowLabels=probs.index,
    colLabels=probs.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.15, 1.5)

plt.title(
    "Empirical probabilities (%)\n"
    "S&P 500 Returns (rows) vs Previous-Day VIX (columns)\n"
    "01/02/1987 - 10/29/2026",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    "ern_probability_matrix_1987_2026.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nSaved:")
print("  conditional_probability_matrix_1987_2026.csv")
print("  worst_sp500_declines_1987_2026.csv")
print("  ern_probability_matrix_1987_2026.png")
