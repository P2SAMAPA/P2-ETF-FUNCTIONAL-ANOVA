import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-functional-anova-results"

MACRO_COLUMNS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ]
}

# Rolling windows (days)
WINDOWS = [63, 252, 504, 1008, 2016]

# B‑spline parameters
N_BASIS = 10                     # number of B‑spline basis functions
DEGREE = 3                       # cubic B‑spline

# Regime definition: use VIX > median as 'high_vol' else 'low_vol' (or use HHMM regime)
REGIME_COL = "VIX"               # macro column to define regime
REGIME_MEDIAN_WINDOW = 252       # window for median threshold

TOP_N = 3
