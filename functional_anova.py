import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge

def spline_norm(returns_series, n_basis=10, degree=3, alpha=1.0):
    """
    Fit a smoothing spline to the return series and return the L2 norm of the coefficients.
    """
    t = np.linspace(0, 1, len(returns_series))
    X = SplineTransformer(n_knots=n_basis, degree=degree, include_bias=False).fit_transform(t.reshape(-1,1))
    model = Ridge(alpha=alpha)
    model.fit(X, returns_series)
    coef = model.coef_
    return np.linalg.norm(coef)

def regime_effect(returns_series, regime_series):
    """
    Compute absolute difference in mean returns between regime=0 and regime=1.
    """
    if regime_series.sum() == 0 or regime_series.sum() == len(regime_series):
        return 0.0
    mean0 = returns_series[regime_series == 0].mean()
    mean1 = returns_series[regime_series == 1].mean()
    return abs(mean1 - mean0)

def compute_score(returns_df, regime_df, n_basis=10, degree=3, alpha=1.0):
    """
    For each ETF, compute score = spline norm + regime effect.
    Returns dictionary of scores.
    """
    scores = {}
    for etf in returns_df.columns:
        rets = returns_df[etf].dropna().values
        if len(rets) < n_basis + 2:
            continue
        # Align regime to same length
        regime = regime_df.values[-len(rets):]
        spline_n = spline_norm(rets, n_basis, degree, alpha)
        reg_eff = regime_effect(rets, regime)
        scores[etf] = spline_n + reg_eff
    return scores
