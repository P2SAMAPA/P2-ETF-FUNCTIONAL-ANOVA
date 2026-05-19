# Functional ANOVA Engine

Decomposes ETF return curves into functional components using B‑spline basis expansion. Computes an ANOVA‑style score: spline norm (curvature/amplitude) + regime effect (difference in mean returns between high‑ and low‑volatility periods). Higher score indicates a more dynamic return shape and stronger regime sensitivity → overweight signal.

- **Method:** B‑spline smoothing + regime interaction (VIX‑based)
- **Score:** L2 norm of spline coefficients + |mean(high‑vol) - mean(low‑vol)|
- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Output:** top 3 ETFs per universe

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
