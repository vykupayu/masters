# %%
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from properscoring import crps_ensemble

# %%
# ---------- 1) Load data ----------
path = "pribor_3m_daily.csv"   # uprav cestu dle sebe
df = pd.read_csv(path, parse_dates=["date"])
df = df.sort_values("date").dropna(subset=["3M_PRIBOR"])

# target in decimal (e.g., 5% -> 0.05)
y = (df["3M_PRIBOR"].to_numpy() / 100.0).astype(float)
dates = df["date"].to_numpy()

# ---------- 2) Train/test split ----------
n_test = 120
y_train = y[:-n_test]
y_test  = y[-n_test:]
dates_test = dates[-n_test:]

# lags for train and test
y_lag_train = y_train[:-1]
y_obs_train = y_train[1:]

y_lag_test = y[-n_test-1:-1]     # lagged values aligned to test
y_obs_test = y_test

# ---------- 3) Fit Bayesian AR(1) with Student-t errors ----------
with pm.Model() as m:
    x = pm.Data("y_lag", y_lag_train)

    alpha = pm.Normal("alpha", mu=0.0, sigma=0.10)      # in decimals, 0.10 ~ 10 p.p.
    rho   = pm.Uniform("rho", lower=-0.999, upper=0.999)
    sigma = pm.HalfNormal("sigma", sigma=0.02)          # 2 p.p. in decimals
    nu    = pm.Exponential("nu_minus2", lam=1/10) + 2   # df > 2

    mu = alpha + rho * x
    y_like = pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=y_obs_train)

    idata = pm.sample(1000, tune=1000, chains=2, target_accept=0.9, progressbar=True)

    # ---------- 4) Posterior predictive for test ----------
    pm.set_data({"y_lag": y_lag_test})
    ppc = pm.sample_posterior_predictive(idata, var_names=["y"], random_seed=123, progressbar=True)

# ppc["y"] has shape (chains, draws, n_test). Convert to (draws_total, n_test)
y_ppc = ppc["y"]
y_ppc = y_ppc.reshape(-1, y_ppc.shape[-1])

# ---------- 5) PIT ----------
pit = (y_ppc <= y_obs_test[None, :]).mean(axis=0)

# ---------- 6) CRPS ----------
# properscoring expects (n_obs, n_draws)
crps = crps_ensemble(y_obs_test, y_ppc.T)

# ---------- 7) Plot (PIT + CRPS) ----------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].hist(pit, bins=20)
ax[0].set_title("PIT histogram (test)")
ax[0].set_xlabel("PIT")
ax[0].set_ylabel("count")

ax[1].plot(dates_test, crps)
ax[1].set_title("CRPS(t) (test)")
ax[1].set_xlabel("date")
ax[1].set_ylabel("CRPS")

plt.tight_layout()
plt.savefig("figures/pit_crps.pdf")
plt.close()

# ---------- 8) Export summary table for LaTeX ----------
stats = pd.Series(y, name="PRIBOR_3M").describe()
# back to percent for readability
stats_pct = (pd.Series(y*100).describe()).to_frame(name="PRIBOR 3M (%)")

stats_pct.to_latex("tables/summary_stats.tex", float_format="%.3f")

# ---------- 9) Save a small software output ----------
summ = az.summary(idata, var_names=["alpha","rho","sigma","nu_minus2"], round_to=3)
summ.to_csv("tables/arviz_summary.csv")

print("Saved figures/pit_crps.pdf and tables/summary_stats.tex")
print(summ)
