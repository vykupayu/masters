# %%
# Načtení knihoven
import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
from properscoring import crps_ensemble

# %%

CSV_PATH = "pribor_3m_daily.csv"

N_WINDOWS = 60
TRAIN_LEN = 500         
VI_ITERS = 1200         
VI_DRAWS = 500
FIX_NU = True
NU_FIXED = 8.0
SEED = 123


df = pd.read_csv(CSV_PATH, parse_dates=["date"])
df = df.sort_values("date").dropna(subset=["3M_PRIBOR"])
y = (df["3M_PRIBOR"].to_numpy(dtype=float) / 100.0)
dates = df["date"].to_numpy()

need = TRAIN_LEN + N_WINDOWS + 1
if len(y) < need:
    raise ValueError(f"Moc málo dat: {len(y)} < {need}. Zkrať TRAIN_LEN/N_WINDOWS.")

y_seg = y[-need:]
dates_seg = dates[-need:]
forecast_dates = dates_seg[TRAIN_LEN + 1: TRAIN_LEN + 1 + N_WINDOWS]


# MODEL AR(1) + Student-t

Data = getattr(pm, "MutableData", pm.Data)

with pm.Model() as model:
    y_lag = Data("y_lag", np.zeros(TRAIN_LEN))
    y_obs = Data("y_obs", np.zeros(TRAIN_LEN))

    alpha = pm.Normal("alpha", mu=0.0, sigma=0.05)
    rho_raw = pm.Normal("rho_raw", mu=0.0, sigma=1.0)
    rho = pm.math.tanh(rho_raw)          # (-1,1)

    log_sigma = pm.Normal("log_sigma", mu=np.log(0.01), sigma=1.0)
    sigma = pm.math.exp(log_sigma)

    nu = NU_FIXED if FIX_NU else (pm.Exponential("nu_minus2", lam=1/10) + 2.0)

    mu = alpha + rho * y_lag
    pm.StudentT("y_like", nu=nu, mu=mu, sigma=sigma, observed=y_obs)

def get_draws(obj, name: str) -> np.ndarray:
    # InferenceData 
    if hasattr(obj, "posterior"):
        arr = obj.posterior[name].values  
        return arr.reshape(-1)
    # MultiTrace 
    return obj.get_values(name, combine=True)

def simulate_ens(alpha_s, rho_raw_s, log_sigma_s, lag_value, nu_value, seed):
    rng = np.random.default_rng(seed)
    rho_s = np.tanh(rho_raw_s)
    sigma_s = np.exp(log_sigma_s)
    mu_pred = alpha_s + rho_s * lag_value

    z = rng.standard_normal(size=mu_pred.shape[0])
    u = rng.chisquare(df=nu_value, size=mu_pred.shape[0])
    t = z / np.sqrt(u / nu_value)

    return mu_pred + sigma_s * t


pit_advi = np.zeros(N_WINDOWS)
crps_advi = np.zeros(N_WINDOWS)

pit_fr = np.zeros(N_WINDOWS)
crps_fr = np.zeros(N_WINDOWS)

for j in range(N_WINDOWS):
    block = y_seg[j : j + TRAIN_LEN + 1]
    lag_train = block[:-1]
    obs_train = block[1:]

    lag_pred = y_seg[j + TRAIN_LEN]
    y_true = y_seg[j + TRAIN_LEN + 1]

    pm.set_data({"y_lag": lag_train, "y_obs": obs_train}, model=model)

    # ---- ADVI ----
    with model:
        approx_mf = pm.fit(
            n=VI_ITERS,
            method=pm.ADVI(),
            random_seed=SEED + 1000 + j,
            progressbar=False
        )
        trace_mf = approx_mf.sample(VI_DRAWS,
                                    random_seed=SEED + 2000 + j, return_inferencedata=True)

    alpha_mf = get_draws(trace_mf, "alpha")
    rho_raw_mf = get_draws(trace_mf, "rho_raw")
    log_sigma_mf = get_draws(trace_mf, "log_sigma")

    ens_mf = simulate_ens(alpha_mf, rho_raw_mf, log_sigma_mf, lag_pred, NU_FIXED, seed=SEED + 3000 + j)
    pit_advi[j] = np.mean(ens_mf <= y_true)
    crps_advi[j] = crps_ensemble(np.array([y_true]), ens_mf[None, :])[0]

    # ---- FullRankADVI ----
    with model:
        approx_fr = pm.fit(
            n=VI_ITERS,
            method=pm.FullRankADVI(),
            random_seed=SEED + 4000 + j,
            progressbar=False
        )
        trace_fr = approx_fr.sample(VI_DRAWS, random_seed=SEED + 5000 + j)

    alpha_fr = get_draws(trace_fr, "alpha")
    rho_raw_fr = get_draws(trace_fr, "rho_raw")
    log_sigma_fr = get_draws(trace_fr, "log_sigma")

    ens_fr = simulate_ens(alpha_fr, rho_raw_fr, log_sigma_fr, lag_pred, NU_FIXED, seed=SEED + 6000 + j)
    pit_fr[j] = np.mean(ens_fr <= y_true)
    crps_fr[j] = crps_ensemble(np.array([y_true]), ens_fr[None, :])[0]

# %%

fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].hist(pit_advi, bins=12, alpha=0.6, label="VI (ADVI)")
ax[0].hist(pit_fr, bins=12, alpha=0.6, label="VI (FullRankADVI)")
ax[0].set_title("PIT histogram (60 rolling-window validací bodů)")
ax[0].set_xlabel("PIT"); ax[0].set_ylabel("počet")
ax[0].legend()

ax[1].plot(forecast_dates, crps_advi, label="VI (ADVI)")
ax[1].plot(forecast_dates, crps_fr, label="VI (FullRankADVI)")
ax[1].set_title("CRPS v čase (\\metoda \"1-step ahead\")")
ax[1].set_xlabel("datum"); ax[1].set_ylabel("CRPS")
ax[1].legend()

plt.tight_layout()
plt.savefig("pit_crps_compare.pdf")
plt.close()

# %%

summary = pd.DataFrame({
    "metoda": ["VI (ADVI)", "VI (FullRankADVI)"],
    "průměrná hodnota CRPS": [crps_advi.mean(), crps_fr.mean()],
    "průměrná hodnota PIT": [pit_advi.mean(), pit_fr.mean()],
})
summary.to_latex("rolling_metrics.tex", index=False, float_format="%.4f")

print("Saved: pit_crps_compare.pdf")
print("Saved: rolling_metrics.tex")
print(summary)


# %%
