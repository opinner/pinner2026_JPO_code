---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import equations as eq
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.helper
```

```{code-cell} ipython3
nsquared_data = src.helper.load_pickle("../../derived_data/nsquared_data_at_measurements.pkl")
nsquared_data_at_C = nsquared_data["C"][2350]
nsquared_data_at_C
```

```{code-cell} ipython3
def get_mean_N_squared_in_chunk(chunk_time_index, N_squared_series):

    try:
        assert N_squared_series.index[0] <= pd.to_datetime(chunk_time_index[0])
        assert N_squared_series.index[-1] >= pd.to_datetime(chunk_time_index[-1])
        return N_squared_series.loc[chunk_time_index[0] : chunk_time_index[-1]].mean()

    # requested chunk is outside the given time series
    except AssertionError as e:
        print(
            "FROM: available:",
            N_squared_series.index[0],
            "requested:",
            chunk_time_index[0],
            N_squared_series.index[0] - chunk_time_index[0],
        )
        print("TO: available:", N_squared_series.index[-1], "requested:", chunk_time_index[-1])

        # time difference is small, proceed anyway
        if np.abs(N_squared_series.index[0] - chunk_time_index[0]) < pd.Timedelta(days=1):
            return N_squared_series.loc[chunk_time_index[0] : chunk_time_index[-1]].mean()

        else:
            raise e
            # return np.nan
```

```{code-cell} ipython3

```

```{code-cell} ipython3
available_energy_df = pd.read_csv("./available_energy.csv", index_col=0)
available_energy_df
```

```{code-cell} ipython3
# for index, energy in available_energy_df.iterrows():

#     _, indexa, _, indexb, _  = index.split("'")
#     print(pd.to_datetime(indexa) - pd.to_datetime(indexb))
```

```{code-cell} ipython3
latitude_at_mooring_C = -63.66
CORIOLIS_FREQUENCY_IN_CPD = src.helper.get_coriolis_frequency(
    latitude_at_mooring_C, unit="cpd", absolute=True
)
CORIOLIS_FREQUENCY_IN_CPD
coriolis_frequency_in_rads = src.helper.get_coriolis_frequency(
    latitude_at_mooring_C, unit="rad/s", absolute=True
)
```

```{code-cell} ipython3
time_stamps = []
dissipation_rates = []

for index, energy in available_energy_df.iterrows():
    _, indexa, _, indexb, _ = index.split("'")
    mean_N_squared_in_rads = get_mean_N_squared_in_chunk((indexa, indexb), nsquared_data_at_C)
    mean_N_in_rads = np.sqrt(mean_N_squared_in_rads)

    assert len(energy) == 1
    energy = energy.iloc[0]

    eps = eq.calculate_dissipation_rate(
        coriolis_frequency=coriolis_frequency_in_rads,
        buoyancy_frequency=mean_N_in_rads,
        energy_level=energy,
    )
    a = pd.to_datetime(indexa)
    b = pd.to_datetime(indexb)
    time_stamps.append(a + (b - a) / 2)  # time stamp in the middle of chunk
    dissipation_rates.append(eps)

dissipation_rate_time_series = pd.Series(index=time_stamps, data=dissipation_rates)
dissipation_rate_time_series
```

```{code-cell} ipython3
dissipation_rate_time_series.to_csv(
    "../../derived_data/wave_induced_dissipation_rate_time_series_at_C.csv"
)
```

```{code-cell} ipython3
surface_stress = pd.read_csv("../../derived_data/daily_momentum_transfer.csv")
surface_stress.set_index(pd.to_datetime(surface_stress["Unnamed: 0"]), inplace=True)
surface_stress.drop("Unnamed: 0", inplace=True, axis="columns")
# momentum_transfer = momentum_transfer.loc[smoothed_hke.index[0]:]
for i, col in enumerate(surface_stress.columns):
    surface_stress[col] = surface_stress[col].str.replace("i", "j").apply(lambda x: complex(x))
surface_stress["C"]
```

```{code-cell} ipython3
cm = 1 / 2.54  # centimeters in inches
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white", "font.size": 7})

fig, ax = plt.subplots(1, figsize=(14 * cm, 5 * cm), layout="constrained")

result_of_Pinner2025 = 1.7809504867405678e-09
l_paper = ax.axhline(
    result_of_Pinner2025,
    color="black",
    ls="--",
    label="$\\overline{\\varepsilon_\\text{IGW}}$",
    alpha=0.5,
)
l_avrg = ax.axhline(
    np.mean(dissipation_rates),
    ls="-",
    label="$\\langle\\varepsilon_\\text{IGW}(t)\\rangle$",
    color="black",
    alpha=0.5,
)
(l_eps,) = ax.plot(
    dissipation_rate_time_series.index,
    dissipation_rate_time_series.values,
    label="$\\varepsilon_\\text{IGW}(t)$",
    lw=2,
)
ax.set_ylabel("Wave-induced dissipation rate\n$\\varepsilon_{\\mathrm{IGW}}\\,$(W$\\,$kg$^{-1}$)")

locator = mdates.AutoDateLocator(minticks=8, maxticks=12)
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

ax2 = ax.twinx()
ax2.set_zorder(-1)
ax.set_facecolor("None")
(l_tau,) = ax2.plot(surface_stress["C"].abs(), color="tab:red", label="$\\tau(t)$", lw=2)
ax2.set_ylabel("surface stress\n$|\\tau|$ / (N$\\,$m$^{-2}$)")

ax.legend(
    handles=[l_eps, l_avrg, l_paper, l_tau],
    title="measured at mooring C, 2350$\\,$m depth, 143$\\,$mab",
    ncol=4,
    fontsize=6,
    framealpha=0.9,
)
fig.savefig("dissipation_rate_at_mooring_C.pdf")
```

```{code-cell} ipython3
break
```

```{code-cell} ipython3
cm = 1 / 2.54  # centimeters in inches
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white", "font.size": 7})

# plt.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

fig, ax = plt.subplots(1, figsize=(14 * cm, 5 * cm), layout="constrained")

result_of_Pinner2025 = 1.7809504867405678e-09
l_paper = ax.axhline(result_of_Pinner2025, color="black", ls="--", label="Pinner2025", alpha=0.5)
l_avrg = ax.axhline(
    np.mean(dissipation_rates),
    ls="-",
    label="$\\langle\varepsilon_\text{IGW}(t)\rangle$",
    color="black",
    alpha=0.5,
)
(l_eps,) = ax.plot(
    dissipation_rate_time_series.index,
    dissipation_rate_time_series.values,
    label="$\varepsilon_\text{IGW}(t)$",
    lw=2,
)
ax.set_ylabel(
    "Wave-induced dissipation rate\newline$\varepsilon_{\\mathrm{IGW}}\\,$(\\mathrm{W}$\\,$kg$^{-1}$)"
)

locator = mdates.AutoDateLocator(minticks=8, maxticks=12)
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

ax2 = ax.twinx()
ax2.set_zorder(-1)
ax.set_facecolor("None")
(l_tau,) = ax2.plot(surface_stress["C"].abs(), color="tab:red", label="$\tau(t)$", lw=2)
ax2.set_ylabel("surface stress\n$|\tau|$ / (N$\\,$m$^{-2}$)")

ax.legend(
    handles=[l_eps, l_avrg, l_paper, l_tau],
    title=r"measured at mooring C, 2350$\,$m depth, 143$\,$mab",
    ncol=4,
    fontsize=6,
    framealpha=0.9,
)
# fig.savefig("dissipation_rate_at_mooring_C.pgf")
```

```{code-cell} ipython3
# plt.plot(dissipation_rate_time_series.index,".")
# plt.plot(surface_stress["C"].index,".")
print(dissipation_rate_time_series.index)
print(surface_stress["C"].index[8:])
```
