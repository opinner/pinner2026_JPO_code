from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal

import pinner2026_jpo_code

INTERIM_DATA_DIR = pinner2026_jpo_code.DATA_DIR / "interim"
DERIVED_DATA_DIR = pinner2026_jpo_code.DATA_DIR / "derived"
FIGURE_DIR: Path = pinner2026_jpo_code.FIGURE_DIR


# LOADING DATA
dissipation_rate_time_series = pd.read_csv(
    DERIVED_DATA_DIR / "wave_induced_dissipation_rate_time_series_at_C.csv", index_col=0
)
dissipation_rate_time_series = dissipation_rate_time_series.set_index(
    pd.to_datetime(dissipation_rate_time_series.index)
).squeeze()

surface_stress = pd.read_csv(INTERIM_DATA_DIR / "daily_momentum_transfer.csv")
surface_stress = surface_stress.set_index(pd.to_datetime(surface_stress["Unnamed: 0"]))
surface_stress = surface_stress.drop("Unnamed: 0", axis="columns")
for _i, col in enumerate(surface_stress.columns):
    surface_stress[col] = surface_stress[col].str.replace("i", "j").apply(lambda x: complex(x))
surface_stress_at_C = surface_stress["C"]

velocity = pd.read_csv(INTERIM_DATA_DIR / "velo_at_C.csv")
velocity = velocity.set_index(pd.to_datetime(velocity["time_2350"]))
velocity = velocity.drop("time_2350", axis="columns")
velocity["2350"] = velocity["2350"].apply(lambda x: complex(x))
# 5 day resampled, absolute value and time shifted, as resample used the left bin edge for labeling
mean_flow = (
    pd.Series(index=velocity.index + pd.Timedelta(hours=48), data=velocity["2350"].values)
    .resample("5D")
    .mean()
    .abs()
)


# inertial_wind = pd.read_csv(INTERIM_DATA_DIR / "inertial_wind.csv")
# inertial_wind = inertial_wind.set_index(pd.to_datetime(inertial_wind["Unnamed: 0"]))
# inertial_wind = inertial_wind.drop("Unnamed: 0", axis="columns")
# for _i, col in enumerate(surface_stress.columns):
#     inertial_wind[col] = inertial_wind[col].str.replace("i", "j").apply(lambda x: complex(x))
# inertial_wind_at_C = inertial_wind["C"]
# inertial_wind_at_C = inertial_wind_at_C.loc[: dissipation_rate_time_series.index[-1]]

cm = 1 / 2.54  # centimeters in inches
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white", "font.size": 7})

fig, ax = plt.subplots(1, figsize=(12 * cm, 5 * cm), layout="constrained")

result_of_Pinner2025 = 1.7809504867405678e-09
# l_paper = ax.axhline(
#     result_of_Pinner2025,
#     color="black",
#     ls="--",
#     label="$\\overline{\\varepsilon_\\text{IGW}}$",
#     alpha=0.5,
# )

# l_avrg = ax.axhline(
#     np.mean(dissipation_rate_time_series),
#     ls="-",
#     label="$\\langle\\varepsilon_\\text{IGW}(t)\\rangle$",
#     color="black",
#     alpha=0.5,
# )

(l_eps,) = ax.plot(
    dissipation_rate_time_series.index,
    dissipation_rate_time_series.to_numpy(),
    label="$\\varepsilon_\\text{IW}(t)$",
    lw=1.5,
    color="k",
)

ax.set_ylabel("wave-induced dissipation rate\n$\\varepsilon_{\\mathrm{IW}}\\,$(W$\\,$kg$^{-1}$)")

locator = mdates.AutoDateLocator(minticks=8, maxticks=12)
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

ax2 = ax.twinx()
ax2.set_zorder(-1)
ax2.set_ylabel("surface stress $\\tau$ / (N$\\,$m$^{-2}$)")
ax2.yaxis.label.set_color("#d95f02")
ax.set_facecolor("None")
(l_tau,) = ax2.plot(
    surface_stress_at_C.index,
    surface_stress_at_C.abs().to_numpy(),
    color="#d95f02",
    # ls="--",
    label="$\\tau(t)$",
    lw=1.5,
)


def smooth(array, length):
    window = signal.windows.hann(length)  # construct Hanning window
    # smooth by convolving speed with window
    smoothed = signal.convolve(array, window, mode="same") / np.sum(window)
    return smoothed


ax3 = ax.twinx()
ax3.spines["right"].set_position(("axes", 1.2))
ax3.set_ylabel("mean flow $|\\overline{u}|$ / (m/s)")
ax3.yaxis.label.set_color("#7570b3")
ax3.set_zorder(-2)
ax2.set_facecolor("None")
(l_uv,) = ax3.plot(
    mean_flow.index,
    mean_flow.to_numpy(),
    color="#7570b3",
    label="$|\\overline{u}|$",
    lw=1.5,
    alpha=0.6,
)

# ax3 = ax.twinx()
# ax3.spines["right"].set_position(("axes", 1.2))
# ax3.set_ylabel("inertial wind / (m/s)")
# ax3.set_zorder(-2)
# ax2.set_facecolor("None")
# (l_uv,) = ax3.plot(
#     inertial_wind_at_C.index,
#     smooth(inertial_wind_at_C.abs().to_numpy(), 100),
#     color="#7fc97f",
#     label=" inertial wind",
#     lw=1.5,
#     alpha=0.9,
# )

ax.legend(
    # handles=[l_eps, l_avrg, l_paper, l_tau, l_uv],
    handles=[l_eps, l_tau, l_uv],
    ncol=3,  # 2,
    fontsize=6,
    title_fontsize=6,
    framealpha=0.9,
)

ax.set_title("measured at mooring C, 2350$\\,$m depth, 143$\\,$mab")

fig.savefig(FIGURE_DIR / "f09_defense_simplified.svg")
# plt.show()
