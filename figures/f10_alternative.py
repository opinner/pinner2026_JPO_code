# TODO a \star b order!!!

import pickle
import warnings
from pathlib import Path

# import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal

import pinner2026_jpo_code
import pinner2026_jpo_code.cross_correlation as cc
import pinner2026_jpo_code.plots

warnings.filterwarnings("error")

ONE_COLUMN_WIDTH = pinner2026_jpo_code.plots.ONE_COLUMN_WIDTH
TWO_COLUMN_WIDTH = pinner2026_jpo_code.plots.TWO_COLUMN_WIDTH

INTERIM_DATA_DIR: Path = pinner2026_jpo_code.DATA_DIR / "interim"
DERIVED_DATA_DIR: Path = pinner2026_jpo_code.DATA_DIR / "derived"
FIGURE_DIR: Path = pinner2026_jpo_code.FIGURE_DIR

# ----------------------------------------------
# ----- HELPER ---------------------------
# ----------------------------------------------


# def smooth(array, length):
#     window = signal.windows.hann(length)  # construct Hanning window
#     # smooth by convolving speed with window
#     smoothed = signal.convolve(array, window, mode="same") / np.sum(window)
#     return smoothed


# ----------------------------------------------
# ----- LOADING DATA ---------------------------
# ----------------------------------------------

dissipation_rate_time_series = pd.read_csv(
    DERIVED_DATA_DIR / "wave_induced_dissipation_rate_time_series_at_C.csv", index_col=0
)
dissip_at_C = (
    dissipation_rate_time_series.set_index(pd.to_datetime(dissipation_rate_time_series.index))
    .squeeze()
    .rolling("14D")
    .mean()
)

# ---------------- Row 1: Local parameters ----------------

hke_all = pd.read_csv(INTERIM_DATA_DIR / "HKE.csv", index_col=0, header=[0, 1])
hke_all.index = pd.to_datetime(hke_all.index).floor("D")
hke_at_C = hke_all["C", "2350"].rolling("14D").mean()

velocity = pd.read_csv(INTERIM_DATA_DIR / "velo_at_C.csv")
velocity = velocity.set_index(pd.to_datetime(velocity["time_2350"]))
velocity = velocity.drop("time_2350", axis="columns")
velocity["2350"] = velocity["2350"].apply(lambda x: complex(x))
# 5-day resampling of the absolute value
# time shifted, as resample uses left bin edge for timestamping/labeling
mean_flow = (
    pd.Series(index=velocity.index + pd.Timedelta(hours=48), data=velocity["2350"].to_numpy())
    .abs()
    .resample("5D")
    .mean()
    .rolling("14D")
    .mean()
)


N2_at_C = pd.read_csv(INTERIM_DATA_DIR / "N2_at_C_2350m.csv", index_col=0).squeeze()
# 5-day resampling of the absolute value
# time shifted, as resample uses left bin edge for timestamping/labeling
N2_at_C.index = pd.to_datetime(N2_at_C.index) + pd.Timedelta(hours=48)
N2_at_C = N2_at_C.resample("5D").mean().rolling("14D").mean()


# ---------------- Row 2: Remote parameters ----------------

surface_stress = pd.read_csv(INTERIM_DATA_DIR / "daily_momentum_transfer.csv")
surface_stress = surface_stress.set_index(pd.to_datetime(surface_stress["Unnamed: 0"]))
surface_stress = surface_stress.drop("Unnamed: 0", axis="columns")
for _i, col in enumerate(surface_stress.columns):
    surface_stress[col] = surface_stress[col].str.replace("i", "j").apply(lambda x: complex(x))
surface_stress_at_C = surface_stress["C"].iloc[8:-76].abs().rolling("14D").mean()
surface_stress_at_C = surface_stress_at_C.squeeze()

inertial_wind = pd.read_csv(INTERIM_DATA_DIR / "inertial_wind.csv")
inertial_wind = inertial_wind.set_index(pd.to_datetime(inertial_wind["Unnamed: 0"]))
inertial_wind = inertial_wind.drop("Unnamed: 0", axis="columns")
for _i, col in enumerate(surface_stress.columns):
    inertial_wind[col] = inertial_wind[col].str.replace("i", "j").apply(lambda x: complex(x))
inertial_wind_at_C = inertial_wind["C"]
inertial_wind_at_C.index = pd.to_datetime(inertial_wind_at_C.index) + pd.Timedelta(hours=48)
inertial_wind_at_C = (
    inertial_wind_at_C.loc[: dissip_at_C.index[-1]]
    .abs()
    .resample("5D")
    .mean()
    .rolling("14D")
    .mean()
)  # pyright: ignore[reportIndexIssue, reportAttributeAccessIssue]

# ----------------------------------------------
# ----- PLOTTING -------------------------------
# ----------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(TWO_COLUMN_WIDTH, TWO_COLUMN_WIDTH * 0.6),
    width_ratios=[2, 1],
    layout="constrained",
    sharex="col",
    sharey="col",
)

ax_local_ts, ax_local_xc = axes[0]
ax_remote_ts, ax_remote_xc = axes[1]

# ---------------- Row 1: Local parameters ----------------
(l_eps,) = ax_local_ts.plot(
    dissip_at_C.index,  # type: ignore
    dissip_at_C.to_numpy(),  # type: ignore
    label="$\\varepsilon_\\text{IGW}$",
    lw=2,
    color="k",
)

ax1 = ax_local_ts.twinx()
ax1.spines["right"].set_position(("axes", 1.33))
ax1.set_ylabel("HKE / (m$^2$ s$^{-2}$)")
ax1.yaxis.label.set_color("#d3144d")
ax1.tick_params(axis="y", colors="#d3144d")
ax1.spines["right"].set_color("#d3144d")
ax1.set_zorder(-2)
ax_local_ts.set_facecolor("None")
(l_hke,) = ax1.plot(
    hke_at_C.index,
    hke_at_C.to_numpy(),
    color="#d3144d",
    label="$|\\overline{u}|$",
    lw=2,
    alpha=0.6,
)

ax3 = ax_local_ts.twinx()
ax3.spines["right"].set_position(("axes", 1.15))
ax3.set_ylabel("mean flow $|\\overline{u}|$ / (m/s)")
ax3.yaxis.label.set_color("#7570b3")
ax3.tick_params(axis="y", colors="#7570b3")
ax3.spines["right"].set_color("#7570b3")
ax3.set_zorder(-2)
ax_local_ts.set_facecolor("None")
(l_uv,) = ax3.plot(
    mean_flow.index,
    mean_flow.to_numpy(),
    color="#7570b3",
    label="$|\\overline{u}|$",
    lw=2,
    alpha=0.6,
)

ax4 = ax_local_ts.twinx()
ax4.set_ylabel("$N^2$ / (rad$^2$ s$^{-2}$)")
# twinx3.spines['right'].set_position(('axes', 1.4)) # shift by 10% of the figure width
ax4.spines["right"].set_color("#1b9e77")
ax4.set_ylim(1.3e-6, 2.5e-6)
ax4.tick_params(axis="y", colors="#1b9e77")
ax4.yaxis.label.set_color("#1b9e77")
(l_N2,) = ax4.plot(N2_at_C.index, N2_at_C, color="#1b9e77", alpha=0.8, lw=1.5, label="$N^2$")

ax_local_ts.set_title("Local parameters", loc="left")
# ax_local_ts.set_ylabel("wave-induced dissipation rate\n$\\varepsilon_{\\mathrm{IW}}\\,$(W$\\,$kg$^{-1}$)")
ax_local_ts.legend(
    # handles=[l_eps, l_avrg, l_paper, l_tau, l_uv],
    handles=[l_eps, l_hke, l_uv, l_N2],
    ncol=4,
    fontsize=6,
    title_fontsize=6,
    framealpha=0.9,
)

# Cross correlation
ax_local_xc.axhline(0, color="k", alpha=0.3, ls=":")
ax_local_xc.axvline(0, color="k", alpha=0.5, ls="--")

a = hke_at_C.iloc[3:]
b = dissip_at_C.iloc[:-3]
assert np.all(a.index == b.index)
lags, crosscorr = cc.get_normalized_crosscorr(a.to_numpy().flatten(), b.to_numpy().flatten())
time_lags = lags * b.index.diff().mean().days
N = len(b)
label = "$\\text{HKE} \\star \\varepsilon_\\mathrm{IGW}$"
color = "#d3144d"
cc.plot_corr_with_error_bands(
    time_lags,
    crosscorr,
    x_err=5,
    y_err=cc.get_crosscorr_standard_error(crosscorr, N),
    axis=ax_local_xc,
    label=label,
    color=color,
)

# TODO plot looks less smooth here than in the jupyter notebook cross_correlation.ipynb. Why?
a = mean_flow.iloc[3:-3]
b = dissip_at_C
assert np.all(a.index == b.index)
lags, crosscorr = cc.get_normalized_crosscorr(a.to_numpy().flatten(), b.to_numpy().flatten())
time_lags = lags * b.index.diff().mean().days
label = "$|\\overline{u}| \\star \\varepsilon_\\mathrm{IGW}$"
color = "#7570b3"
cc.plot_corr_with_error_bands(
    time_lags,
    crosscorr,
    x_err=5,
    y_err=cc.get_crosscorr_standard_error(crosscorr, len(b)),
    axis=ax_local_xc,
    label=label,
    color=color,
)  # , plot_errorbars=False)

a = N2_at_C.iloc[3:-3]
b = dissip_at_C.iloc[:]
assert np.all(a.index == b.index)
lags, crosscorr = cc.get_normalized_crosscorr(a.to_numpy().flatten(), b.to_numpy().flatten())
time_lags = lags * b.index.diff().mean().days
label = "$N^2 \\star \\varepsilon_\\mathrm{IGW}$"
color = "#1b9e77"
cc.plot_corr_with_error_bands(
    time_lags,
    crosscorr,
    x_err=5,
    y_err=cc.get_crosscorr_standard_error(crosscorr, len(b)),
    axis=ax_local_xc,
    label=label,
    color=color,
)

ax_local_xc.set_xlim(-55, 70)
ax_local_xc.set_ylim(-0.41, 0.81)
ax_local_xc.set_title("$\\varepsilon_{\\mathrm{IGW}} \\star \\text{local params}$", loc="left")
ax_local_xc.set_ylabel("Cross-correlation")
ax_local_xc.legend(fontsize=6)


# ---------------- Row 2: Remote parameters ----------------
(l_eps,) = ax_remote_ts.plot(
    dissip_at_C.index,
    dissip_at_C.to_numpy(),
    label="$\\varepsilon_\\text{IGW}$",
    lw=2,
    color="k",
)

ax_remote_ts.set_facecolor("None")
ax2 = ax_remote_ts.twinx()
ax2.set_facecolor("None")
ax2.set_zorder(-1)
ax2.set_ylabel("surface stress $\\tau$ / (N$\\,$m$^{-2}$)")
ax2.yaxis.label.set_color("#d95f02")
ax2.tick_params(axis="y", colors="#d95f02")
ax2.spines["right"].set_color("#d95f02")

(l_tau,) = ax2.plot(
    surface_stress_at_C.index,
    surface_stress_at_C.to_numpy(),
    color="#d95f02",
    # ls="--",
    label="$\\tau(t)$",
    lw=2,
)

ax3 = ax_remote_ts.twinx()
ax3.spines["right"].set_position(("axes", 1.15))
ax3.set_ylabel("inertial wind / (m/s)")
ax3.set_zorder(-2)
ax3.yaxis.label.set_color("#7fc97f")
ax3.tick_params(axis="y", colors="#7fc97f")
ax3.spines["right"].set_color("#7fc97f")

(l_uv,) = ax3.plot(
    inertial_wind_at_C.index,
    inertial_wind_at_C,
    color="#7fc97f",
    label=" inertial wind",
    lw=2,
    alpha=0.9,
)

ax4 = ax_remote_ts.twinx()
ax4.spines["right"].set_position(("axes", 1.33))
ax4.set_ylabel("$|\\overline{u}_\\mathrm{tidal}|$ / (m s$^{-1}$)")
ax4.yaxis.label.set_color("#d31414")
ax4.tick_params(axis="y", colors="#d31414")
ax4.spines["right"].set_color("#d31414")
ax4.set_zorder(-3)
ax3.set_facecolor("None")
(l_utidal,) = ax4.plot(
    [],
    [],
    color="#d31414",
    label="$|\\overline{u}_\\mathrm{tidal}|$",
    lw=2,
    alpha=0.6,
)

ax_remote_ts.set_title("Remote parameters", loc="left")
ax_remote_ts.set_xlabel("Time")
fig.supylabel("wave-induced dissipation rate\n$\\varepsilon_{\\mathrm{IW}}\\,$(W$\\,$kg$^{-1}$)")
ax_remote_ts.legend(
    # handles=[l_eps, l_avrg, l_paper, l_tau, l_uv],
    handles=[l_eps, l_tau, l_uv, l_utidal],
    ncol=4,  # 2,
    fontsize=6,
    title_fontsize=6,
    framealpha=0.9,
)


# Cross correlation
ax_remote_xc.axhline(0, color="k", alpha=0.3, ls=":")
ax_remote_xc.axvline(0, color="k", alpha=0.5, ls="--")

a = surface_stress_at_C.iloc[:]
b = dissip_at_C.loc[: a.index[-1]]
assert np.all(a.index == b.index)
lags, crosscorr = cc.get_normalized_crosscorr(a.to_numpy().flatten(), b.to_numpy().flatten())
time_lags = lags * b.index.diff().mean().days
label = "$\\tau \\star \\varepsilon_\\mathrm{IGW}$"
color = "#d95f02"
cc.plot_corr_with_error_bands(
    time_lags,
    crosscorr,
    x_err=5,
    y_err=cc.get_crosscorr_standard_error(crosscorr, len(b)),
    axis=ax_remote_xc,
    label=label,
    color=color,
)  # , plot_errorbars=False)

ax_remote_xc.plot([], [], label="inertial wind", color="#7fc97f")
a = inertial_wind_at_C.iloc[8:]
b = dissip_at_C.loc[: a.index[-1]]
assert np.all(a.index == b.index)
lags, crosscorr = cc.get_normalized_crosscorr(a.to_numpy().flatten(), b.to_numpy().flatten())
time_lags = lags * dissip_at_C.index.diff().mean().days
label = "$|u_f| \\star \\varepsilon_\\mathrm{IGW}$"
color = "#7fc97f"
cc.plot_corr_with_error_bands(
    time_lags,
    crosscorr,
    x_err=5,
    y_err=cc.get_crosscorr_standard_error(crosscorr, len(b)),
    axis=ax_remote_xc,
    label=label,
    color=color,
)


ax_remote_xc.plot([], [], label="tidal seasonality", color="tab:red", ls="--")

ax_remote_xc.set_xlabel("Lag")
ax_remote_xc.set_ylabel("Cross-correlation")
ax_remote_xc.legend(fontsize=6)
ax_remote_xc.set_title("$\\varepsilon_{\\mathrm{IGW}} \\star \\text{remote params}$", loc="left")

locator = mdates.AutoDateLocator(minticks=8, maxticks=12)
formatter = mdates.ConciseDateFormatter(locator)
ax_remote_ts.xaxis.set_major_locator(locator)
ax_remote_ts.xaxis.set_major_formatter(formatter)

fig.savefig(FIGURE_DIR / "f10_dissip+cross_corr.pdf")
plt.show()
