---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

```{code-cell} ipython3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from pathlib import Path
import cmocean as cm
import pinner2026_jpo_code

RAW_DATA_DIR: Path = pinner2026_jpo_code.DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = pinner2026_jpo_code.DATA_DIR / "interim"
```

```{code-cell} ipython3
u = pd.read_csv(RAW_DATA_DIR / "moored_ADCP" / "AWI207_bottom_v.csv", skiprows=11, delimiter="\t")
u = u.drop(labels=["Unnamed: 5"], axis="columns")
v = pd.read_csv(RAW_DATA_DIR / "moored_ADCP" / "AWI207_bottom_u.csv", skiprows=11, delimiter="\t")
v = v.drop(labels=["Unnamed: 5"], axis="columns")
```

```{code-cell} ipython3
assert np.all(u.columns == v.columns)
assert u.shape == v.shape
assert u[["Ens", "HH", "MM", "SS"]].equals(v[["Ens", "HH", "MM", "SS"]])
```

```{code-cell} ipython3
v.info()
```

- File Size 14.082.048 bytes
- BB/WH Ensemble Length 954 bytes
- System Frequency: 153.6 kHz
- 1st Bin 12,23 m, Bin Size 8,00 m
- No. Bins 40, Pings/Ens 42, Time/Ping 01:25.71
- First Ensemble 00000001 22/04/20 15:09:47.16
- Last  Ensemble 00014761 23/12/26 15:09:47.16 Extra Data in File
- Average Ensemble Interval 00:10:20,69
- distance from ground: 10 m

```{code-cell} ipython3
DISTANCE_FROM_GROUND_METERS = 10.0
DISTANCE_FROM_ADCP = 12.23
VERTICAL_BIN_SIZE = 8.0
FIRST_DAY = pd.to_datetime("22/04/20", yearfirst=True)
FIRST_ENSEMBLE = pd.to_datetime("22/04/20 15:09:47.16", yearfirst=True)
LAST_ENSEMBLE = pd.to_datetime("23/12/26 15:09:47.16", yearfirst=True)
ENSEMBLE_COUNT = u.shape[0]
print(FIRST_ENSEMBLE)
print(LAST_ENSEMBLE)
```

```{code-cell} ipython3
# Get the start time from the first row
start_time = pd.Timedelta(hours=u["HH"].iloc[0], minutes=u["MM"].iloc[0], seconds=u["SS"].iloc[0])

# Compute each row's timedelta from midnight of the start date
row_times = pd.to_timedelta(u["HH"], unit="h") + \
            pd.to_timedelta(u["MM"], unit="m") + \
            pd.to_timedelta(u["SS"], unit="s") 

# Find where the time rolls over (i.e. row time < previous row time)
# and compute a day offset that increments at each rollover
day_offsets = (row_times < row_times.shift(1, fill_value=row_times.iloc[0])).cumsum()

# Combine: start_date + day_offset + row_time
timestamps = FIRST_DAY + pd.to_timedelta(day_offsets, unit="D") + row_times
assert LAST_ENSEMBLE - timestamps.iloc[-1] < pd.to_timedelta(1, unit="s")

u.index = timestamps
v.index = timestamps
u = u.drop(labels=["Ens", "HH", "MM", "SS", "HH.1"], axis="columns")
v = v.drop(labels=["Ens", "HH", "MM", "SS", "HH.1"], axis="columns")
```

```{code-cell} ipython3
u.columns = (DISTANCE_FROM_GROUND_METERS + DISTANCE_FROM_ADCP + u.columns.astype(int) * VERTICAL_BIN_SIZE).round(2)
v.columns = (DISTANCE_FROM_GROUND_METERS + DISTANCE_FROM_ADCP + v.columns.astype(int) * VERTICAL_BIN_SIZE).round(2)
```

```{code-cell} ipython3
# conversion from mm/s to m/s
u = u/100
v = v/100
```

```{code-cell} ipython3
fig, ax = plt.subplots(ncols=2, sharex = True, sharey=True)
binwidth = 0.5
ubins=np.arange(u.min(axis=None), u.max(axis=None) + binwidth, binwidth)
vbins=np.arange(u.min(axis=None), u.max(axis=None) + binwidth, binwidth)
ax[0].hist(u.to_numpy().flatten(), bins=ubins, log=True, color="0.7")
ax[1].hist(v.to_numpy().flatten(), bins=vbins, log=True, color="0.7")

ut = u.copy()
vt = v.copy()

MAX_REASONABLE_VELOCITY = 4
# remove everything faster than MAX_REASONABLE_VELOCITY
ut[u.abs() > MAX_REASONABLE_VELOCITY] = np.nan
vt[v.abs() > MAX_REASONABLE_VELOCITY] = np.nan

ax[0].hist(ut.to_numpy().flatten(), bins=ubins, log=True, color="k")
ax[1].hist(vt.to_numpy().flatten(), bins=vbins, log=True, color="k")

ax[0].set_title("u")
ax[1].set_title("v")

ax[0].set_ylabel("count")

ax[0].set_xlabel("u / (m/s)")
ax[1].set_xlabel("v / (m/s)")

# for axis in ax:
#     axis.axvline(MAX_REASONABLE_VELOCITY, color="tab:red", ls = "--")
#     axis.axvline(-MAX_REASONABLE_VELOCITY, color="tab:red", ls = "--")
```

```{code-cell} ipython3
ut.to_csv(INTERIM_DATA_DIR / "AWI207_bottom_u.csv")
vt.to_csv(INTERIM_DATA_DIR / "AWI207_bottom_v.csv")
```

```{code-cell} ipython3
uclean = pd.read_csv(INTERIM_DATA_DIR / "AWI207_bottom_u.csv", parse_dates=True, index_col=0)
uclean.columns = uclean.columns.astype(float)
vclean = pd.read_csv(INTERIM_DATA_DIR / "AWI207_bottom_v.csv", parse_dates=True, index_col=0)
vclean.columns = vclean.columns.astype(float)
```

```{code-cell} ipython3
fig, ax = plt.subplots(nrows=2, sharex = True, sharey=True)

max_plot_velocity= MAX_REASONABLE_VELOCITY
#max_plot_velocity= 2

im0 = ax[0].pcolormesh(
    uclean.index, uclean.columns, uclean.to_numpy().T,
    norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-max_plot_velocity, vmax=max_plot_velocity),
    cmap=cm.cm.balance
)

im1 = ax[1].pcolormesh(
    vclean.index,
    vclean.columns,
    vclean.to_numpy().T,
    norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-max_plot_velocity, vmax=max_plot_velocity),
    cmap=cm.cm.balance
)

fig.colorbar(im0, ax=ax[0])
fig.colorbar(im1, ax=ax[1])

for axis in ax:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(formatter)
```

```{code-cell} ipython3
#u.describe()
```

```{code-cell} ipython3
u_shear = pd.DataFrame(
    np.gradient(uclean.values, uclean.columns, axis=1),
    index=uclean.index,
    columns=uclean.columns
)

v_shear = pd.DataFrame(
    np.gradient(vclean.values, vclean.columns, axis=1),
    index=vclean.index,
    columns=vclean.columns
)

Su = u_shear**2
Sv = v_shear**2

u_shear_control = uclean.diff(axis=1) / VERTICAL_BIN_SIZE
v_shear_control = vclean.diff(axis=1) / VERTICAL_BIN_SIZE
Su_control = u_shear_control**2
Sv_control = v_shear_control**2
```

```{code-cell} ipython3
#plt.hist((Sv- Sv_control).to_numpy().flatten(), log=True, bins=20)
```

```{code-cell} ipython3
fig, ax = plt.subplots(ncols=2, sharex = True, sharey=True)
binwidth = (Su.max(axis=None) - Su.min(axis=None)) / 30
ubins=np.arange(Su.min(axis=None), Su.max(axis=None) + binwidth, binwidth)
vbins=np.arange(Sv.min(axis=None), Sv.max(axis=None) + binwidth, binwidth)

ax[0].hist(Su_control.to_numpy().flatten(), bins=ubins, log=True, color="tab:blue", alpha=0.3, label="np.diff")
ax[1].hist(Sv_control.to_numpy().flatten(), bins=vbins, log=True, color="tab:blue", alpha=0.3, label="np.diff")

ax[0].hist(Su.to_numpy().flatten(), bins=ubins, log=True, color="0.7", label="np.gradient")
ax[1].hist(Sv.to_numpy().flatten(), bins=vbins, log=True, color="0.7", label="np.gradient")

ax[0].set_title("u")
ax[1].set_title("v")
ax[0].legend()
ax[1].legend()
ax[0].set_ylabel("count")

ax[0].set_xlabel("$S_u^2$ / (s$^{-2}$)")
ax[1].set_xlabel("$S_v^2$ / (s$^{-2}$)")
```

```{code-cell} ipython3
fig, ax = plt.subplots(nrows=2, sharex = True, sharey=True)

# max_plot_shear =  2* MAX_REASONABLE_VELOCITY / VERTICAL_BIN_SIZE
max_plot_shear_squared =  1*1e-3

norm = mcolors.LogNorm(vmin= 1e-6, vmax=1e-2)
norm = mcolors.PowerNorm(gamma=1, vmin=0, vmax=1e-3)

im0 = ax[0].pcolormesh(
    Su.index, Su.columns, Su.to_numpy().T,
    norm=norm,
    cmap=cm.cm.speed
)


im1 = ax[1].pcolormesh(
    Sv.index,
    Sv.columns,
    Sv.to_numpy().T,
    norm=norm,
    cmap=cm.cm.speed
)

ubar = fig.colorbar(im0, ax=ax[0])
ubar.set_label("$S_u^2$ / (s$^{-2}$)")
vbar = fig.colorbar(im1, ax=ax[1])
vbar.set_label("$S_v^2$ / s$^{-2}$")

for axis in ax:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(formatter)
```

```{code-cell} ipython3
fig, ax = plt.subplots(nrows=2, sharex = True, sharey=True)

linthresh = 1e-3

im0 = ax[0].pcolormesh(
    u_shear.index, u_shear.columns, u_shear.to_numpy().T,
    norm=mcolors.SymLogNorm(linthresh),
    cmap=cm.cm.balance
)

im1 = ax[1].pcolormesh(
    v_shear.index,
    v_shear.columns,
    v_shear.to_numpy().T,
    norm=mcolors.SymLogNorm(linthresh),
    cmap=cm.cm.balance
)
    
fig.colorbar(im0, ax=ax[0])
fig.colorbar(im1, ax=ax[1])

for axis in ax:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(formatter)
```

```{code-cell} ipython3
fig, ax = plt.subplots(nrows=2, sharex = True, sharey=True)

# max_plot_shear = 2 * MAX_REASONABLE_VELOCITY / VERTICAL_BIN_SIZE
# max_plot_shear = max_plot_shear * 1e-3
# #max_plot_shear= 2

im0 = ax[0].pcolormesh(
    u_shear.index, u_shear.columns, u_shear.to_numpy().T,
    norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-max_plot_shear, vmax=max_plot_shear),
    cmap=cm.cm.balance
)

im1 = ax[1].pcolormesh(
    v_shear.index,
    v_shear.columns,
    v_shear.to_numpy().T,
    norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-max_plot_shear, vmax=max_plot_shear),
    cmap=cm.cm.balance
)
    
fig.colorbar(im0, ax=ax[0])
fig.colorbar(im1, ax=ax[1])

for axis in ax:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(formatter)
```
