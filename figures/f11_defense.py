import datetime
import pickle
from pathlib import Path

import cmocean as cm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.units as munits

# import pandas as pd  # not called, but pandas Dataframe are pickled
import numpy as np

import pinner2026_jpo_code
import pinner2026_jpo_code.plots

ONE_COLUMN_WIDTH = pinner2026_jpo_code.plots.ONE_COLUMN_WIDTH
TWO_COLUMN_WIDTH = pinner2026_jpo_code.plots.TWO_COLUMN_WIDTH


# converter = mdates.ConciseDateConverter()
# munits.registry[np.datetime64] = converter
# munits.registry[datetime.date] = converter
# munits.registry[datetime.datetime] = converter


INTERIM_DATA_DIR: Path = pinner2026_jpo_code.DATA_DIR / "interim"
FIGURE_DIR: Path = pinner2026_jpo_code.FIGURE_DIR

with open(INTERIM_DATA_DIR / "NIW_filtered.pickle", "rb") as NIW_file:
    NIW_dict = pickle.load(NIW_file)

with open(INTERIM_DATA_DIR / "IT_filtered.pickle", "rb") as IT_file:
    IT_dict = pickle.load(IT_file)


fig, ax = plt.subplots(ncols=4, sharey=True, sharex=True, figsize=(7, 2), layout="constrained")

HALFRANGE = 0.05

# NIW plotting
for i, (name, data) in enumerate(NIW_dict.items()):
    if name == "raw":
        continue
    if "CCW" not in name:
        continue

    data = data.apply(np.imag).abs().rolling("1D").mean()

    if "dn" in name:
        ax[0 + i].set_title(f"downwards")
    else:
        ax[0 + i].set_title(f"upwards")
    im = ax[0 + i].pcolormesh(
        data.index,
        data.columns,
        data.to_numpy().T,  # v (east) velocity
        cmap=cm.cm.speed,
        vmax=0.08,
        rasterized=True,
    )


for i, (name, data) in enumerate(IT_dict.items()):
    if name == "raw":
        continue
    if "CCW" not in name:
        continue

    data = data.apply(np.imag).abs().rolling("1D").mean()

    if "dn" in name:
        ax[2 + i].set_title(f"downwards")
    else:
        ax[2 + i].set_title(f"upwards")
    im = ax[2 + i].pcolormesh(
        data.index,
        data.columns,
        data.to_numpy().T,  # v (east) velocity
        cmap=cm.cm.speed,
        vmax=0.08,
        rasterized=True,
    )

fig.supylabel("Meters above ground")

for axis in ax.flatten():
    # axis.set_ylim(30, 150)
    # axis.set_xlim(mdates.datestr2num("2023-02"),mdates.datestr2num("2023-03"))
    axis.set_xlim(mdates.datestr2num("2022-11-15"), mdates.datestr2num("2022-12-24"))
    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(formatter)

cbar = fig.colorbar(im, ax=ax, extend="max")
cbar.set_label("mean v / (m/s)")

# ax0 = ax[0].twinx()
# ax0.plot(
#     inertial_wind_at_C.loc[CV.index[0] : CV.index[-1]].index,
#     inertial_wind_at_C.loc[CV.index[0] : CV.index[-1]].abs().rolling(window="1D").mean(),
#     alpha=0.6,
#     color="tab:orange",
# )
# ax1 = ax[1].twinx()
# ax1.plot(
#     inertial_wind_at_C.loc[CV.index[0] : CV.index[-1]].index,
#     inertial_wind_at_C.loc[CV.index[0] : CV.index[-1]].abs().rolling(window="1D").mean(),
#     alpha=0.6,
#     color="tab:orange",
# )


fig.savefig(FIGURE_DIR / "f11_defense.svg", dpi=300)
plt.show()
