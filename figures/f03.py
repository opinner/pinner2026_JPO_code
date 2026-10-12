from datetime import datetime
from pathlib import Path

import cmocean
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal

import pinner2026_jpo_code
import pinner2026_jpo_code.plots

INTERIM_DATA_DIR: Path = pinner2026_jpo_code.DATA_DIR / "interim"
DERIVED_DATA_DIR: Path = pinner2026_jpo_code.DATA_DIR / "derived"
FIGURE_DIR: Path = pinner2026_jpo_code.FIGURE_DIR

ONE_COLUMN_WIDTH = pinner2026_jpo_code.plots.ONE_COLUMN_WIDTH

plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white", "font.size": 9})


def smooth(array, length):
    window = signal.windows.hann(length)  # construct Hanning window
    smoothed = signal.convolve(array, window, mode="same") / np.sum(
        window
    )  # smooth by convolving speed with window
    return smoothed


hke = pd.read_csv(INTERIM_DATA_DIR / "HKE.csv", index_col=0, header=[0, 1])
hke.index = pd.to_datetime(hke.index).floor("D")

names = ["A", "B", "C", "D", "E", "F", "G"]  # hke.columns.get_level_values(0)
id_dict = {"A": 0, "B": 1, "C": 2, "control": 2, "D": 3, "E": 4, "F": 5, "G": 6}


# focus only on the seasonal variability
# smoothed_hke = hke.apply(smooth, args=(5,))

cmap = plt.cm.viridis
number_of_moorings = 7  # len(list_of_moorings)
color_list = [cmap(i / number_of_moorings) for i in range(number_of_moorings)]
color_list.append("xkcd:rose")
discrete_cmap = mcolors.LinearSegmentedColormap.from_list(
    "mooring_ids", color_list, N=number_of_moorings + 1
)

fig, ax = plt.subplots(
    nrows=2, sharex=True, figsize=(2 * ONE_COLUMN_WIDTH, 2 * ONE_COLUMN_WIDTH), layout="constrained"
)
ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)

# color = color_list[mooring_id]

line_styles = ["solid", "dashed", "dotted"]
control_inestyle = ".-"

groups = {}
for col in hke.columns:
    groups.setdefault(col[0], []).append(col)

for letter, cols in groups.items():
    # Largest number first; float() guards against numbers stored as strings
    cols_sorted = sorted(cols, key=lambda c: float(c[1]), reverse=True)
    color = color_list[id_dict[letter]]

    for rank, col in enumerate(cols_sorted):
        ax[0].plot(
            hke.index,
            hke[col],
            color=color,
            linestyle=line_styles[rank % len(line_styles)],
            # label=f"{col[0]}, {col[1]}",
            # **plot_kwargs,
        )

    if letter != "A":
        for rank, col in enumerate(cols_sorted):
            ax[1].plot(
                hke.index,
                hke[col],
                color=color,
                linestyle=line_styles[rank % len(line_styles)],
                # label=f"{col[0]}, {col[1]}",
                # **plot_kwargs,
            )

norm = mcolors.Normalize(vmin=0, vmax=1)
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=discrete_cmap), ax=ax)
cbar.ax.set_yticks(np.arange(0.5 / (number_of_moorings + 1), 1, 1 / (number_of_moorings + 1)))
cbar.ax.invert_yaxis()
cbar.ax.set_yticklabels(list(names) + ["control"])
# ax.axvline(datetime(year = 2017, month = 1, day = 1), ls = "--", color = "k", alpha = 0.5)
ax[0].axvline(datetime(year=2018, month=1, day=1), ls="--", color="k", alpha=0.5)
ax[0].axvline(datetime(year=2019, month=1, day=1), ls="--", color="k", alpha=0.5)
ax[1].axvline(datetime(year=2018, month=1, day=1), ls="--", color="k", alpha=0.5)
ax[1].axvline(datetime(year=2019, month=1, day=1), ls="--", color="k", alpha=0.5)
ax[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax[0].xaxis.get_major_locator()))
fig.supylabel(r"Kinetic wave field energy (m$^2\,$s$^{-2}$)")

fig.savefig(FIGURE_DIR / "f03_HKE_timeseries.pdf")
plt.show()
