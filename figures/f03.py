import datetime as dt
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
from matplotlib.transforms import Bbox, TransformedBbox
from mpl_toolkits.axes_grid1.inset_locator import BboxConnector

import pinner2026_jpo_code
import pinner2026_jpo_code.plots

INTERIM_DATA_DIR: Path = pinner2026_jpo_code.DATA_DIR / "interim"
DERIVED_DATA_DIR: Path = pinner2026_jpo_code.DATA_DIR / "derived"
FIGURE_DIR: Path = pinner2026_jpo_code.FIGURE_DIR

ONE_COLUMN_WIDTH = pinner2026_jpo_code.plots.ONE_COLUMN_WIDTH
TWO_COLUMN_WIDTH = pinner2026_jpo_code.plots.TWO_COLUMN_WIDTH
GOLDEN_RATIO = pinner2026_jpo_code.plots.GOLDEN_RATIO

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
id_dict = {"A": 0, "B": 1, "C": 2, "control": 7, "D": 3, "E": 4, "F": 5, "G": 6}


# focus only on the seasonal variability
# smoothed_hke = hke.apply(smooth, args=(5,))

cmap = plt.cm.viridis
number_of_moorings = 7  # len(list_of_moorings)
# color_list = [cmap(i / number_of_moorings) for i in range(number_of_moorings)]
# color_list.append("xkcd:rose")
color_list = [
    "#7fc97f",
    "#beaed4",
    "#fdc086",
    "#ffff99",
    "#386cb0",
    "#f0027f",
    "#bf5b17",
    "#666666",
]
discrete_cmap = mcolors.LinearSegmentedColormap.from_list(
    "mooring_ids", color_list, N=number_of_moorings + 1
)

fig, ax = plt.subplots(
    ncols=2,
    sharex=True,
    figsize=(TWO_COLUMN_WIDTH, TWO_COLUMN_WIDTH / GOLDEN_RATIO),
    layout="constrained",
)
for axis in ax:
    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)


line_styles = ["solid", "dashed", "dotted"]
control_inestyle = ".-"

# reformat to dictionary of each mooring site
moorings = {}
for instrument_info in hke.columns:
    letter_id = instrument_info[0]
    moorings.setdefault(letter_id, []).append(instrument_info)

for letter_id, depths in moorings.items():
    # sort depths, with bottom instrument first
    # float() guards against numbers stored as strings
    depths_sorted = sorted(depths, key=lambda c: float(c[1]), reverse=True)
    color = color_list[id_dict[letter_id]]

    for rank, depth in enumerate(depths_sorted):
        ax[0].plot(
            hke.index,
            hke[depth],
            color=color,
            linestyle=line_styles[rank % len(line_styles)],
        )

    # leave out site A for the second subfigure
    if letter_id != "A":
        for rank, depth in enumerate(depths_sorted):
            ax[1].plot(
                hke.index,
                hke[depth],
                color=color,
                linestyle=line_styles[rank % len(line_styles)],
            )

norm = mcolors.Normalize(vmin=0, vmax=1)
cbar = fig.colorbar(
    mappable=mpl.cm.ScalarMappable(norm=norm, cmap=discrete_cmap),
    ax=ax,
    location="bottom",
    shrink=0.9,
    aspect=60,
)
cbar.ax.set_xticks(np.arange(0.5 / (number_of_moorings + 1), 1, 1 / (number_of_moorings + 1)))
# cbar.ax.invert_xaxis()
cbar.ax.set_xticklabels(list(names) + ["control"])

ax[0].axvline(dt.datetime(year=2018, month=1, day=1, tzinfo=dt.UTC), ls="--", color="k", alpha=0.5)
ax[0].axvline(dt.datetime(year=2019, month=1, day=1, tzinfo=dt.UTC), ls="--", color="k", alpha=0.5)
ax[1].axvline(dt.datetime(year=2018, month=1, day=1, tzinfo=dt.UTC), ls="--", color="k", alpha=0.5)
ax[1].axvline(dt.datetime(year=2019, month=1, day=1, tzinfo=dt.UTC), ls="--", color="k", alpha=0.5)
ax[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax[0].xaxis.get_major_locator()))
fig.supylabel(r"Kinetic wave field energy (m$^2\,$s$^{-2}$)")


def connect_y_range(ax_left, ax_right, **kwargs):
    """
    Draw two lines connecting the y-range shown in *ax_right* (its min and
    max y-limits) to the same y-range in *ax_left*, which sits to its left.
    Extra keyword arguments (color, lw, ls, ...) are passed to the lines.
        written by claude
    """
    ymin, ymax = sorted(ax_right.get_ylim())

    # Box spanning the full axes width (0..1 in axes coords) and ymin..ymax in data coords
    bbox = Bbox.from_extents(0, ymin, 1, ymax)
    bbox_left = TransformedBbox(bbox, ax_left.get_yaxis_transform())
    bbox_right = TransformedBbox(bbox, ax_right.get_yaxis_transform())

    # loc codes: 1 = upper right, 2 = upper left, 3 = lower left, 4 = lower right
    upper = BboxConnector(bbox_left, bbox_right, loc1=1, loc2=2, clip_on=False, **kwargs)
    lower = BboxConnector(bbox_left, bbox_right, loc1=4, loc2=3, clip_on=False, **kwargs)

    ax_right.add_patch(upper)
    ax_right.add_patch(lower)
    return upper, lower


ax[0].set_ylim(-0.1e-3, 4.3e-3)
ax[1].set_ylim(0.05e-3, 1.61e-3)
connect_y_range(ax[0], ax[1], color="k", lw=1)

fig.savefig(FIGURE_DIR / "f03_HKE_timeseries.pdf")
# plt.show()
