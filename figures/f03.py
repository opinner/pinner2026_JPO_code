from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cmocean
import pandas as pd
import scipy.signal as signal
import pinner2026_jpo_code

INTERIM_DATA_DIR: Path = pinner2026_jpo_code.DATA_DIR / "interim"
DERIVED_DATA_DIR: Path = pinner2026_jpo_code.DATA_DIR / "derived"

cm = 1 / 2.54  # centimeters in inches
TWO_COLUMN_WIDTH = 12

plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white", "font.size": 9})


def smooth(array, length):
    window = signal.windows.hann(length)  # construct Hanning window
    smoothed = signal.convolve(array, window, mode="same") / np.sum(
        window
    )  # smooth by convolving speed with window
    return smoothed


hke = pd.read_csv(INTERIM_DATA_DIR / "HKE.csv", index_col=0, header=[0, 1])
hke.index = pd.to_datetime(hke.index).floor("D")

hke.plot()
plt.show()

# focus only on the seasonal variability
# smoothed_hke = hke.apply(smooth, args=(5,))

cmap = plt.cm.viridis
number_of_moorings = 7  # len(list_of_moorings)
color_list = [cmap(i / number_of_moorings) for i in range(number_of_moorings)]
color_list.append("xkcd:rose")
discrete_cmap = mcolors.LinearSegmentedColormap.from_list(
    "mooring_ids", color_list, N=number_of_moorings + 1
)

fig, ax = plt.subplots(1, figsize=(6, 3), layout="constrained")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)

color = color_list[mooring_id]

for row in hke.iterrows():
    pass

# ax.plot(wavefield.index, wavefield.values, color=color, lw=2.5, zorder=i)

norm = mcolors.Normalize(vmin=0, vmax=1)
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=discrete_cmap), ax=ax)
cbar.ax.set_yticks(np.arange(0.5 / (number_of_moorings + 1), 1, 1 / (number_of_moorings + 1)))
cbar.ax.invert_yaxis()
cbar.ax.set_yticklabels(names + ["control"])
# ax.axvline(datetime(year = 2017, month = 1, day = 1), ls = "--", color = "k", alpha = 0.5)
ax.axvline(datetime(year=2018, month=1, day=1), ls="--", color="k", alpha=0.5)
ax.axvline(datetime(year=2019, month=1, day=1), ls="--", color="k", alpha=0.5)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.set_ylabel(r"Kinetic wave field energy (m$^2\,$s$^{-2}$)")
fig.savefig(FIGURE_DIR / "HKE_timeseries.pdf")
plt.show()
