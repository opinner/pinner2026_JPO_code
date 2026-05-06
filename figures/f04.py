import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cmocean
import pandas as pd
import scipy.signal as signal
from pathlib import Path
import pinner2026_jpo_code

INTERIM_DATA_DIR: Path = pinner2026_jpo_code.DATA_DIR / "interim"
DERIVED_DATA_DIR: Path = pinner2026_jpo_code.DATA_DIR / "derived"
FIGURE_DIR: Path = pinner2026_jpo_code.FIGURE_DIR

cm = 1 / 2.54  # centimeters in inches
TWO_COLUMN_WIDTH = 12

plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white", "font.size": 9})


def smooth(array, length):
    window = signal.windows.hann(length)  # construct Hanning window
    # smooth by convolving speed with window
    smoothed = signal.convolve(array, window, mode="same") / np.sum(window)
    return smoothed


hke = pd.read_csv(INTERIM_DATA_DIR / "HKE.csv", index_col=0, header=[0, 1])
hke.index = pd.to_datetime(hke.index).floor("D")

# to focus only on the seasonal variability
# we smooth over 5 data points
smoothed_hke = hke.apply(smooth, args=(5,))


corr = smoothed_hke.corr(method="spearman")
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")

# Generate a custom diverging colormap
cmap = cmocean.cm.amp
clipped_cmap = cmocean.tools.crop_by_percent(cmap, 20, which="min", N=None)
clipped_cmap.set_extremes(under="lightgray")
clipped_correlation = 0.4

# Draw the heatmap with the mask and correct aspect ratio
matrix = sns.heatmap(
    corr,
    mask=mask,
    cmap=clipped_cmap,
    vmax=1,
    vmin=clipped_correlation,
    # center=0,
    linewidths=0.5,
    cbar=False,
    cbar_kws={"shrink": 0.4},  # , set_frame_on:True},
    annot=True,
    annot_kws={"fontsize": 9},
    fmt=".1f",
    xticklabels=corr.index.get_level_values(1),
    yticklabels=corr.index.get_level_values(1),
)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

cax = ax.inset_axes([0.7, 0.4, 0.05, 0.5])
norm = mpl.colors.Normalize(vmin=clipped_correlation, vmax=1)
fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=clipped_cmap),
    extend="min",
    cax=cax,
    orientation="vertical",
    label="Spearman correlation",
    shrink=0.1,
)

ax.set(xlabel="", ylabel="")
# label the classes:
sec = ax.secondary_yaxis(location=0)
labels = [f"{s:{' '}<16}" for s in pd.unique(corr.index.get_level_values(0))]
sec.set_yticks([1.5, 4.5, 7, 8.5, 10.5, 13.5, 15.5, 17], labels=labels)
sec.tick_params("y", length=0)
# lines between the classes:
sec2 = ax.secondary_yaxis(location=0)
sec2.set_yticks([3, 6, 8, 9, 12, 15, 16], labels=[])
sec2.tick_params("y", length=40, width=1)

sec = ax.secondary_xaxis(location=0)
newline = "\n"
labels = [f"{4 * newline}{mooring}" for mooring in pd.unique(corr.index.get_level_values(0))]
# for i,label in enumerate(labels.copy()):
#     if "control" in label:
#         labels[i] = "\n"+label
sec.set_xticks([1.5, 4.5, 7, 8.5, 10.5, 13.5, 15.5, 17], labels=labels)
sec.tick_params("x", length=0)
# lines between the classes:
sec2 = ax.secondary_xaxis(location=0)
sec2.set_xticks([3, 6, 8, 9, 12, 15, 16], labels=[])
sec2.tick_params("x", length=40, width=1)
ax.tick_params(axis="x", labelrotation=90)
# ax.set_title("Pairwise Correlation")
ax.text(-0.08, -0.08, "depths (m)", transform=ax.transAxes, ha="center", va="center", rotation=45)
fig.savefig(FIGURE_DIR / "f04_HKE_pair_correlation.pdf")
plt.show()
