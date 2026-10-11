from pathlib import Path

import cmocean
import geopy.distance
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import seaborn as sns

import pinner2026_jpo_code
import pinner2026_jpo_code.plots

INTERIM_DATA_DIR: Path = pinner2026_jpo_code.DATA_DIR / "interim"
DERIVED_DATA_DIR: Path = pinner2026_jpo_code.DATA_DIR / "derived"
FIGURE_DIR: Path = pinner2026_jpo_code.FIGURE_DIR

ONE_COLUMN_WIDTH = pinner2026_jpo_code.plots.ONE_COLUMN_WIDTH
TWO_THIRDS_PAGE_WIDTH = 4.5

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 9,
        # "pdf.fonttype": 42
    }
)


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
fig, ax = plt.subplots(
    nrows=2,
    height_ratios=(3, 2),
    figsize=(TWO_THIRDS_PAGE_WIDTH, 2 * ONE_COLUMN_WIDTH),
    gridspec_kw={"hspace": 0.1},
    layout="constrained",
)

# --------------------- ax 0 --------------------------------------

# Generate a custom diverging colormap
cmap = cmocean.cm.amp
clipped_cmap = cmocean.tools.crop_by_percent(cmap, 20, which="min", N=None)
clipped_cmap.set_extremes(under="lightgray")
clipped_correlation = 0.4

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(
    data=corr,
    mask=mask,
    cmap=clipped_cmap,
    vmax=1,
    vmin=clipped_correlation,
    # center=0,
    linewidths=0.5,
    cbar=False,
    cbar_kws={"shrink": 0.4},  # , set_frame_on:True},
    annot=True,
    annot_kws={"fontsize": 7},
    fmt=".1f",
    ax=ax[0],
    xticklabels=corr.index.get_level_values(1),  # type: ignore
    yticklabels=corr.index.get_level_values(1),  # type: ignore
)

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

cax = ax[0].inset_axes([0.7, 0.4, 0.05, 0.5])
norm = mpl.colors.Normalize(vmin=clipped_correlation, vmax=1)  # type: ignore
fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=clipped_cmap),  # type: ignore
    extend="min",
    cax=cax,
    orientation="vertical",
    label="Spearman correlation",
    shrink=0.1,
)

ax[0].set(xlabel="", ylabel="")
# label the classes:
sec = ax[0].secondary_yaxis(location=0)
labels = [f"{s:{' '}<16}" for s in pd.unique(corr.index.get_level_values(0))]
sec.set_yticks([1.5, 4.5, 7, 8.5, 10.5, 13.5, 15.5, 17], labels=labels)
sec.tick_params("y", length=0)
# lines between the classes:
sec2 = ax[0].secondary_yaxis(location=0)
sec2.set_yticks([3, 6, 8, 9, 12, 15, 16], labels=[])
sec2.tick_params("y", length=40, width=1)

sec = ax[0].secondary_xaxis(location=0)
newline = "\n"
labels = [f"{4 * newline}{mooring}" for mooring in pd.unique(corr.index.get_level_values(0))]
# for i,label in enumerate(labels.copy()):
#     if "control" in label:
#         labels[i] = "\n"+label
sec.set_xticks([1.5, 4.5, 7, 8.5, 10.5, 13.5, 15.5, 17], labels=labels)
sec.tick_params("x", length=0)
# lines between the classes:
sec2 = ax[0].secondary_xaxis(location=0)
sec2.set_xticks([3, 6, 8, 9, 12, 15, 16], labels=[])
sec2.tick_params("x", length=40, width=1)
ax[0].tick_params(axis="x", labelrotation=90)
# ax[0].set_title("Pairwise Correlation")
ax[0].text(
    -0.08, -0.08, "depths / m", transform=ax[0].transAxes, ha="center", va="center", rotation=45
)

# --------------------- ax 1 --------------------------------------

masked_corr_df = corr.where(mask)  # mask to triagonal matrix
masked_corr_df[np.eye(len(masked_corr_df), dtype=bool) == 1] = np.nan  # mask diagonal

mooring_info = pd.read_csv(INTERIM_DATA_DIR / "rcm_infos.csv", index_col=0)

# add control data point
line = pd.DataFrame(mooring_info.iloc[[6]].to_dict(orient="list"), index=[6])
line["region"] = "control"
line["color"] = "white"
line["rounded depth"] = 750
line["rounded mab"] = np.nan
mooring_info = pd.concat([mooring_info.iloc[:6], line, mooring_info.iloc[6:]]).reset_index(
    drop=True
)

coords = list(zip(mooring_info["lat"], mooring_info["lon"], strict=True))
n = len(coords)

dist = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        d = geopy.distance.great_circle(coords[i], coords[j]).km
        dist[i, j] = dist[j, i] = d

dist_df = pd.DataFrame(dist, index=mooring_info["name"], columns=mooring_info["name"])

assert dist_df.shape == corr.shape
dist_list = dist_df.to_numpy().flatten()
corr_list = masked_corr_df.to_numpy().flatten()
assert np.shape(dist_list) == np.shape(corr_list)

colors = np.where(corr_list <= 0.4, "lightgray", "tab:blue")
ax[1].scatter(
    dist_list,
    corr_list,
    c=corr_list,
    norm=norm,
    cmap=clipped_cmap,
    s=20,
)

ax[1].set_xlabel("distance between measurements / km")
ax[1].set_ylabel("Spearman correlation")

# --------------------- saving --------------------------------------

fig.savefig(FIGURE_DIR / "f04_HKE_pair_correlation_preformat.svg")
# plt.show()
