# ----------------------------------------------------------------------
# Converted from Matlab with claude.ai
# ----------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib import cm as mcm
from matplotlib.colors import ListedColormap

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
data = sio.loadmat("historical_obs_WeddellSea.mat", squeeze_me=True, struct_as_record=False)
bodc_data = data["BODC_all"]
adelie_data = data["ADELIE"]
pangaea_data = data["f_in_all"]
ctd_data = data["CTD"]

# ----------------------------------------------------------------------
# Build year groupings for colouring
#
# Each dataset was collected during a different cruise/year. We want all
# datasets to share a single colour scale based on year, so we collect
# every year across all datasets, find the unique set, and use the index
# into that unique set as a "group" label for colouring. The CTD/PICCOLO
# dataset doesn't have an associated year field, so it's manually assigned
# to "2024" to give it its own group/colour.
# ----------------------------------------------------------------------
pangaea_years = np.array(
    [dt.year if hasattr(dt, "year") else int(dt) for dt in np.atleast_1d(pangaea_data.datetime)]
)
bodc_years = np.atleast_1d(bodc_data.year).astype(int)
adelie_year = np.atleast_1d(adelie_data.year)[0]

all_years = np.concatenate([pangaea_years, bodc_years, [adelie_year], [2024]])
unique_years, year_group_index = np.unique(all_years, return_inverse=True)

n_pangaea = len(pangaea_years)
n_bodc = len(bodc_years)

# Slice the combined group index back out per-dataset, in the same order
# they were concatenated above.
pangaea_group = year_group_index[:n_pangaea]
bodc_group = year_group_index[n_pangaea : n_pangaea + n_bodc]
adelie_group = year_group_index[-2]
ctd_group = year_group_index[-1]  # the "2024" placeholder group for CTD/PICCOLO

# ----------------------------------------------------------------------
# Colormap: RdPu for cruise years, plus a distinct blue for CTD/PICCOLO
#
# We drop the first (lightest, near-white) colour of RdPu because it's
# too low-contrast against the white figure background, then append a
# blue swatch so the most recent (PICCOLO) data stands out clearly from
# the older cruise years.
# ----------------------------------------------------------------------
n_groups = len(unique_years)
rdpu_colors = mcm.get_cmap("RdPu", n_groups)(np.linspace(0, 1, n_groups))
blue_color = mcm.get_cmap("Blues", 2)(np.linspace(0, 1, 2))[-1]

colors = np.vstack([rdpu_colors[1:], blue_color])
year_cmap = ListedColormap(colors)

# Colour limits are offset by 0.5 so that each discrete group index lands
# in the centre of its colour swatch rather than on a boundary.
color_min, color_max = -0.5, n_groups - 0.5

# ----------------------------------------------------------------------
# Temperature-Salinity diagram
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(18 / 2.54, 15 / 2.54))

scatter_pangaea = ax.scatter(
    pangaea_data.SA,
    pangaea_data.CT,
    s=6,
    c=pangaea_group,
    cmap=year_cmap,
    vmin=color_min,
    vmax=color_max,
    rasterized=True,
)

# ADELIE data may contain NaNs in SA; filter these out so they don't
# produce empty/invalid points in the scatter plot.
adelie_valid = ~np.isnan(np.atleast_1d(adelie_data.SA))
scatter_adelie = ax.scatter(
    np.atleast_1d(adelie_data.SA)[adelie_valid],
    np.atleast_1d(adelie_data.CT)[adelie_valid],
    s=6,
    c=np.full(adelie_valid.sum(), adelie_group),
    cmap=year_cmap,
    vmin=color_min,
    vmax=color_max,
    rasterized=True,
)

# Same NaN-filtering rationale as for ADELIE above.
bodc_valid = ~np.isnan(np.atleast_1d(bodc_data.SA))
scatter_bodc = ax.scatter(
    np.atleast_1d(bodc_data.SA)[bodc_valid],
    np.atleast_1d(bodc_data.CT)[bodc_valid],
    s=6,
    c=bodc_group[bodc_valid],
    cmap=year_cmap,
    vmin=color_min,
    vmax=color_max,
    rasterized=True,
)

# CTD data is filtered by valid depth (Z) rather than SA/CT directly,
# matching the original selection criterion used for this dataset.
ctd_valid = ~np.isnan(ctd_data.Z)
scatter_ctd = ax.scatter(
    ctd_data.SA[ctd_valid],
    ctd_data.CT[ctd_valid],
    s=6,
    c=np.full(ctd_valid.sum(), ctd_group),
    cmap=year_cmap,
    vmin=color_min,
    vmax=color_max,
    rasterized=True,
)

ax.legend(
    [scatter_pangaea, scatter_adelie, scatter_ctd],
    ["CTD PANGAEA", "CTD ADELIE", "CTD PICCOLO"],
    loc="upper left",
)

# Build a standalone ScalarMappable for the colorbar since the four
# scatter calls each only cover part of the colour range individually.
sm = plt.cm.ScalarMappable(cmap=year_cmap, norm=plt.Normalize(vmin=color_min, vmax=color_max))
colorbar = fig.colorbar(sm, ax=ax)
colorbar.set_label("year")
colorbar.set_ticks(np.arange(n_groups))
colorbar.set_ticklabels(unique_years)

ax.set_xlabel("Absolute Salinity (g kg$^{-1}$)")
ax.set_ylabel("Conservative Temperature (°C)")
ax.set_xlim(33.4, 35)
ax.set_ylim(-2, 1)
ax.grid(True)
ax.set_title("Freshening in the Weddell Sea")

fig.tight_layout()
fig.savefig("freshening_Weddell_TS.pdf")

# Zoomed-in view focusing on the higher-salinity range, saved separately.
ax.set_xlim(34.4, 35)
fig.savefig("freshening_Weddell_TS_zoom.pdf")
