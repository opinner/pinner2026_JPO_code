import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import pandas as pd
import cmocean as cm
import pinner2026_jpo_code

INTERIM_DATA_DIR = pinner2026_jpo_code.DATA_DIR / "interim"
DERIVED_DATA_DIR = pinner2026_jpo_code.DATA_DIR / "derived"
FIGURE_DIR: Path = pinner2026_jpo_code.FIGURE_DIR

pd.read_csv(INTERIM_DATA_DIR / "adcp_shear.csv")

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)

# symmetric logarithmic norm
linthresh = 1e-3
norm = mcolors.SymLogNorm(linthresh)

# default linear norm
norm = mcolors.Normalize()

im0 = ax[0].pcolormesh(
    u_shear.index,
    u_shear.columns,
    u_shear.to_numpy().T,
    norm=norm,
    cmap=cm.cm.balance,
)

im1 = ax[1].pcolormesh(
    v_shear.index,
    v_shear.columns,
    v_shear.to_numpy().T,
    norm=norm,
    cmap=cm.cm.balance,
)

ubar = fig.colorbar(im0, ax=ax[0])
ubar.set_label("$\\frac{\\partial u}{\\partial z}$ / (s$^{-1}$)")
vbar = fig.colorbar(im1, ax=ax[1])
vbar.set_label("$\\frac{\\partial v}{\\partial z}$ / s$^{-1}$")

for axis in ax:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(formatter)
