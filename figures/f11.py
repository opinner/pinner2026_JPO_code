import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import pandas as pd
import cmocean as cm
import pinner2026_jpo_code

INTERIM_DATA_DIR = pinner2026_jpo_code.DATA_DIR / "interim"
DERIVED_DATA_DIR = pinner2026_jpo_code.DATA_DIR / "derived"
FIGURE_DIR: Path = pinner2026_jpo_code.FIGURE_DIR

u_shear = pd.read_csv(INTERIM_DATA_DIR / "AWI207_bottom_u_shear.csv", parse_dates=True, index_col=0)
u_shear.columns = u_shear.columns.astype(float)
v_shear = pd.read_csv(INTERIM_DATA_DIR / "AWI207_bottom_v_shear.csv", parse_dates=True, index_col=0)
v_shear.columns = v_shear.columns.astype(float)

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, layout="constrained")

# symmetric logarithmic norm
linthresh = 1e-2
norm = mcolors.SymLogNorm(linthresh)

# default linear norm
# norm = mcolors.CenteredNorm(
#     vcenter=0,
#     halfrange=5e-2,
# )

im0 = ax[0].pcolormesh(
    u_shear.index,
    u_shear.columns,
    u_shear.to_numpy().T,
    norm=norm,
    cmap=cm.cm.balance,
    rasterized=True,
)

im1 = ax[1].pcolormesh(
    v_shear.index,
    v_shear.columns,
    v_shear.to_numpy().T,
    norm=norm,
    cmap=cm.cm.balance,
    rasterized=True,
)

ubar = fig.colorbar(im0, ax=ax[0])
ubar.set_label("$\\frac{\\partial u}{\\partial z}$ / (s$^{-1}$)")
vbar = fig.colorbar(im1, ax=ax[1])
vbar.set_label("$\\frac{\\partial v}{\\partial z}$ / s$^{-1}$")

ax[0].set_ylim(30, 200)

for axis in ax:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(formatter)

fig.savefig(FIGURE_DIR / "f11_adcp_shear.pdf", dpi=300)
plt.show()
