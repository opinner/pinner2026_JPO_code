import matplotlib as plt
import matplotlib.ticker as mticker
import numpy as np

from pinner2026_jpo_code.plots import plot_xy_errorbands

cm = 1 / 2.54  # centimeters in inches
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white", "font.size": 9})

fig, ax = plt.subplots(ncols=1, figsize=(9 * cm, 6 * cm), layout="constrained")
ax = ["", ax]
ax[1].axhline(0, color="k", alpha=0.3, ls=":")
ax[1].axvline(0, color="k", alpha=0.5, ls="--")

ax[1].xaxis.set_major_locator(mticker.FixedLocator(np.arange(-100, 150, 50)))
ax[1].xaxis.set_minor_locator(mticker.FixedLocator(np.arange(-100, 110, 25)))


def get_pearson_standard_error(x, y, N):
    # standard error of pearson correlation according to Bonett2008, recommended in Gnambs2023
    standard_error = (1 - y**2) / np.sqrt(N - 3)
    return standard_error
    # axis.fill_between(x,y-standard_error,y+standard_error, color=color, alpha=alpha, edgecolor=None)


def plot_corr_with_error_bands(x, y, N, axis, label, color, plot_errorbars=False):
    y_style = {"hatch": "||||", "facecolor": "None", "linewidth": 0}
    x_style = {"hatch": "---", "facecolor": "None", "linewidth": 0}
    x_style["edgecolor"] = color
    y_style["edgecolor"] = color
    # x=5*lags_tau_HKE
    # y=corr_tau_HKE
    x_err = 5
    y_err = get_pearson_standard_error(x=x, y=y, N=N)
    plot_xy_errorbands(x, y, x_err, y_err, x_style, y_style, plot_errorbars=plot_errorbars, ax=axis)
    axis.plot(x, y, color=color, label=label, lw=2.5)


# plot_standard_error(x=5*lags_tau_HKE,y=corr_tau_HKE,N=len(pair["hke"]),axis=ax[1], color="#d95f02", alpha=0.5)
# y_style = {"hatch":"||||", "facecolor":"None", "linewidth":0}
# x_style = {"hatch":"---", "facecolor":"None", "linewidth":0}
# y_style = {"facecolor":"xkcd:pale green", "edgecolor":"k", "alpha":0.9, "linewidth":0}
# x_style = {"facecolor":"xkcd:pale green", "edgecolor":"k", "linewidth":0}

x = 5 * lags_dissip_hke
y = corr_dissip_hke
N = len(dissip_hke_pair["hke"])
label = "$\\varepsilon_\\mathrm{IGW} \\star \\text{HKE}$"
color = "k"
plot_corr_with_error_bands(x, y, N, axis=ax[1], label=label, color=color, plot_errorbars=False)

x = 5 * lags_tau_HKE
y = corr_tau_HKE
N = len(pair["hke"])
label = "$\\tau \\star \\text{HKE}$"
color = "#d95f02"
plot_corr_with_error_bands(x, y, N, axis=ax[1], label=label, color=color, plot_errorbars=False)

x = 5 * lags_mean_flow_HKE
y = corr_mean_flow_HKE
N = len(pair["hke"])
label = "$|\\overline{u}| \\star \\text{HKE}$"
color = "#7570b3"
plot_corr_with_error_bands(x, y, N, axis=ax[1], label=label, color=color, plot_errorbars=False)

x = 5 * lags_N2_HKE
y = corr_N2_HKE
N = len(pair["hke"])
label = "$N^2 \\star \\text{HKE}$"
color = "#1b9e77"
plot_corr_with_error_bands(x, y, N, axis=ax[1], label=label, color=color, plot_errorbars=False)

# x_err = 5
# y_err = get_standard_error(x=5*lags_tau_HKE,y=corr_tau_HKE,)
# plot_xy_errorbands(x,y,x_err,y_err,x_style, y_style,plot_errorbars=True,ax=ax[1])
# ax[1].plot(x,y, "#d95f02", label="($|\\vec{\\tau}| \\star \\text{HKE}$)", lw=2.5)

# x_style["edgecolor"] = "#7570b3"
# y_style["edgecolor"] = "#7570b3"
# x=5*lags_mean_flow_HKE
# y=corr_mean_flow_HKE
# x_err = 5
# y_err = get_standard_error(x=5*lags_mean_flow_HKE,y=corr_tau_HKE,N=len(pair["hke"]))
# plot_xy_errorbands(x,y,x_err,y_err,x_style, y_style,plot_errorbars=True,ax=ax[1])
# ax[1].plot(5*lags_mean_flow_HKE, corr_mean_flow_HKE, "#7570b3", label="($|\\vec{u}| \\star \\text{HKE}$)", lw=2.5)

# # standard_error_mean_flow_HKE = (1-corr_mean_flow_HKE**2)/np.sqrt(len(pair["hke"]))
# # ax[1].fill_between(5*lags_tau_HKE,corr_mean_flow_HKE-standard_error_mean_flow_HKE,corr_mean_flow_HKE+standard_error_mean_flow_HKE, color="#7570b3", alpha=0.7)

# plot_standard_error(x=5*lags_N2_HKE,y=corr_N2_HKE,N=len(pair["hke"]),axis=ax[1], color="#1b9e77", alpha=0.5)
# ax[1].plot(5*lags_N2_HKE, corr_N2_HKE, "#1b9e77", label="($N^2 \\star \\text{HKE}$)", lw=2.5)

# # standard_error_tau_HKE = (1-corr_tau_HKE**2)/np.sqrt(len(pair["hke"]))
# # ax[1].fill_between(5*lags_tau_HKE,corr_tau_HKE-standard_error_tau_HKE,corr_tau_HKE+standard_error_tau_HKE, color="#1b9e77", alpha=0.7)

# #ax[1].plot(5*lags2,corr2_in_percent, "k", alpha = 0.3, label="raw", lw=0.7)
# ax[1].set_xlim(-50,70)
ax[1].set_xlim(-55, 70)
# ax[1].set_xlim(-101,120)
ax[1].set_xlabel("days of lag")
ax[1].set_ylabel("normalized\ncross-correlation / (1)")
# #ax[1].yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
fig.legend(fontsize=7, loc="upper right", framealpha=0.9)
fig.savefig(pinner2026_jpo_code.FIGURE_DIR / "f10_surface_hke_corr.pdf")
