import matplotlib as plt
import matplotlib.dates as mdates
import numpy as np

cm = 1 / 2.54  # centimeters in inches
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white", "font.size": 9})

# fig, ax = plt.subplots(nrows=2, figsize=(18*cm, 12*cm),layout="constrained")
fig, ax = plt.subplots(
    ncols=1, figsize=(20 * cm, 6 * cm), layout="constrained"
)  # ,width_ratios=[2,1])
ax = [ax]

line1 = ax[0].plot(
    pair.index,
    pair["hke"],
    "k",
    label="HKE",
    lw=3,
)  # momentum_transfer["C"].abs(), hke["C","2350"]
# ax[0].plot(hke["C","2350"].index, hke["C","2350"], "k", alpha=0.3, lw=0.7)
ax[0].set_ylabel(r"HKE / (m$^2\,$s$^{-2}$)", fontweight="bold")

twinx3 = ax[0].twinx()
twinx3.set_ylabel("$N^2$ / (rad$^2\,$s$^{-2}$)")
# twinx3.spines['right'].set_position(('axes', 1.4)) # shift by 10% of the figure width
twinx3.spines["right"].set_color("#1b9e77")
twinx3.set_ylim(1.3e-6, 2.5e-6)
# twinx2.set_ylim(0.04,np.max(smoothed_truncated_mean_flow)*1.1)
twinx3.tick_params(axis="y", colors="#1b9e77")
twinx3.yaxis.label.set_color("#1b9e77")
line4 = twinx3.plot(
    nsquared.index, smoothed(nsquared, 20), color="#1b9e77", alpha=0.8, lw=2.2, label="$N^2"
)  # , label="$|\\tau|$")

twinx = ax[0].twinx()
twinx.spines["right"].set_position(("axes", 1.25))
twinx.set_ylabel("surface stress $|\\tau|$ / (N$\,$m$^{-2}$)", fontweight="bold")
line2 = twinx.plot(pair.index, pair["tau"], "#d95f02", label="$|\\tau|$", lw=3)
# twinx.plot(momentum_transfer.index, momentum_transfer["C"].abs(), color="#d95f02", alpha = 0.4) #, label="$|\\tau|$")
twinx.spines["right"].set_color("#d95f02")
twinx.tick_params(axis="y", colors="#d95f02")
twinx.yaxis.label.set_color("#d95f02")

twinx2 = ax[0].twinx()
twinx2.set_ylabel("mean flow $|u + $i$v|$ / (m$\,$s$^{-1}$)")
twinx2.spines["right"].set_position(("axes", 1.1))  # shift by 10% of the figure width
twinx2.spines["right"].set_color("#7570b3")
smoothed_truncated_mean_flow = smoothed(truncated_mean_flow, 6)
# twinx2.set_ylim(0.04,np.max(smoothed_truncated_mean_flow)*1.1)
twinx2.tick_params(axis="y", colors="#7570b3")
twinx2.yaxis.label.set_color("#7570b3")

line3 = twinx2.plot(
    smoothed_truncated_mean_flow.index,
    smoothed_truncated_mean_flow,
    color="#7570b3",
    alpha=0.8,
    lw=2.2,
    label="mean flow",
)  # , label="$|\\tau|$")


locator = mdates.AutoDateLocator(minticks=8, maxticks=12)
formatter = mdates.ConciseDateFormatter(locator)
ax[0].xaxis.set_major_locator(locator)
ax[0].xaxis.set_major_formatter(formatter)
ax[0].set_facecolor("None")
ax[0].set_zorder(10)
# # added these three lines
# lns = line3+line2+line1
# labs = [l.get_label() for l in lns]
# ax[0].legend(lns, labs, ncols=3, loc="upper center", fontsize=7)
fig.savefig(pinner2026_jpo_code.FIGURE_DIR / "f07_surface_hke.pdf")
