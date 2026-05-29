import datetime

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.signal as signal

import pinner2026_jpo_code.IO
import pinner2026_jpo_code.location
import pinner2026_jpo_code.mooring
import pinner2026_jpo_code.plots
import pinner2026_jpo_code.spectra

RAW_DATA_DIR = pinner2026_jpo_code.DATA_DIR / "raw"
INTERIM_DATA_DIR = pinner2026_jpo_code.DATA_DIR / "interim"
DERIVED_DATA_DIR = pinner2026_jpo_code.DATA_DIR / "derived"


def filter_to_semidiurnal(series: pd.Series) -> pd.Series:
    fs = 1 / (2 * 60 * 60)  # sampling period of 2 hours
    lowcut = 1 / (14 * 60 * 60)  # period of 14 hours
    highcut = 1 / (10 * 60 * 60)  # period of 10 hours
    order = 40

    sos = signal.butter(order, [lowcut, highcut], "bandpass", fs=fs, output="sos")
    # use filtfilt with 2 filtering passes to prevent an artificial phase shift
    M2 = pd.Series(index=series.index, data=signal.sosfiltfilt(sos, series.values))
    return M2


def locate_spring_and_neap_tides(series: pd.Series) -> [pd.Series, pd.Series]:
    series = series.dropna()
    M2 = filter_to_semidiurnal(series)
    assert not np.any(np.isnan(M2.values))
    stdM2 = M2.abs().rolling(window="48h", center=True).std()
    neap_tides = stdM2.loc[stdM2 == stdM2.rolling(window="14D", center=True).min()]
    spring_tides = stdM2.loc[stdM2 == stdM2.rolling(window="14D", center=True).max()]
    assert len(neap_tides) != 0 and len(spring_tides) != 0
    return [spring_tides, neap_tides]


def generate_time_chunks(series: pd.Series, centers: pd.Series, halfwidth: pd.Timedelta) -> list:
    chunks = []
    for center_index in centers.index:
        a, b = center_index - halfwidth, center_index + halfwidth
        chunk = series[a:b]
        chunks.append(chunk)
    assert len(chunks) != 0
    return chunks


def compute_spectra(chunks: list, centers: pd.Series, P: int = 2):
    # need frequency index, pick the second chunk to ensure a full frequency range
    freq, _ = pinner2026_jpo_code.spectra.total_multitaper(chunks[1].values, dt=1 / 12, P=P)
    frame = pd.DataFrame(index=freq)

    # leave out first and last chunk, as they can be of different length
    chunk_with_nans = 0
    for i, (chunk_center, chunk) in enumerate(zip(centers.iloc[1:-1].index, chunks[1:-1])):
        if np.any(np.isnan(chunk.values)):
            chunk_with_nans += 1
            frame[chunk_center] = np.nan
            continue
        freq, spectrum = pinner2026_jpo_code.spectra.total_multitaper(chunk.values, dt=1 / 12, P=P)
        assert np.all(frame.index == freq)
        frame[chunk_center] = spectrum

    return frame


def generate_spectral_chunks(
    series: pd.Series, centers: pd.Series, halfwidth: pd.Timedelta
) -> list:
    spectral_chunks = []
    for center_index in centers.index:
        a, b = center_index - halfwidth, center_index + halfwidth
        chunk = series[a:b]
        _freq, spectrum = pinner2026_jpo_code.spectra.total_multitaper(chunk.values, dt=1 / 12, P=2)
        spectral_chunks.append(spectrum)
    return spectral_chunks


def get_matching_metadata(name, col, categorization):
    mab_match = categorization.loc[
        (categorization["name"] == name) & (categorization["rounded depth"] == int(col)),
        "rounded mab",
    ]
    if not mab_match.empty:
        mab = mab_match.iloc[0]  # Extract the first match
        if mab == 0:
            mab = 2
    else:
        mab = 1600

    region_match = categorization.loc[
        (categorization["name"] == name) & (categorization["rounded depth"] == int(col)), "region"
    ]
    if not region_match.empty:
        region = region_match.iloc[0]  # Extract the first match
    else:
        region = "column"

    color_match = categorization.loc[
        (categorization["name"] == name) & (categorization["rounded depth"] == int(col)), "color"
    ]
    if not color_match.empty:
        color = color_match.iloc[0]  # Extract the first match
    else:
        color = "xkcd:rose"

    return mab, region, color


def draw_tidal_vlines(measurement, axis):
    # f_cpd = pinner2026_jpo_code.helper.get_coriolis_frequency(
    #     measurement.location.lat, unit="cpd", absolute=True
    # )
    # freqs = np.array(
    #     [24 / (14 * 24), 24 / 12.4, 2 * 24 / 12.4, 4 * 24 / 12.4, f_cpd, 2 * f_cpd]
    # )
    # freq_labels = ["14 days", "M2", "2M2", "4M2", " \nf", " \n2f"]
    freqs = np.array([2, 3, 4, 5])

    ylim = axis.get_ylim()
    axis.vlines(freqs, ymin=ylim[0], ymax=ylim[1], lw=0.5, color="lightgrey", alpha=0.8, zorder=0)


def custom_colorbar(figure, axis):
    color_dict = pinner2026_jpo_code.plots.get_region_color_dict(linebreak=True)
    color_dict.pop("air")
    cmap = mcolors.ListedColormap(color_dict.values())
    mappable = mpl.cm.ScalarMappable(norm=mcolors.Normalize(-0.5, 3.5), cmap=cmap)
    colorbar = figure.colorbar(mappable, ax=axis, orientation="vertical")
    colorbar.set_ticks(ticks=np.arange(4), labels=list(color_dict.keys()))
    colorbar.ax.invert_yaxis()
    return colorbar


def main():
    # load all 7 moorings as dataframes
    list_of_moorings = pinner2026_jpo_code.IO.load_pickle(
        name=INTERIM_DATA_DIR / "list_of_moorings.pkl"
    )
    mooring_C = pinner2026_jpo_code.helper.load_pickle(name="../preprocessing/mooring_C.pkl")
    region_categorization = pd.read_csv("../../derived_data/rcm_infos.csv", index_col=0)
    names = ["A", "B", "C", "D", "E", "F", "G"]
    moorings = {}
    for name, mooring in zip(names, list_of_moorings):
        moorings[name] = mooring.set_index("time")
        print(mooring.location, list(mooring.columns))

    # define figures
    fig, axarr = plt.subplots(ncols=4, nrows=2, sharex=True, sharey=True, layout="constrained")
    ax = np.ravel(axarr)
    fig2, axarr2 = plt.subplots(ncols=4, nrows=2, sharex=True, sharey=True, layout="constrained")
    ax2 = np.ravel(axarr2)
    fig3, ax3 = plt.subplots(1)

    # line style cycler to differentiate multiple measurements in the same layer
    # (currently only needed for mooring A)
    # solid, densely dashed, densely dotted
    ls_cycler = (ls for ls in ["solid", "dashed", (0, (1, 1))])

    for i, (name, mooring) in enumerate(moorings.items()):
        for j, col in enumerate(mooring.columns):
            print(name, col)
            # because there are 7 moorings in the gravity current and 1 control time series far way
            # some reshuffling is necessary for 2 figures to get the subplot order right
            # the data in the gravity current skip subplot 3, where the control is plotted
            if i < 3:
                pos = i
            else:
                pos = i + 1
            if name == "C":
                series = pd.Series(
                    index=mooring_C[f"time_{col}"].values, data=mooring_C[col].values
                )
                series = series.dropna(axis="index")
                if col == "750":
                    pos = 3
            else:
                series = mooring[col]

            spring_tides, neap_tides = locate_spring_and_neap_tides(series)

            spring_chunks = generate_time_chunks(
                series, centers=spring_tides, halfwidth=pd.Timedelta(days=5)
            )
            neap_chunks = generate_time_chunks(
                series, centers=neap_tides, halfwidth=pd.Timedelta(days=5)
            )

            spring_spectra = compute_spectra(spring_chunks, spring_tides)
            neap_spectra = compute_spectra(neap_chunks, neap_tides)

            # average over all chunks
            spring_spectrum = spring_spectra.mean(axis="columns")
            spring_q05 = spring_spectra.quantile(q=0.1, axis="columns")
            spring_q95 = spring_spectra.quantile(q=0.9, axis="columns")
            spring_var = spring_spectra.var(axis="columns")
            # neap_spectra_df = pd.concat(neap_spectra, axis="columns", keys=range(len(neap_spectra)))
            neap_spectrum = neap_spectra.mean(axis="columns")
            neap_q05 = neap_spectra.quantile(q=0.1, axis="columns")
            neap_q95 = neap_spectra.quantile(q=0.9, axis="columns")
            neap_var = spring_spectra.var(axis="columns")

            tidal_ratio = spring_spectrum / neap_spectrum
            tidal_ratio_std = (
                (spring_spectrum**2 / neap_spectrum**2)
                * ((spring_var / spring_spectrum**2) + (neap_var / neap_spectrum**2))
            ).pow(0.5)

            mab, region, color = get_matching_metadata(name, col, region_categorization)
            label = rf"{mab:4}$\,$mab"  # , {region}"

            ax[pos].semilogy(
                spring_spectrum.index, spring_spectrum.values, label=label, color=color
            )
            ax[pos].fill_between(spring_q05.index, spring_q05, spring_q95, alpha=0.2, color=color)
            ax[pos].semilogy(
                neap_spectrum.index, neap_spectrum.values, label=label, ls="--", color=color
            )
            ax[pos].fill_between(neap_q05.index, neap_q05, neap_q95, alpha=0.2, color=color)

            ls = "solid"
            if name == "A":
                ls = next(ls_cycler)

            ax2[pos].plot(
                tidal_ratio.index, tidal_ratio.values, label=label, color=color, lw=2, ls=ls
            )
            ax2[pos].fill_between(
                tidal_ratio.index,
                (tidal_ratio - tidal_ratio_std).values,
                (tidal_ratio + tidal_ratio_std).values,
                alpha=0.1,
                color=color,
            )

            ax3.plot(
                spring_tides.index,
                len(spring_tides) * [i + 0.1 * j],
                "o",
                color=color,
                markeredgecolor="tab:red",
            )
            ax3.plot(neap_tides.index, len(neap_tides) * [i + 0.1 * j], "s", color=color)

        # after each mooring is done, label the subplot
        if i < 3:
            pos = i
        else:
            pos = i + 1
        draw_tidal_vlines(mooring, ax2[pos])
        ax[pos].legend(fontsize=7)
        ax2[pos].legend(fontsize=7, handlelength=1.5, loc="upper right")
        ax[pos].set_title(f"mooring {name}")
        ax2[pos].set_title(f"mooring {name}")
        ax2[pos].axhline(1, color="k", alpha=0.8, zorder=5, ls="--")

    # label the control subplot 3
    ax[3].set_title(f"Control")
    ax2[3].set_title(f"Control")
    ax2[3].axhline(1, color="k", alpha=0.8, zorder=5, ls="--")
    ax[3].legend(fontsize=7)
    ax2[3].legend(fontsize=7, handlelength=1.5, loc="upper right")
    draw_tidal_vlines(moorings["C"], ax2[3])

    ax[0].set_xlim((1.5, 6))
    ax2[0].set_xlim((1.5, 6))
    ax2[0].set_ylim((0.7, 2.5))
    _colorbar = custom_colorbar(fig2, axarr2)
    fig2.supylabel(r"Spectral spring/neap ratio $\mathcal{R}$ (1)")
    fig2.supxlabel("Cycles per day (cpd)")
    fig2.savefig("spring_nipp_ratio.pdf")
    plt.show()


if __name__ == "__main__":
    main()
