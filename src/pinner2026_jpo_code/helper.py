import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# -------------------------------------------
# Functions for Conversions
# -------------------------------------------


def matlab2datetime(matlab_datenum):
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum % 1) - dt.timedelta(days=366)
    return day + dayfrac


# -------------------------------------------
# Functions for loading data
# -------------------------------------------


def save_as_pickle(figure, name="figure"):
    import pickle

    # save the mpl figure as pickle format
    with open(f"{name}", "wb") as fs:
        pickle.dump(figure, fs)
    return


def load_pickle(name="figure"):
    import pickle

    with open(f"{name}", "rb") as fs:
        return pickle.load(fs)


def get_filepaths_from_directory(directory="/media/sf_VM_Folder/data", inclusive="", exclusive=()):

    # endswith method requires string or tuple of strings
    if type(inclusive) is not str and type(inclusive) is not tuple:
        inclusive = tuple(inclusive)
    if not type(inclusive) is not str and type(inclusive) is not tuple:
        exclusive = tuple(exclusive)

    def all_files(folder):
        import os

        for path, dirs, files in os.walk(folder):
            for f in files:
                yield os.path.join(path, f)

    list_of_files = [f for f in all_files(directory)]

    return [f for f in list_of_files if f.endswith(inclusive) and not f.endswith(exclusive)]


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 1/7200 #sampling period of 2 hours
    lowcut = 1/(14*60*60) #period of 18 hours
    highcut = 1/(10*60*60) #period of 10 hours
    filtered_u = butter_bandpass_filter(patched_u, lowcut, highcut, fs, order=6)
    filtered_v = butter_bandpass_filter(patched_v, lowcut, highcut, fs, order=6)
    """
    # from scipy.signal import lfilter
    # b, a = Data.butter_bandpass(lowcut, highcut, fs, order=order)
    # y = lfilter(b, a, data)

    import scipy.signal as signal

    # use of second-order sections (sos) as recommended in https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.signal.butter.html
    sos = signal.butter(order, [lowcut, highcut], "bandpass", fs=fs, output="sos")
    # use filtfilt with 2 filtering passes to prevent an artificial phase shift
    filtered = signal.sosfiltfilt(sos, data)

    return filtered


# -------------------------------------------
# Data Analysis
# -------------------------------------------


def get_tidal_frequencies_in_hours(tide_type="semidiurnal"):

    # tidal_constit = {"M2": 12.42, "S2": 12.00, "N2":12.66}
    # the diurnal constituents, K1, O1, P1, Q1, and S1, with periods of 23.93, 25.82, 24.07, 26.87, and 24.00 h,
    # semidiurnal constituents M2, S2, N2, and S2, with periods of 12.42, 12.00, 12.66, and 11.97 h,
    if tide_type == "Padman":
        return {
            "M2": 12.42,
            "S2": 12.00,
            "N2": 12.66,
            "K2": 11.96,
            "K1": 23.93,
            "O1": 25.82,
            "P1": 24.07,
            "Q1": 26.87,
            "Mf": 327.86,
            "Mm": 661.31,
        }

    if tide_type == "continuum":
        return {"M2": 12.42, "S2": 12.00, "N2": 12.66, "K2": 11.96, "3S2": 4, "4S2": 3}

    elif tide_type == "semidiurnal":
        return {"M2": 12.42, "S2": 12.00, "N2": 12.66, "K2": 11.96}

    else:
        raise NotImplementedError
        tidal_constit = {
            "M2": 12.42,
            "K1": 23.93,
            "S2": 12.00,
            "O1": 25.82,
            "P1": 24.07,
            "N2": 12.66,
        }  # , "K2"}
        return tidal_constit


def get_coriolis_frequency(lat, unit="rad/s", absolute=False):

    if absolute == True:
        lat = abs(lat)

    Omega = 7.2921 * 1e-5  # unit: rad/s

    f = 2 * Omega * np.sin(lat * np.pi / 180.0)

    if unit == "rad/s":
        return f

    if unit == "Hz":
        return f / (2 * np.pi)

    if unit == "cpd":
        return f / (2 * np.pi) * 3600 * 24

    raise NotImplementedError("Specified unit is not implemented")


def cut_to_f_N_interval(freq, psd, f, N):
    f_index = np.argmin(np.abs(freq - f))
    N_index = np.argmin(np.abs(freq - N))

    fN_freq = freq[f_index:N_index]
    fN_psd = psd[f_index:N_index]

    if np.shape(fN_freq) == np.shape([]):
        raise AssertionError

    return fN_freq, fN_psd


def get_bathymetry_data(path="/media/sf_VM_Folder/data/bathymetry.mat"):
    data = sio.loadmat(path)
    # print(data.keys())
    # print("\n")
    lat = data["lat"]
    lon = data["lon"]
    z = data["z"]

    return lat, lon, z


# -------------------------------------------
# Plotting
# -------------------------------------------


def path_as_footnote(fig=None, path="", rot="vertical"):
    """
    Add the script origin as a footnote below or to the right side of a figure
    """

    if fig == None:
        fig = plt.gcf()

    if path == None:
        import os

        try:
            path = os.path.realpath(__file__)
        except NameError:
            # if run in Jupyter Notebook __file__ is not defined
            # and the path is only the passed string
            path = os.path.abspath("")

    if rot == "vertical":
        plt.figtext(
            0.99,
            0.5,
            f"{path}",
            horizontalalignment="right",
            va="center",
            figure=fig,
            rotation="vertical",
            fontsize=6,
        )
    else:
        plt.figtext(
            0.99, 0, f"{path}", horizontalalignment="right", va="bottom", figure=fig, fontsize=6
        )


def add_tidal_and_f_ticks(
    ax, f_cpd, lower_bound=1e-6, upper_bound=1e-3, freqs=None, freqs_labels=None
):

    if freqs is None and freqs_labels is None:
        freqs = np.array(
            [24 / (14 * 24), 24 / 12.4, 2 * 24 / 12.4, 4 * 24 / 12.4, f_cpd, 2 * f_cpd, 1]
        )
        freq_labels = ["14 days", "M2", "2M2", "4M2", " \nf", " \n2f", "K1"]

    axis = ax.get_ylim()
    for freq in freqs:
        if freq == 24 / (14 * 24):
            continue
        ax.vlines(
            freq, lower_bound, upper_bound, color="k", alpha=0.5, linestyle="-", linewidth=0.75
        )
    ax.set_ylim(axis)

    ax2 = ax.secondary_xaxis(location="bottom")
    ax2 = _axstyle(ax2, ticks="in", grid=False, spine_offset=40)
    ax2.xaxis.set_ticks([])
    ax2.xaxis.set_ticklabels([])
    ax2.minorticks_off()
    ax2.xaxis.set_ticks(freqs)
    ax2.xaxis.set_ticklabels(freq_labels)

    return ax2


def _axstyle(
    ax=None,
    fontsize=9,
    nospine=False,
    grid=True,
    ticks="off",
    ticklength=2,
    spine_offset=5,
):
    """
    Apply own style to axis.

    Parameters
    ----------
    ax : AxesSubplot (optional)
        Current axis will be chosen if no axis provided

    Returns
    -------
    ax : AxesSubplot
        Axis handle
    """
    # find out background color - if this is set to ayu dark, adjust some axis
    # colors
    figcolor = plt.rcParams["figure.facecolor"]
    dark = True if figcolor == "#0d1318" else False

    if ax is None:
        ax = plt.gca()

    # Remove top and right axes lines ("spines")
    spines_to_remove = ["top", "right"]
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)

    # Remove bottom and left spines as well if desired
    if nospine:
        more_spines_to_remove = ["bottom", "left"]
        for spine in more_spines_to_remove:
            ax.spines[spine].set_visible(False)

    # For remaining spines, thin out their line and change
    # the black to a slightly off-black dark grey
    almost_black = "#262626"
    # if figure background is dark, set this close to white
    if dark:
        almost_black = "#ebe6d7"

    if ticks == "off":
        # Change the labels to the off-black
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=fontsize,
            colors=almost_black,
        )
        # Get rid of ticks.
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")
    elif ticks == "in":
        # Change the labels to the off-black
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=fontsize,
            colors=almost_black,
            direction="in",
            length=ticklength,
        )

    spines_to_keep = ["bottom", "left"]
    for spine in spines_to_keep:
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color(almost_black)
        ax.spines[spine].set_position(("outward", spine_offset))

    # Change the labels to the off-black
    ax.yaxis.label.set_color(almost_black)
    ax.yaxis.label.set_size(fontsize)
    ax.yaxis.offsetText.set_fontsize(fontsize)
    ax.xaxis.label.set_color(almost_black)
    ax.xaxis.label.set_size(fontsize)
    ax.xaxis.offsetText.set_fontsize(fontsize)

    # Change the axis title to off-black
    ax.title.set_color(almost_black)
    ax.title.set_size(fontsize + 1)

    # turn grid on
    if grid:
        ax.grid(
            which="major",
            axis="both",
            color="0.5",
            linewidth=0.25,
            linestyle="-",
            alpha=0.8,
        )

    # change legend fontsize (if there is one)
    try:
        plt.setp(ax.get_legend().get_texts(), fontsize=fontsize)
    except AttributeError:
        noleg = 1

    return ax
