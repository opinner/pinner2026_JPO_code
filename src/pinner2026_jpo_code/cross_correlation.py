import matplotlib as mpl
import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal


def get_normalized_crosscorr(a, b):
    assert len(a) == len(b)
    n = len(a)

    lags = signal.correlation_lags(len(a), len(b), mode="same")
    c = signal.correlate(a - a.mean(), b - b.mean(), "same") / (np.std(a * np.std(b)))
    cross_correlation = c / n
    return lags, cross_correlation


def get_crosscorr_standard_error(corr, N):
    # standard error of pearson correlation according to Bonett2008, recommended in Gnambs2023
    standard_error = (1 - corr**2) / np.sqrt(N - 3)
    return standard_error


def plot_xy_errorbands(
    x,
    y,
    x_errors,
    y_errors,
    x_style: dict | None = None,
    y_style: dict | None = None,
    x_label: str = "",
    y_label: str = "",
    plot_errorbars: bool = False,
    ax: mpl.axes.Axes = None,
):

    if y_style is None:
        y_style = {}
    if x_style is None:
        x_style = {}
    x_lower = x - x_errors
    x_upper = x + x_errors
    y_lower = y - y_errors
    y_upper = y + y_errors

    if ax is None:
        fig, ax = plt.subplots(1)

    # Build a list of quads between consecutive points
    verts = [
        [
            (x_upper[i], y[i]),
            (x_upper[i + 1], y[i + 1]),
            (x_lower[i + 1], y[i + 1]),
            (x_lower[i], y[i]),
        ]
        for i in range(len(x) - 1)
    ]

    poly = mcollections.PolyCollection(verts, **x_style)
    ax.add_collection(poly)

    ax.fill_between(x, y_lower, y_upper, **y_style, label=y_label)


def plot_corr_with_error_bands(lags, corr, x_err, y_err, axis, label, color):
    y_style = {"hatch": "||||", "facecolor": "None", "linewidth": 0}
    x_style = {"hatch": "---", "facecolor": "None", "linewidth": 0}
    x_style["edgecolor"] = color
    y_style["edgecolor"] = color
    plot_xy_errorbands(lags, corr, x_err, y_err, x_style, y_style, ax=axis)
    axis.plot(lags, corr, color=color, label=label, lw=2.5)
