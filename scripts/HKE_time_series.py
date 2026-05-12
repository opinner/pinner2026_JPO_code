from datetime import datetime

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.helper
import src.plots
import src.spectra

# load all 7 moorings as dataframes
list_of_moorings = src.helper.load_pickle(name="../../data/mooring/list_of_moorings.pkl")
mooring_C = src.helper.load_pickle(name="../preprocessing/mooring_C.pkl")
region_categorization = pd.read_csv("../../derived_data/rcm_infos.csv", index_col=0)
names = ["A", "B", "C", "D", "E", "F", "G"]
moorings = {}
for name, mooring in zip(names, list_of_moorings):
    moorings[name] = mooring.set_index("time")
    print(mooring.location, list(mooring.columns))


def find_latest_start(moorings):
    starttimes = []
    for i, (name, mooring) in enumerate(moorings.items()):
        for j, col in enumerate(mooring.columns):
            if name == "C":
                series = pd.Series(
                    index=mooring_C[f"time_{col}"].values, data=mooring_C[col].values
                )
            else:
                series = mooring[col]
            starttimes.append(series.index[0])

    latest_start = np.max(np.array(starttimes))
    return latest_start


def cut_to_f_N_interval(freq, psd, f, N):
    f_index = np.argmin(np.abs(freq - f))
    N_index = np.argmin(np.abs(freq - N))

    fN_freq = freq[f_index:N_index]
    fN_psd = psd[f_index:N_index]

    if np.shape(fN_freq) == np.shape([]):
        raise AssertionError

    return fN_freq, fN_psd


def get_integration_intervals_for_tidal_peaks(freq, P, tidal_periods):
    integration_indices_intervals = []

    # get interval of width 2P for each tidal frequency
    for tidal_period in tidal_periods:
        tidal_freq_in_cpd = 24 / tidal_period
        tidal_index = np.argmin(np.abs(freq - (tidal_freq_in_cpd)))
        width = (tidal_index - P, tidal_index + P + 1)
        integration_indices_intervals.append(width)

    # because the peaks all have the same width 2P, I can check the ascending order
    # by checking if the start or the end points are in ascending order. I test here both to be extra sure
    def is_monotonic_increasing(L):
        return all(x <= y for x, y in zip(L, L[1:]))

    # slicing a list of tuples is complicated
    interval_start_points = list(zip(*integration_indices_intervals))[0]
    try:
        assert is_monotonic_increasing(interval_start_points)
    except AssertionError as e:
        print(f"{interval_start_points =}")
        raise e

    # slicing a list of tuples is complicated
    interval_end_points = list(zip(*integration_indices_intervals))[1]
    try:
        assert is_monotonic_increasing(interval_end_points)
    except AssertionError as e:
        print(f"{interval_end_points =}")
        raise e

    # check intervals for overlaps until none are found
    while True:
        Overlap = False
        # Iteration is over a copy to not change the object iterated over
        for i, interval in enumerate(integration_indices_intervals.copy()):
            if i + 1 == len(integration_indices_intervals):
                break
            next_interval = integration_indices_intervals[i + 1]

            start1, end1 = interval
            start2, end2 = next_interval

            if end1 >= start2 and end2 >= start1:
                Overlap = True
                integration_indices_intervals[i] = (start1, end2)
                integration_indices_intervals[i + 1] = (start1, end2)

        # remove duplicated intervals
        # by creating a dictionary with the tuples as keys and None values and then converting back to list
        integration_indices_intervals = list(dict.fromkeys(integration_indices_intervals))

        if not Overlap:
            break
    return integration_indices_intervals


def func_powerlaw(x, s, c):
    return x**s * c


def get_r_squared(func, popt, pcov, xdata, ydata):
    # taken from https://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit
    # You can get the residual sum of squares (ss_tot) with
    residuals = ydata - func(xdata, *popt)
    ss_res = np.sum(residuals**2)
    # You can get the total sum of squares (ss_tot) with
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    # And finally, the r_squared-value with,
    r_squared = 1 - (ss_res / ss_tot)
    # assert the fit worked at all
    return r_squared


common_start_time = pd.to_datetime(find_latest_start(moorings)).round("h")
print(f"{common_start_time = }")


TIME_BANDWIDTH_PRODUCT = 3
MEASUREMENT_PERIOD = 1 / 12  # 2 hour timestep

# SEMIDIURNAL_TIDAL_CONSTITUENTS = src.helper.get_tidal_frequencies_in_hours(tide_type="continuum")
# # sort the constituents in descending order after their tidal periods
# # This is important for finding overlapping tidal peaks
# SEMIDIURNAL_TIDAL_CONSTITUENTS = dict(sorted(SEMIDIURNAL_TIDAL_CONSTITUENTS.items(), key=lambda x: x[1], reverse=True))

cmap = plt.cm.viridis
number_of_moorings = len(list_of_moorings)
color_list = [cmap(i / number_of_moorings) for i in range(number_of_moorings)]
color_list.append("xkcd:rose")
discrete_cmap = mcolors.LinearSegmentedColormap.from_list(
    "mooring_ids", color_list, N=number_of_moorings + 1
)

fig, ax = plt.subplots(1, figsize=(6, 3), layout="constrained")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)

results_list = []
for i, (name, mooring) in enumerate(moorings.items()):
    for j, col in enumerate(mooring.columns):
        if name == "C":
            series = pd.Series(index=mooring_C[f"time_{col}"].values, data=mooring_C[col].values)
            series = series[series.index.dropna()]
            assert series.index.is_monotonic_increasing
            series = series.loc[pd.to_datetime(common_start_time) :]
            mooring_id = i
            col_name = name
            if int(col) == 750:
                mooring_id = -1
                col_name = "control"
        else:
            series = mooring[col]
            series = series.loc[pd.to_datetime(common_start_time) :]
            mooring_id = i
            col_name = name

        series = series.dropna()
        timestamps = []
        wave_field_energies = []

        coriolis_frequency_in_cpd = src.helper.get_coriolis_frequency(
            mooring.location.lat, unit="cpd", absolute=True
        )

        print(name, col)
        count = 0
        number_of_days: int = 30
        for run, chunk in enumerate(series.rolling(window=f"{number_of_days}D", center=True)):
            if run % (12 * 5) != 0:
                continue  # skip 5 days every loop
            if len(chunk) < int(number_of_days / MEASUREMENT_PERIOD):
                continue  # skip windows with not enough data points
            count += 1
            freq, velocity_spectrum = src.spectra.total_multitaper(
                chunk.values, dt=MEASUREMENT_PERIOD, P=TIME_BANDWIDTH_PRODUCT
            )
            assert not np.any(np.isnan(velocity_spectrum))
            HKE_spectrum = 0.5 * velocity_spectrum
            HKE_series = pd.Series(index=freq, data=HKE_spectrum)
            fN_interval, fN_HKE_spectrum = cut_to_f_N_interval(
                freq, HKE_spectrum, coriolis_frequency_in_cpd, N=6
            )
            # fN_series = pd.Series(index=fN_interval, data=fN_HKE_spectrum)
            # energy_at_f = pd.Series([fN_series.index[0]], [fN_series.iloc[0]])
            # HKE_spectrum_cut_out_tides = pd.concat([HKE_series.loc[2.4:2.75], HKE_series.loc[3.3:3.7], HKE_series.loc[4.1:6]]) #,, energy_at_f,
            # x, y = HKE_spectrum_cut_out_tides.index, HKE_spectrum_cut_out_tides.values
            # popt, pcov, _ = curve_fit(func_powerlaw, x, y, p0=np.asarray([-2, 1e-4]), bounds=([-2.5,-np.inf], [-1.5,np.inf]))
            # r_squared = get_r_squared(func_powerlaw, popt, pcov, x, y)

            # print(f"{r_squared:.1f}", end=", ")

            HKE = src.spectra.integrate_psd_interval(fN_interval, fN_HKE_spectrum)
            # HKE = src.spectra.integrate_psd_interval(fN_interval,fN_HKE_spectrum,a=4,b=6)

            # horizontal_kinetic_energy_per_peak = []
            # integration_indices_intervals = get_integration_intervals_for_tidal_peaks(
            #     fN_interval, P=TIME_BANDWIDTH_PRODUCT, tidal_periods=SEMIDIURNAL_TIDAL_CONSTITUENTS.values()
            # )
            # print(integration_indices_intervals)
            # # iterate over the tidal peaks and calculate energy per peak
            # for interval_tuple in integration_indices_intervals:
            #     start = interval_tuple[0]
            #     stop = interval_tuple[1]
            #     assert start > 0
            #     a = fN_interval[start]
            #     if interval_tuple[1] > len(fN_interval): stop=-1
            #     b = fN_interval[stop]

            #     plt.semilogy([a,b], [HKE_spectrum[start], HKE_spectrum[stop]], color="tab:red")

            #     peak_integral = src.spectra.integrate_psd_interval(
            #         fN_interval,
            #         fN_HKE_spectrum,
            #         a=a,
            #         b=b
            #     )
            #     horizontal_kinetic_energy_per_peak.append(peak_integral)

            # horizontal_kinetic_energy_at_tidal_frequencies = np.sum(horizontal_kinetic_energy_per_peak)

            # HKE = src.spectra.integrate_psd_interval(fN_interval,HKE_spectrum, a=3.5, b=6)
            timestamp = chunk.index.round("h")[0]
            timestamps.append(timestamp)
            wave_field_energies.append(HKE)  # -horizontal_kinetic_energy_at_tidal_frequencies

            # plt.semilogy(freq, HKE_spectrum)
            # # plt.semilogy(fN_interval, func_powerlaw(fN_interval, *popt), label = f"{tuple(popt)[0]:.1f}, {tuple(popt)[1]:.1e}, {r_squared:.1f}")
            # # for m in [HKE_series.loc[2.4:2.75], HKE_series.loc[3.3:3.7], HKE_series.loc[4.1:6]]:
            # #     plt.semilogy(m.index, m.values, color="tab:red")
            # plt.axvline(coriolis_frequency_in_cpd)
            # plt.legend()
            # plt.show()

        wavefield = pd.Series(index=timestamps, data=wave_field_energies)
        # remove duplicated indices
        wavefield = wavefield[~wavefield.index.duplicated(keep="last")]
        results_list.append((col_name, col, wavefield))
        color = color_list[mooring_id]
        ax.plot(wavefield.index, wavefield.values, color=color, lw=2.5, zorder=i)
        # ax.plot(wavefield.index, wavefield.values-np.mean(wavefield.values), color=color, lw=2.5, zorder=i)

# Build MultiIndex for indexing the final DataFrame
index_tuples = [(lvl1, lvl2) for lvl1, lvl2, _ in results_list]
multi_index = pd.MultiIndex.from_tuples(index_tuples, names=["mooring", "depth"])

# Concatenate Series into a DataFrame
results = pd.DataFrame({(lvl1, lvl2): series for lvl1, lvl2, series in results_list})
# results.to_csv("../../derived_data/HKE.csv", na_rep="NaN")

print(count)
norm = mcolors.Normalize(vmin=0, vmax=1)
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=discrete_cmap), ax=ax)
cbar.ax.set_yticks(np.arange(0.5 / (number_of_moorings + 1), 1, 1 / (number_of_moorings + 1)))
cbar.ax.invert_yaxis()
cbar.ax.set_yticklabels(names + ["control"])
# ax.axvline(datetime(year = 2017, month = 1, day = 1), ls = "--", color = "k", alpha = 0.5)
ax.axvline(datetime(year=2018, month=1, day=1), ls="--", color="k", alpha=0.5)
ax.axvline(datetime(year=2019, month=1, day=1), ls="--", color="k", alpha=0.5)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.set_ylabel(r"Kinetic wave field energy (m$^2\,$s$^{-2}$)")
# fig.savefig("./HKE_timeseries_full_integral.pdf")
# fig.savefig("./HKE_timeseries.pdf")
plt.show()
