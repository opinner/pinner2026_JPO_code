import numpy as np


def get_tidal_frequencies_in_hours(tide_type="semidiurnal"):

    # tidal_constit = {"M2": 12.42, "S2": 12.00, "N2":12.66}
    # the diurnal constituents, K1, O1, P1, Q1, and S1, with periods of 23.93, 25.82, 24.07, 26.87, and 24.00 h,
    # semidiurnal constituents M2, S2, N2, and S2, with periods of 12.42, 12.00, 12.66, and 11.97 h,
    if tide_type.lower() == "padman":
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

    if tide_type.lower() == "continuum":
        return {"M2": 12.42, "S2": 12.00, "N2": 12.66, "K2": 11.96, "3S2": 4, "4S2": 3}

    elif tide_type.lower() == "semidiurnal":
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

    if absolute:
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
