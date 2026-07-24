from pathlib import Path

import pandas as pd

import pinner2026_jpo_code

INTERIM_DATA_DIR: Path = pinner2026_jpo_code.DATA_DIR / "interim"
DERIVED_DATA_DIR: Path = pinner2026_jpo_code.DATA_DIR / "derived"


def test_horizontal_kinetic_energy_output_all_moorings(dataframe_regression):
    HKE_df = pd.read_csv(
        INTERIM_DATA_DIR / "HKE.csv",
        index_col=0,
        parse_dates=True,
    )
    dataframe_regression.check(HKE_df, default_tolerance=dict(rtol=1e-05, atol=1e-10))


def test_dissipation_rate_time_series_output_at_C(dataframe_regression):
    eps_IGW_at_C_df = pd.read_csv(
        DERIVED_DATA_DIR / "wave_induced_dissipation_rate_time_series_at_C.csv",
        index_col=0,
        parse_dates=True,
    )
    dataframe_regression.check(eps_IGW_at_C_df, default_tolerance=dict(rtol=1e-05, atol=1e-10))
