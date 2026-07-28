from pathlib import Path

import dissipy  #
import pandas as pd

import pinner2026_jpo_code

RAW_DATA_DIR: Path = pinner2026_jpo_code.DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = pinner2026_jpo_code.DATA_DIR / "interim"


# RAW_DATA_DIR /

eps_IGW, diag = dissipy.dissipation_rate_time_series()


dissipation_rate_time_series = pd.Series(index=time, data=eps_IGW)


dissipation_rate_time_series.to_csv(
    DERIVED_DATA_DIR / "wave_induced_dissipation_rate_time_series_at_C.csv"
)
