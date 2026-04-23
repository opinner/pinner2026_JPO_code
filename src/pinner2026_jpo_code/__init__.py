# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Ole Pinner
# eMail:  ole.pinner@awi.de
# Date:   2026-04-20
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from .core.utils import save, setup_logger

__all__ = [
    "save",
    "setup_logger",
]  # as recommended in https://docs.astral.sh/ruff/rules/unused-import/#why-is-this-bad to avoid linting error
__version__ = "0.1.0"


# Make some of the basic directories globally available in your environment
BASE_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "reports/figures"
jupyter_startup_script = BASE_DIR / "notebooks/jupyter_startup.ipy"

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
# load up the entries as environment variables
load_dotenv(dotenv_path)

sys.path.append(str(BASE_DIR / "scripts"))
