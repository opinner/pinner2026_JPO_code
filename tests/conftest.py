# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Ole Pinner
# eMail:  ole.pinner@awi.de
# Date:   2026-04-20
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import pytest


@pytest.fixture
def global_fixture(request):
    return "Test"
