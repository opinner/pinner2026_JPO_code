# !/usr/bin/env python3
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Ole Pinner
# eMail:  ole.pinner@awi.de
# Date:   2026-04-20
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import pytest
from pinner2026_jpo_code.submodule import generate_int_list

def test_subfunc(global_fixture):
    l = generate_int_list()
    assert isinstance(l, list)
    assert isinstance(global_fixture, str)
