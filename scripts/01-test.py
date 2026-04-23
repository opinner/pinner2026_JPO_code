#!/usr/bin/env python
# -*- coding utf-8 -*-
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Ole Pinner
# eMail:  ole.pinner@awi.de
# Date:   2026-04-20
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
"""The functions below only serve for demonstrating the logger setup. Feel free to delete them
as well as this docstring once the concept is understood :-)
"""
import logging

from pinner2026_jpo_code import *
from pinner2026_jpo_code.submodule import generate_int_list

# This is only necessary if this file gets imported by another one so that logs get piped
logger = logging.getLogger(__name__)
# Alternatively, set `logger = setup_logger()` as shown below.


def main():
    logger.info("Calling main function, doing nothing")


if __name__ == '__main__':
    logger = setup_logger()
    main()
    logger.info("Calling function from submodule")
    mylist = generate_int_list(6)
    print(mylist)
