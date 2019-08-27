import pytest

import pytma
from pytma import Utility


class LogError(object):
    pass

def test_logger():
    """
    Testing the logging helper.

    """
    log = Utility.Logger('.', "pytma")

    try:
        log.info("information")
    except LogError:
        pytest.fail("Unexpected LogError ..")

    try:
        log.debug("debugging info")
    except LogError:
        pytest.fail("Unexpected LogError ..")

    try:
        log.warning("warning")
    except LogError:
        pytest.fail("Unexpected LogError ..")

    try:
        log.error("error")
    except LogError:
        pytest.fail("Unexpected LogError ..")

    try:
        log.critical("critical message")
    except LogError:
        pytest.fail("Unexpected LogError ..")
