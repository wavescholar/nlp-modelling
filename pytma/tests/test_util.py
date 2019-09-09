import unittest

from pytma.Utility import Logger


class TestUtil(unittest.TestCase):
    def test_util(self):
        log = Logger('.', "pytma")

        log.info("information")

        log.debug("debugging info")

        log.warning("warning")

        log.error("error")

        log.critical("critical message")
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
