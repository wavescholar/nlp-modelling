#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import nltk


def nltk_init(datasets):
    """
    Function that initializes NLTK data sets

    Parameters
    ----------
    sets : list of data sets to get.
        ['stopwords','punkt','averaged_perceptron_tagger','wordnet']

    Returns
    -------
    """
    for set in datasets:
        try:
            nltk.download(set)
        except:
            print("nltk.download fail on " + set)
            raise


class LogError(Exception):
    pass


class Logger:
    """
    Log wrapper
        Sets up logging to pipe to file and stdout

    :parameter

    """

    def __init__(self, log_path, file_name):
        try:
            logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
            rootLogger = logging.getLogger()

            fileHandler = logging.FileHandler("{0}/{1}.log".format(log_path, file_name))
            fileHandler.setFormatter(logFormatter)
            rootLogger.addHandler(fileHandler)

            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(logFormatter)
            rootLogger.addHandler(consoleHandler)

            logging.getLogger().setLevel(logging.INFO)

        except Exception as e:
            raise LogError

    def setLogLevel(self, level):
        """
        Set the global log level
        logging.****
        DEBUG :Detailed information, typically of interest only when diagnosing problems.
        INFO: Confirmation that things are working as expected.
        WARNING: An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected.
        ERROR:Due to a more serious problem, the software has not been able to perform some function.
        CRITICAL:A serious error, indicating that the program itself may be unable to continue running.

        :param level:
        CRITICAL = 50
        ERROR = 40
        WARNING = 30
        INFO = 20
        DEBUG = 10
        NOTSET = 0

        :return:
        """

    # DEBUG :Detailed information, typically of interest only when diagnosing problems.
    def debug(self, msg):
        logging.debug(msg)

    # INFO: Confirmation that things are working as expected.
    def info(self, msg):
        logging.info(msg)

    # WARNING: An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected.
    def warning(self, msg):
        logging.warning(msg)

    # ERROR:Due to a more serious problem, the software has not been able to perform some function.
    def error(self, msg):
        logging.error(msg)

    # CRITICAL:A serious error, indicating that the program itself may be unable to continue running.
    def critical(self, msg):
        logging.critical(msg)


"""
Global Logger for pytma
"""
log = Logger('.', "pytma")

if __name__ == '__main__':

    from pytma.tests.test_util import TestUtil

    TestUtil.test_util()
