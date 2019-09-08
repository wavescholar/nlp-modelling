# -*- coding: utf-8 -*-

import os.path

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 0
_version_micro = 2  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 2 - Pre-Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Topic Modelling"]

description = "pytma: Topic Modelling"
# Long description will go up on the pypi page
long_description = """

pytma
========
pytma is a library for topic modelling using.

Some of the repeatable components of a topic modelling workflow are 
functionalized. Several vignettes demonstrating topic modelling workflows
are included. 

To get started using these components in your own software, please go to the
repository README_.

.. _README: https://github.com/uwescience/pytma/blob/master/README.md

License
=======
``pytma`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

"""

NAME = "pytma"
MAINTAINER = "Bruce Campbell"
MAINTAINER_EMAIL = "bruce@aloidia.solutions"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/brucebcampbell/pytma"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Bruce Campbell"
AUTHOR_EMAIL = "bruce@aloidia.solutions"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'pytma': [os.path.join('data', '*')]}
REQUIRES = ["numpy"]
