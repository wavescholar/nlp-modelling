## pytma
[![Build Status](https://travis-ci.com/brucebcampbell/nlp-modelling.svg?branch=master)](https://travis-ci.com/brucebcampbell/nlp-modelling.svg?branch=master)

[![codecov](https://codecov.io/gh/brucebcampbell/nlp-modelling/branch/master/graph/badge.svg)](https://codecov.io/gh/brucebcampbell/nlp-modelling)

[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)



nlp-modelling is a Python project that demonstrating a variety of NLP modelling tasks. There are a number of workflows that call into modules for topic modelling and document classification.

### Organization of the  project

This project uses TravisCI for continuous integration, Sphinx for documentation, and Make for Pep8. Unit tests are run with pytest.
We have tried to implement integration with Codecov - but that's been intermittent. For now we'll put the coverage
reports in the repository. 
The project has the following structure:

```
├── doc
│   ├── api.rst
│   ├── conf.py
│   ├── index.rst
│   ├── Makefile
│   └── theory.rst
├── coverage.xml
├── Dockerfile
├── LICENSE
├── Makefile
├── pytma
│   ├── CTMModel.py
│   ├── data
│   │   ├── cache
│   │   │   ├── AnnaKarenina.txt
│   │   │   ├── Boyhood.txt
│   │   │   ├── Childhood.txt
│   │   │   ├── LDAAnalysis.pkl
│   │   │   ├── LDAAnalysisPreprocessed.pkl
│   │   │   ├── LDAWorkflow.preprocesed.pkl
│   │   │   ├── PredictionExample.lda_feature_vecs.pkl
│   │   │   ├── PredictionExample.preprocesed.pkl
│   │   │   ├── PredictionExample.term_topics.pkl
│   │   │   ├── TheCossacks.txt
│   │   │   ├── TheKreutzerSonata.txt
│   │   │   ├── WarAndPeace.txt
│   │   │   └── Youth.txt
│   │   ├── mtsamples.csv
│   │   ├── ortho.csv
│   │   └── para.csv
│   ├── DataSources.py
│   ├── Featurize.py
│   ├── __init__.py
│   ├── Lemmatize.py
│   ├── Parse.py
│   ├── POSTag.py
│   ├── Predict.py
│   ├── Preprocess.py
│   ├── Sentiment.py
│   ├── StopWord.py
│   ├── tests
│   │   ├── __init__.py
│   │   ├── test_CTMModel.py
│   │   ├── test_DataSources.py
│   │   ├── test_example.py
│   │   ├── test_Featureize.py
│   │   ├── test_gensim.py
│   │   ├── test_StanfordNLP.py
│   │   └── test_utility.py
│   ├── Tokenizer.py
│   ├── TopicModel.py
│   ├── Utility.py
│   └── version.py
├── README.md
├── references
├── requirements-dev.txt
├── requirements.txt
├── scripts
│   ├── doc_classification_results.txt
│   ├── LDAWorkflow.py
│   ├── PredictionExample.py
├── setup.py
```

In the module code, we follow the convention that all functions are either
imported from other places, or are defined in lines that precede the lines that
use that function. This helps readability of the code, because you know that if
you see some name, the definition of that name will appear earlier in the file,
either as a function/variable definition, or as an import from some other module
or package.

We try to follow the
[PEP8 code formatting standard](https://www.python.org/dev/peps/pep-0008/), and  enforce this by running a code-linter
[`flake8`](http://flake8.pycqa.org/en/latest/), which automatically checks the
code and reports any violations of the PEP8 standard (and checks for other
  general code hygiene issues), see below.

### Project Data
The main project data is rather small, and recorded in csv files.  Thus, it can be stored 
alongside the module code. The data can be found in `pytma/data`
We pull Tolstoy novels from the Gutenberg Project and store them in `pytma/data/cache` 
We also use the cache director to store pickled models and preprocessing checkpoints
 
### Testing

We use the ['pytest'](http://pytest.org/latest/) library for
testing. The `py.test` application traverses the directory tree in which it is
issued, looking for files with the names that match the pattern `test_*.py`
(typically, something like our `pytma/tests/test_pytma.py`). Within each
of these files, it looks for functions with names that match the pattern
`test_*`. Typically each function in the module would have a corresponding test
(e.g. `test_transform_data`).

To run the tests on the command line, change your present working directory to
the top-level directory of the repository,and type:

    py.test pytma

This will exercise all of the tests in your code directory. If a test fails, you
will see a message such as:

```

    pytma/tests/test_pytma.py .F...

    =================================== FAILURES ===================================
    ________________________________ test_that_failes ________________________________


  pytma/tests/test_pytma.py:49: AssertionError
    ====================== 1 failed, 4 passed in 0.82 seconds ======================
```



The `Makefile` allows you to run the tests with more
verbose and informative output from the top-level directory, by issuing the
following from the command line:

    make test

### Styling

Run `flake8` :

```
flake8 --ignore N802,N806 `find . -name *.py | grep -v setup.py | grep -v /doc/`
```

This means, check all .py files, but exclude setup.py and everything in
directories named "doc". Do all checks except N802 and N806, which enforce
lowercase-only names for variables and functions.

The `Makefile` contains an instruction for running this command as well:

    make flake8

### Documentation

We follow the [numpy docstring
standard](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt),
which specifies in detail the inputs/outputs of every function, and specifies
how to document additional details, such as references to scientific articles,
notes about the mathematics behind the implementation, etc.

To document `pytma` we use the [sphinx documentation
system](http://sphinx-doc.org/). You can follow the instructions on the sphinx
website, and the example [here](http://matplotlib.org/sampledoc/) to set up the
system. Sphinx uses a `Makefile` to build different outputs of your documentation. Forexample, if you want to generate the HTML rendering of the documentation (web
pages that you can upload to a website to explain the software), you will type:

	make html

This will generate the static webpages in the `doc/_build/html`, which you
can then upload to a website of your choice.


We also deploy documentation to, [readthedocs.org](https://readthedocs.org)
![RTD conf](https://github.com/uwescience/pytma/blob/master/doc/_static/RTD-advanced-conf.png)

 http://pytma.readthedocs.org/en/latest/


### Installation

For installation and distribution we will use the python standard
library `distutils` module. This module uses a `setup.py` file to
figure out how to install your software on a particular system. For a
small project such as this one, managing installation of the software
modules and the data is rather simple.

A `pytma/version.py` contains all of the information needed for the
installation and for setting up the [PyPI
page](https://pypi.python.org/pypi/pytma) for the software. This
also makes it possible to install your software with using `pip` and
`easy_install`, which are package managers for Python software. The
`setup.py` file reads this information from there and passes it to the
`setup` function which takes care of the rest.

Much more information on packaging Python software can be found in the
[Hitchhiker's guide to
packaging](https://the-hitchhikers-guide-to-packaging.readthedocs.org).


### Continuous integration : Travis-CI

For `pytma`, we use the
[`Miniconda`](http://conda.pydata.org/miniconda.html) software distribution (not
to be confused with [`Anaconda`](https://store.continuum.io/cshop/anaconda/),
though they are similar and both produced by Continuum).

For details on setting up Travis-CI with github, see Travis-CI's
[getting started
page](https://docs.travis-ci.com/user/getting-started/#To-get-started-with-Travis-CI%3A).

### Distribution

The main venue for distribution of Python software is the [Python
Package Index](https://pypi.python.org/), or PyPI, also lovingly known
as "the cheese-shop".

To distribute your software on PyPI, you will need to create a user account on
[PyPI](http://python-packaging-user-guide.readthedocs.org/en/latest/distributing/#register-your-project).
It is recommended that you upload your software using
[twine](http://python-packaging-user-guide.readthedocs.org/en/latest/distributing/#upload-your-distributions).

Using Travis, you can automatically upload your software to PyPI,
every time you push a tag of your software to github. The instructions
on setting this up can be found
[here](http://docs.travis-ci.com/user/deployment/pypi/). You will need
to install the travis command-line interface

### Licensing

We use the MIT license. You can read the conditions of the license in the
`LICENSE` file.


### Scripts

The scripts directory contains workflow scripts and Jupyter Notebooks. Generally we prefer to keep code in .py files. Using Sphynx and rst to blend latex, plots, and python is a great alternative to Jupyter. Otherwise, using Jupyter like R markdown to render static html is acceptable if you're mainly calling into module code.  Generally we try to keep away from doing lots of development in the Jupyer ecosystem. This way by using small and incremental commits of Python code to git repositories we can see how the project evolved. This is better than doing lots of upfront development in a notebook and then trying to refactor it.  We maintain that over time - starting with good devops practices more agile than the quick and dirty notebook style of data science development.
