#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import wget
import pytma
from pytma.Utility import log


class DataSourceError(Exception):
    pass

def get_transcription_data():
    """
        Function that returns medical transcription data
        This data was scraped from mtsamples.com
        data schema
            - description: Short description of transcription
            - medical_specialty: Medical specialty classification of transcription
            - sample_name: Transcription title
            - transcription: Sample medical transcriptions
            - keywords: Relevant keywords from transcription

        Parameters
        ----------

        Returns
        medical_df : pandas data frame with transcription data.
        -------
        """
    try:
        log.info("preparing transcription data")
        data_path = os.path.join(os.path.dirname(__file__), 'data')
        file_name = data_path + "/mtsamples.csv"
        medical_df = pd.read_csv(file_name)
        medical_df = medical_df.dropna(axis=0, how='any')
        return medical_df
    except Exception as e:
        print(e.message, e.args)
        raise DataSourceError


def download_tolstoy_novels(location='data/cache/'):
    """
    Download Tolstoy novels from project Gutenberg
    These are placed in the cache folder

        - Anna Karenina
        - Boyhood
        - Childhood
        - The Cossacks
        - The Kreutzer Sonata
        - Youth

    ==================

    These are works we'd like to add to the collection :

        - Resurrection
        - The Death of Ivan Ilyich
        - Family Happiness
        - Hadji Murat

    :return:
    """
    if not os.path.exists(location):
        try:
            os.mkdir(location)
        except OSError:
            log.info("Creation of the cache directory failed")
        else:
            log.info("Successfully created the cache directory ")

    items = dict()

    items["AnnaKarenina"] = "https://www.gutenberg.org/files/1399/1399-0.txt"
    items["Boyhood"] = "https://www.gutenberg.org/files/2450/2450-0.txt"
    items["Childhood"] = "https://www.gutenberg.org/files/2142/2142-0.txt"
    items["TheCossacks"] = "https://www.gutenberg.org/ebooks/4761.txt.utf-8"
    items["TheKreutzerSonata"] = "https://www.gutenberg.org/files/689/689-0.txt"
    items["Youth"] = "https://www.gutenberg.org/files/2637/2637-0.txt"
    items["WarAndPeace"] = "https://www.gutenberg.org/files/2600/2600-0.txt"

    for item in items:
        try:
            log.info(item)
            out_name = location + item + '.txt'
            if not os.path.exists(out_name):
                filename = wget.download(items[item], out =out_name)
                log.info("Dowloaded : " + filename)

            else:
                log.info("Found in cache : " + out_name)

        except Exception as e:
            log.error("problem with getting novel " + out_name)
            raise DataSourceError

if __name__ == '__main__':

    from pytma.tests.test_DataSources import test_novel_data, test_medical_data

    test_novel_data()

    test_medical_data()

    log.info("done")
