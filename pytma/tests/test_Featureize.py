import os
import pytest

from pytma import Tokenizer
from pytma.DataSources import DataSourceError, download_tolstoy_novels, get_transcription_data
import os.path as op
import pytma
from pytma.Featurize import Featurize
from pytma.StopWord import StopWord
from pytma.Utility import log

data_path = op.join(pytma.__path__[0], 'data')



def test_Featureize():
    # This will be the unit test
    test_text = "He received multiple nominations for Nobel Prize in Literature every year from 1902 to 1906, \
         and nominations for Nobel Peace Prize in 1901, 1902 and 1910, and his miss of the prize is a major Nobel \
         prize controversy.[3][4][5][6] Born to an aristocratic Russian family in 1828,[2] he is best known for \
          the novels War and Peace (1869) and Anna Karenina (1877),[7] often cited as pinnacles of realist fiction. \
        [2] He first achieved literary acclaim in his twenties with his semi-autobiographical trilogy,  \
        Childhood, Boyhood, and Youth (1852–1856), and Sevastopol Sketches (1855), based upon his experiences \
        in the Crimean War."

    sw = StopWord(test_text.split())
    swList = StopWord.SwLibEnum.spacy_sw

    text = sw.remove(swList)

    log.info(text)

    text = " ".join(text)

    feat = Featurize(text)

    text_tf = feat.tf()

    log.info(text_tf)

    log.info(len(test_text.split()))

    log.info(len(set(test_text.split())))

    text_tf_idf = feat.tf_idf()

    log.info(text_tf_idf)

    text_vectors = feat.wtv_spacy()

    #feat.pca_wv(text_vectors)

    log.info("done")
