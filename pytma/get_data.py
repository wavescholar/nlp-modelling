import os
import sys
__all__ = ["get_transcription_data"]


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
    names=dir(__all__)
    things = dir(sys.modules[__name__])
    path = os.getcwd()
    data_path = os.path.join(pytma.__path__[0], 'data')
    file_name = data_path + "/mtsamples.csv"
    medical_df = pd.read_csv(file_name)
    medical_df = medical_df.dropna(axis=0, how='any')
    return  medical_df