import os
import wget


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
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    file_name = data_path + "/mtsamples.csv"
    medical_df = pd.read_csv(file_name)
    medical_df = medical_df.dropna(axis=0, how='any')
    return medical_df


def get_tolstoy_novels():
    """
    Anna Karenina
    Boyhood
    Childhood
    The Cossacks
    The Death of Ivan Ilyich
    Family Happiness
    Hadji Murat
    The Kreutzer Sonata
    Resurrection
    Youth

    :return:
    """
    items = dict()

    items["AnnaKarenina"] = "https://www.gutenberg.org/files/1399/1399-0.txt"
    items["Boyhood"] = "https://www.gutenberg.org/files/2450/2450-0.txt"
    items["Childhood"] = "https://www.gutenberg.org/files/2142/2142-0.txt"
    items["TheCossacks"] = "https://www.gutenberg.org/ebooks/4761.txt.utf-8"
    items["TheKreutzerSonata"] = "https://www.gutenberg.org/files/689/689-0.txt"
    items["Youth"] = "https://www.gutenberg.org/files/2637/2637-0.txt"
    items["WarAndPeace"] = "https://www.gutenberg.org/files/2600/2600-0.txt"
    # TheDeathofIvanIlyich = ""
    # FamilyHappiness = ""
    # HadjiMurat = ""
    # Sonata=""
    # Resurrection=""
    for item in items:
        print(item)
        filename = wget.download(items[item])


if __name__ == '__main__':

    get_tolstoy_novels()
