import stanfordnlp

def test_install():
    stanfordnlp.download('en',resource_dir='~/stanfordnlp_resources')   # This downloads the English models for the neural pipeline
    nlp = stanfordnlp.Pipeline() # This sets up a default neural pipeline in English
    doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
    doc.sentences[0].print_dependencies()