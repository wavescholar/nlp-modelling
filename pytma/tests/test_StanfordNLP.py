import stanfordnlp

#This is a noop for now
def test_install():
    print(" Stanford NLP Test ")
    #stanfordnlp.download('en',resource_dir='stanfordnlp_resources',  force=True   )   # This downloads the English models for the neural pipeline
    useCUDA=False
    #BBCREVISIT : "RuntimeError: CUDA error: out of memory" - should not be getting this on Laptop
    if useCUDA:
        nlp = stanfordnlp.Pipeline() # This sets up a default neural pipeline in English
        doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
        doc.sentences[0].print_dependencies()
