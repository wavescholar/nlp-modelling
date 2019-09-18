import unittest

from pytma.Tokenizer import Tokenizer
from pytma.Utility import log


class TestTokenizer(unittest.TestCase):
    def test_tokenizer(self):
        # This will be the unit test

        test_text = """Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.
            The sky is pinkish-blue. You shouldn't eat cardboard"""

        text = Tokenizer.regexTokenize(test_text)

        t = Tokenizer(test_text)
        log.info(t.to_words())
        log.info(t.to_sentences())
        log.info(t.freqs())
        t.plot_freqs()

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
