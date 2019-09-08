import unittest

from pytma.Parse import ParseText
from pytma.Utility import log


class TestParse(unittest.TestCase):
    def test_parse(self):
        test_text = u"It was now reading the sign that said Privet Drive — no, looking at the sign; " \
                    "cats couldn't read maps or signs.He didn't see the owls swooping past in broad daylight, " \
                    "though people down in the street did; they pointed and gazed open-mouthed as owl after " \
                    "owl sped overhead"

        pt = ParseText(test_text)

        #TODO broken
        #parsed = pt.dependency_parse()

        noun_chunks = pt.noun_chunks()

        self.assertEqual(noun_chunks[2], 'Privet Drive')


if __name__ == '__main__':
    unittest.main()
