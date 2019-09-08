import unittest

from pytma.POSTag import POStag

class TestPOSTag(unittest.TestCase):
    def test_pos_tag(self):
        test_text = u"It was now reading the sign that said Privet Drive — no, looking at the sign; " \
                    "cats couldn't read maps or signs.He didn't see the owls swooping past in broad daylight, " \
                    "though people down in the street did; they pointed and gazed open-mouthed as owl after " \
                    "owl sped overhead"

        pos = POStag(test_text, DEBUG=True)

        dict = pos.transform()

        for k in dict:
            print(k, dict[k])

        print("Done")

        self.assertEqual(dict['It'], 'n')


if __name__ == '__main__':
    unittest.main()
