import sys
import unittest
from pymystem3 import Mystem

sys.path.append('../../')

from arekit.processing.pos.mystem_wrap import POSMystemWrapper


class TestPartOfSpeech(unittest.TestCase):

    def test_cases(self):
        self.mystem_wrapper = POSMystemWrapper(mystem=Mystem(entire_input=False))

        term = "книгами"
        print(term)
        print(self.mystem_wrapper.get_term_case(term))
        print(self.mystem_wrapper.get_term_number(term))
        print(self.mystem_wrapper.get_term_pos(term))

        terms = "мама мыла раму"
        cases = self.mystem_wrapper.get_terms_russian_cases(terms)
        print(cases)


if __name__ == '__main__':
    unittest.main()
