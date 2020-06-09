# -*- coding: utf-8 -*-
from pymystem3 import Mystem
from arekit.processing.pos.mystem_wrap import POSMystemWrapper


mw = POSMystemWrapper(mystem=Mystem(entire_input=False))

# terms = u"мама мыла раму"
# cases = mw.get_terms_russian_cases(terms)
# print cases

term = u"книгами"
print term
print mw.get_term_russian_case(term)
print mw.get_term_number(term)

