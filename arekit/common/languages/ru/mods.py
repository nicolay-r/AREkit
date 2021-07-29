# -*- coding: utf-8 -*-
from arekit.common.languages.mods import BaseLanguageMods


class RussianLanguageMods(BaseLanguageMods):

    @staticmethod
    def replace_specific_word_chars(word):
        assert(isinstance(word, unicode))
        return word.replace(u'ё', u'e')

    @staticmethod
    def is_negation_word(word):
        return word == u'не'
