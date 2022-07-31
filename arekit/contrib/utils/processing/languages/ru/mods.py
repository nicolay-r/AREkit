from arekit.contrib.utils.processing.languages.mods import BaseLanguageMods


class RussianLanguageMods(BaseLanguageMods):

    @staticmethod
    def replace_specific_word_chars(word):
        assert(isinstance(word, str))
        return word.replace('ё', 'e')

    @staticmethod
    def is_negation_word(word):
        return word == 'не'
