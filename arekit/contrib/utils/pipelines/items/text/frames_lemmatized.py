from arekit.common.text.stemmer import Stemmer
from arekit.contrib.utils.pipelines.items.text.frames import FrameVariantsParser
from arekit.contrib.utils.processing.languages.ru.mods import RussianLanguageMods


class LemmasBasedFrameVariantsParser(FrameVariantsParser):

    def __init__(self, frame_variants, stemmer, locale_mods=RussianLanguageMods, save_lemmas=False):
        assert(isinstance(stemmer, Stemmer))
        assert(isinstance(save_lemmas, bool))
        super(LemmasBasedFrameVariantsParser, self).__init__(frame_variants=frame_variants)

        self.__frame_variants = frame_variants
        self.__stemmer = stemmer
        self.__save_lemmas = save_lemmas
        self.__max_variant_len = max([len(variant) for _, variant in frame_variants.iter_variants()])
        self.__locale_mods = locale_mods

    def __lemmatize_term(self, term):
        # we first split onto words for lemmatization and then join all of them.
        lemma = "".join(self.__stemmer.lemmatize_to_list(term))
        # then we replace certain chars according to the locale restrictions.
        return self.__locale_mods.replace_specific_word_chars(lemma)

    def __provide_lemmatized_terms(self, terms):
        """
        Compose a list of lemmatized versions of parsed_doc
        PS: Might be significantly slow, depending on stemmer were used.
        """
        assert(isinstance(terms, list))
        return [self.__lemmatize_term(term) if isinstance(term, str) else term for term in terms]

    def apply_core(self, input_data, pipeline_ctx):
        lemmas = self.__provide_lemmatized_terms(input_data)
        processed_it = self._iter_processed(terms=lemmas, origin=lemmas if self.__save_lemmas else input_data)
        return list(processed_it)
