from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.common.text.stemmer import Stemmer


class ExperimentFrameVariantsCollection(FrameVariantsCollection):
    """
    We adopt stemmer for additional mapping checks.
    """

    def __init__(self, stemmer):
        assert(isinstance(stemmer, Stemmer))
        super(ExperimentFrameVariantsCollection, self).__init__()
        self.__lemma_variants = None
        self.__stemmer = stemmer

    @staticmethod
    def __create_lemmatized_variants(variants, stemmer):
        assert(isinstance(variants, dict))
        assert(isinstance(stemmer, Stemmer))

        lemma_variants = {}
        for variant, frame_variant in variants.items():
            key = stemmer.lemmatize_to_str(variant)
            if key in lemma_variants:
                continue
            lemma_variants[key] = frame_variant

        return lemma_variants

    # region public methods

    def fill_from_iterable(self, variants_with_id, overwrite_existed_variant, raise_error_on_existed_variant):
        super(ExperimentFrameVariantsCollection, self).fill_from_iterable(
            variants_with_id=variants_with_id,
            overwrite_existed_variant=overwrite_existed_variant,
            raise_error_on_existed_variant=raise_error_on_existed_variant)

        self.__lemma_variants = self.__create_lemmatized_variants(variants=self.Variants,
                                                                  stemmer=self.__stemmer)

    def get_variant_by_value(self, value):
        result_value = super(ExperimentFrameVariantsCollection, self).get_variant_by_value(value)
        return result_value if result_value is not None else self.__lemma_variants[value]

    def has_variant(self, value):
        has_variant = super(ExperimentFrameVariantsCollection, self).has_variant(value)
        return True if has_variant else value in self.__lemma_variants

    # endregion
