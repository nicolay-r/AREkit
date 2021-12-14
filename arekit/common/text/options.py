from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.text.stemmer import Stemmer


# TODO. This class is weird.
# TODO. All inherited types provide the same values for __init__.
class TextParseOptions(object):

    def __init__(self, skip_entities, frame_variants_collection, stemmer, keep_tokens=True):
        assert(isinstance(skip_entities, bool))
        assert(isinstance(stemmer, Stemmer) or stemmer is None)
        assert(isinstance(frame_variants_collection, FrameVariantsCollection) or frame_variants_collection is None)
        assert(isinstance(keep_tokens, bool))
        self.__skip_entities = skip_entities
        self.__frame_variants_collection = frame_variants_collection
        self.__stemmer = stemmer
        self.__keep_tokens = keep_tokens

    # TODO. This class is weird.
    # TODO. As this parameter related to the particular text-parser implementation.
    @property
    def Stemmer(self):
        return self.__stemmer

    @property
    def SkipEntities(self):
        return self.__skip_entities

    @property
    def ParseFrameVariants(self):
        return self.__frame_variants_collection is not None

    @property
    def FrameVariantsCollection(self):
        return self.__frame_variants_collection

    @property
    def KeepTokens(self):
        return self.__keep_tokens
