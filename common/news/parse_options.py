from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.processing.lemmatization.base import Stemmer


class NewsParseOptions(object):

    def __init__(self, parse_entities, frame_variants_collection, stemmer):
        assert(isinstance(parse_entities, bool))
        assert(isinstance(stemmer, Stemmer) or stemmer is None)
        assert(isinstance(frame_variants_collection, FrameVariantsCollection) or frame_variants_collection is None)
        self.__parse_entities = parse_entities
        self.__frame_variants_collection = frame_variants_collection
        self.__stemmer = stemmer

    @property
    def Stemmer(self):
        return self.__stemmer

    @property
    def ParseEntities(self):
        return self.__parse_entities

    @property
    def ParseFrameVariants(self):
        return self.__frame_variants_collection is not None

    @property
    def FrameVariantsCollection(self):
        return self.__frame_variants_collection

