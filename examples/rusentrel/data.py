from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.text.stemmer import Stemmer
from arekit.contrib.networks.core.input.data_serialization import NetworkSerializationData
from arekit.contrib.networks.embeddings.base import Embedding
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.processing.pos.base import POSTagger


class RuSentRelExperimentSerializationData(NetworkSerializationData):

    def __init__(self, labels_scaler, stemmer, pos_tagger, embedding, terms_per_context,
                     frames_version, str_entity_formatter, rusentrel_version, annotator):
        assert(isinstance(embedding, Embedding))
        assert(isinstance(stemmer, Stemmer))
        assert(isinstance(pos_tagger, POSTagger))
        assert(isinstance(rusentrel_version, RuSentRelVersions))
        assert(isinstance(frames_version, RuSentiFramesVersions))
        assert(isinstance(str_entity_formatter, StringEntitiesFormatter))
        assert(isinstance(terms_per_context, int))

        super(RuSentRelExperimentSerializationData, self).__init__(labels_scaler=labels_scaler,
                                                                   annot=annotator)

        self.__pos_tagger = pos_tagger
        self.__terms_per_context = terms_per_context
        self.__rusentrel_version = rusentrel_version
        self.__str_entity_formatter = str_entity_formatter
        self.__word_embedding = embedding

        self.__frames_collection = RuSentiFramesCollection.read_collection(RuSentiFramesVersions.V20)
        self.__frame_variant_collection = FrameVariantsCollection()
        self.__frame_variant_collection.fill_from_iterable(
            variants_with_id=self.__frames_collection.iter_frame_id_and_variants(),
            overwrite_existed_variant=True,
            raise_error_on_existed_variant=False)

    @property
    def PosTagger(self):
        return self.__pos_tagger

    @property
    def StringEntityFormatter(self):
        return self.__str_entity_formatter

    @property
    def StringEntityEmbeddingFormatter(self):
        return self.__str_entity_formatter

    @property
    def FrameVariantCollection(self):
        return self.__frame_variant_collection

    @property
    def WordEmbedding(self):
        return self.__word_embedding

    @property
    def TermsPerContext(self):
        return self.__terms_per_context
