from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.contrib.experiment_rusentrel.connotations.provider import RuSentiFramesConnotationProvider
from arekit.contrib.experiment_rusentrel.labels.scalers.three import ThreeLabelScaler
from arekit.contrib.networks.core.input.ctx_serialization import NetworkSerializationContext
from arekit.contrib.networks.embeddings.base import Embedding
from arekit.processing.pos.base import POSTagger
from examples.network.common import create_frames_collection, create_and_fill_variant_collection


class CustomSerializationContext(NetworkSerializationContext):

    def __init__(self, labels_scaler, pos_tagger, embedding,
                 terms_per_context, str_entity_formatter, annotator,
                 name_provider, data_folding):
        assert(isinstance(embedding, Embedding))
        assert(isinstance(pos_tagger, POSTagger))
        assert(isinstance(str_entity_formatter, StringEntitiesFormatter))
        assert(isinstance(terms_per_context, int))

        super(CustomSerializationContext, self).__init__(
            labels_scaler=labels_scaler, annot=annotator,
            name_provider=name_provider, data_folding=data_folding)

        self.__pos_tagger = pos_tagger
        self.__terms_per_context = terms_per_context
        self.__str_entity_formatter = str_entity_formatter
        self.__word_embedding = embedding
        self.__frames_collection = create_frames_collection()
        self.__frame_variant_collection = create_and_fill_variant_collection(self.__frames_collection)
        self.__frame_roles_label_scaler = ThreeLabelScaler()
        self.__frames_connotation_provider = RuSentiFramesConnotationProvider(collection=self.__frames_collection)

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

    @property
    def FramesConnotationProvider(self):
        return self.__frames_connotation_provider

    @property
    def FrameRolesLabelScaler(self):
        return self.__frame_roles_label_scaler
