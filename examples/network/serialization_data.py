from arekit.contrib.experiment_rusentrel.connotations.provider import RuSentiFramesConnotationProvider
from arekit.contrib.experiment_rusentrel.entities.str_simple_fmt import StringEntitiesSimpleFormatter
from arekit.contrib.experiment_rusentrel.labels.scalers.three import ThreeLabelScaler
from arekit.contrib.networks.core.input.data_serialization import NetworkSerializationData
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.processing.pos.mystem_wrap import POSMystemWrapper


class CustomSerializationData(NetworkSerializationData):

    def __init__(self, label_scaler, embedding, annot, stemmer,
                 frame_variants_collection, terms_per_context):
        super(CustomSerializationData, self).__init__(labels_scaler=label_scaler, annot=annot)

        frames_collection = RuSentiFramesCollection.read_collection(version=RuSentiFramesVersions.V20)
        self.__frame_roles_label_scaler = ThreeLabelScaler()
        self.__frames_connotation_provider = RuSentiFramesConnotationProvider(collection=frames_collection)
        self.__frame_variants_collection = frame_variants_collection
        self.__entities_formatter = StringEntitiesSimpleFormatter()
        self.__embedding = embedding
        self.__pos_tagger = POSMystemWrapper(stemmer.MystemInstance)
        self.__terms_per_context = terms_per_context

    @property
    def FrameRolesLabelScaler(self):
        return self.__frame_roles_label_scaler

    @property
    def WordEmbedding(self):
        return self.__embedding

    @property
    def PosTagger(self):
        return self.__pos_tagger

    @property
    def StringEntityEmbeddingFormatter(self):
        return self.__entities_formatter

    @property
    def StringEntityFormatter(self):
        return self.__entities_formatter

    @property
    def FramesConnotationProvider(self):
        return self.__frames_connotation_provider

    @property
    def FrameVariantCollection(self):
        return self.__frame_variants_collection

    @property
    def TermsPerContext(self):
        return self.__terms_per_context
