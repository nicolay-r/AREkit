from arekit.common.experiment.api.base import BaseExperiment
from arekit.common.experiment.api.ctx_serialization import SerializationData
from arekit.common.experiment.api.enums import BaseDocumentTag
from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.api.ops_opin import OpinionOperations
from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.opinions.collection import OpinionCollection
from arekit.contrib.experiment_rusentrel.common import entity_to_group_func
from arekit.contrib.experiment_rusentrel.connotations.provider import RuSentiFramesConnotationProvider
from arekit.contrib.experiment_rusentrel.entities.str_simple_fmt import StringEntitiesSimpleFormatter
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions


class SingleDocOperations(DocumentOperations):
    """ Operations over a single document for inference.
    """

    def iter_tagget_doc_ids(self, tag):
        assert(isinstance(tag, BaseDocumentTag))
        assert(tag == BaseDocumentTag.Annotate)
        return 0

    def __init__(self, news, text_parser):
        folding = NoFolding(doc_ids_to_fold=[0], supported_data_types=[DataType.Test])
        super(SingleDocOperations, self).__init__(folding, text_parser)
        self.__doc = news

    def get_doc(self, doc_id):
        return self.__doc


class CustomOpinionOperations(OpinionOperations):

    def __init__(self, labels_formatter, iter_opins, synonyms):
        super(CustomOpinionOperations, self).__init__()
        self.__labels_formatter = labels_formatter
        self.__iter_opins = iter_opins
        self.__synonyms = synonyms

    @property
    def LabelsFormatter(self):
        return self.__labels_formatter

    def iter_opinions_for_extraction(self, doc_id, data_type):
        return self.__iter_opins

    def get_etalon_opinion_collection(self, doc_id):
        raise Exception("Not Supported")

    def get_result_opinion_collection(self, doc_id, data_type, epoch_index):
        raise Exception("Not Supported")

    def create_opinion_collection(self):
        return OpinionCollection(opinions=None,
                                 synonyms=self.__synonyms,
                                 error_on_duplicates=True,
                                 error_on_synonym_end_missed=True)


class CustomExperiment(BaseExperiment):

    def __init__(self, synonyms, exp_data, exp_io_type, opin_ops, doc_ops):
        assert(issubclass(exp_io_type, BaseIOUtils))
        super(CustomExperiment, self).__init__(exp_data=exp_data,
                                               experiment_io=exp_io_type(self),
                                               opin_ops=opin_ops,
                                               doc_ops=doc_ops,
                                               name="test",
                                               extra_name_suffix="test")

        self.__synonyms = synonyms

    def entity_to_group(self, entity):
        return entity_to_group_func(entity, synonyms=self.__synonyms)


class CustomSerializationData(SerializationData):

    def __init__(self, label_scaler, annot, stemmer):
        super(CustomSerializationData, self).__init__(label_scaler=label_scaler, annot=annot, stemmer=stemmer)

        frames_collection = RuSentiFramesCollection.read_collection(version=RuSentiFramesVersions.V20)
        self.__frames_connotation_provider = RuSentiFramesConnotationProvider(collection=frames_collection)
        self.__frame_variant_collection = FrameVariantsCollection()
        self.__entities_formatter = StringEntitiesSimpleFormatter()

    @property
    def StringEntityFormatter(self):
        return self.__entities_formatter

    @property
    def FramesConnotationProvider(self):
        return self.__frames_connotation_provider

    @property
    def FrameVariantCollection(self):
        return self.__frame_variant_collection

    @property
    def TermsPerContext(self):
        return 50