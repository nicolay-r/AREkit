import os

from arekit.common.entities.base import Entity
from arekit.common.experiment.api.base import BaseExperiment
from arekit.common.experiment.api.enums import BaseDocumentTag
from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.api.ops_opin import OpinionOperations
from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.item import BasePipelineItem
from arekit.common.utils import split_by_whitespaces
from arekit.contrib.experiment_rusentrel.connotations.provider import RuSentiFramesConnotationProvider
from arekit.contrib.experiment_rusentrel.entities.str_simple_fmt import StringEntitiesSimpleFormatter
from arekit.contrib.experiment_rusentrel.labels.scalers.three import ThreeLabelScaler
from arekit.contrib.networks.core.input.data_serialization import NetworkSerializationData
from arekit.contrib.experiment_rusentrel.model_io.tf_networks import NetworkIOUtils
from arekit.contrib.source import utils
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.pos.mystem_wrap import POSMystemWrapper
from examples.download import EMBEDDING_FILENAME
from examples.network.embedding import RusvectoresEmbedding


class SingleDocOperations(DocumentOperations):
    """ Operations over a single document for inference.
    """

    def iter_tagget_doc_ids(self, tag):
        assert(isinstance(tag, BaseDocumentTag))
        assert(tag == BaseDocumentTag.Annotate)
        return [0]

    def __init__(self, news, text_parser):
        folding = NoFolding(doc_ids_to_fold=[0], supported_data_types=[DataType.Test])
        super(SingleDocOperations, self).__init__(folding, text_parser)
        self.__doc = news

    def get_doc(self, doc_id):
        return self.__doc


class CustomOpinionOperations(OpinionOperations):

    def __init__(self, labels_formatter, exp_io, synonyms, neutral_labels_fmt):
        super(CustomOpinionOperations, self).__init__()
        self.__labels_formatter = labels_formatter
        self.__exp_io = exp_io
        self.__synonyms = synonyms
        self.__neutral_labels_fmt = neutral_labels_fmt

    @property
    def LabelsFormatter(self):
        return self.__labels_formatter

    def iter_opinions_for_extraction(self, doc_id, data_type):
        # Reading automatically annotated collection of neutral opinions.
        return self.__exp_io.read_opinion_collection(
            target=self.__exp_io.create_result_opinion_collection_target(
                doc_id=doc_id,
                data_type=data_type,
                epoch_index=0),
            labels_formatter=self.__neutral_labels_fmt,
            create_collection_func=self.create_opinion_collection)

    def get_etalon_opinion_collection(self, doc_id):
        return self.create_opinion_collection(None)

    def get_result_opinion_collection(self, doc_id, data_type, epoch_index):
        raise Exception("Not Supported")

    def create_opinion_collection(self, opinions=None):
        return OpinionCollection(opinions=[] if opinions is None else opinions,
                                 synonyms=self.__synonyms,
                                 error_on_duplicates=True,
                                 error_on_synonym_end_missed=True)


class CustomIOUtils(NetworkIOUtils):

    def __create_target(self, doc_id, data_type, epoch_index):
        return "data/result_d{doc_id}_{data_type}_e{epoch_index}.txt".format(
            doc_id=doc_id,
            data_type=data_type.name,
            epoch_index=epoch_index)

    def _get_experiment_sources_dir(self):
        return "data"

    def create_opinion_collection_target(self, doc_id, data_type, check_existance=False):
        return self.__create_target(doc_id=doc_id, data_type=data_type, epoch_index=0)

    def create_result_opinion_collection_target(self, doc_id, data_type, epoch_index):
        return self.__create_target(doc_id=doc_id, data_type=data_type, epoch_index=epoch_index)


class CustomExperiment(BaseExperiment):

    def __init__(self, exp_data, synonyms, doc_ops, labels_formatter, neutral_labels_fmt):

        exp_io = CustomIOUtils(self)

        opin_ops = CustomOpinionOperations(
            labels_formatter=labels_formatter,
            exp_io=exp_io,
            synonyms=synonyms,
            neutral_labels_fmt=neutral_labels_fmt)

        super(CustomExperiment, self).__init__(
            exp_data=exp_data,
            experiment_io=exp_io,
            opin_ops=opin_ops,
            doc_ops=doc_ops,
            name="test",
            extra_name_suffix="test")


class CustomSerializationData(NetworkSerializationData):

    def __init__(self, label_scaler, annot, stemmer, frame_variants_collection):
        assert(isinstance(stemmer, MystemWrapper))
        assert(isinstance(frame_variants_collection, FrameVariantsCollection))

        super(CustomSerializationData, self).__init__(labels_scaler=label_scaler, annot=annot)

        frames_collection = RuSentiFramesCollection.read_collection(version=RuSentiFramesVersions.V20)
        self.__frame_roles_label_scaler = ThreeLabelScaler()
        self.__frames_connotation_provider = RuSentiFramesConnotationProvider(collection=frames_collection)
        self.__frame_variants_collection = frame_variants_collection
        self.__entities_formatter = StringEntitiesSimpleFormatter()
        self.__embedding = RusvectoresEmbedding.from_word2vec_format(
            filepath=os.path.join(utils.get_default_download_dir(), EMBEDDING_FILENAME),
            binary=True)
        self.__embedding.set_stemmer(stemmer)
        self.__pos_tagger = POSMystemWrapper(stemmer.MystemInstance)

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
        return 50


class TermsSplitterParser(BasePipelineItem):

    def apply(self, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        return pipeline_ctx.update(param="src",
                                   value=split_by_whitespaces(pipeline_ctx.provide("src")))


class TextEntitiesParser(BasePipelineItem):

    def __init__(self):
        super(TextEntitiesParser, self).__init__()
        self.__id_in_doc = 0

    def apply(self, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))

        # reset counter.
        self.__id_in_doc = 0

        # extract terms.
        words = pipeline_ctx.provide("src")
        assert(isinstance(words, list))

        # update the result.
        pipeline_ctx.update("src", value=[self.__process_word(w) for w in words])

    def __process_word(self, word):
        assert(isinstance(word, str))

        # If this is a special word which is related to the [entity] mention.
        if word[0] == "[" and word[-1] == "]":
            entity = Entity(value=word[1:-1], e_type=None, id_in_doc=self.__id_in_doc)
            self.__id_in_doc += 1
            return entity

        return word


class CustomNetworkIOUtils(NetworkIOUtils):

    def _get_experiment_sources_dir(self):
        return "."
