import collections

from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.handler import ExperimentIterationHandler
from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.networks.core.input.ctx_serialization import NetworkSerializationContext
from arekit.contrib.networks.core.input.embedding.matrix import create_term_embedding_matrix
from arekit.contrib.networks.core.input.embedding.offsets import TermsEmbeddingOffsets
from arekit.contrib.networks.core.input.formatters.pos_mapper import PosTermsMapper
from arekit.contrib.networks.core.input.providers.sample import NetworkSampleRowProvider
from arekit.contrib.networks.core.input.providers.text import NetworkSingleTextProvider
from arekit.contrib.networks.core.input.terms_mapping import StringWithEmbeddingNetworkTermMapping
from arekit.contrib.networks.embedding import Embedding
from arekit.contrib.utils.model_io.tf_networks import DefaultNetworkIOUtils
from arekit.contrib.utils.serializer import InputDataSerializationHelper


class NetworksInputSerializerExperimentIteration(ExperimentIterationHandler):

    def __init__(self, data_type_pipelines, vectorizers, save_labels_func,
                 exp_ctx, exp_io, doc_ops, balance, save_embedding):
        """ This hanlder allows to perform a data preparation for neural network models.

            considering a list of the whole data_types with the related pipelines,
            which are supported and required in a hadler. It is necessary to know
            data_types in advance as it allows to create a complete vocabulary of input terms,
            with the related embeddings.

            balance: bool
                declares whethere there is a need to balance Train samples

            vectorizers: dict in which for every type there is an assigned Vectorizer
                vectorization of term types.
                {
                    TermType.Word: Vectorizer,
                    TermType.Entity: Vectorizer,
                    ...
                }

            save_labels_func: function
                data_type -> bool

            data_type_pipelines: dict of, for example:
                {
                    DataType.Train: BasePipeline,
                    DataType.Test: BasePipeline
                }

                pipeline: doc_id -> parsed_news -> annot -> opinion linkages
                    for example, function: sentiment_attitude_extraction_default_pipeline

            save_embedding: bool
                save embedding and all the related information to it.
        """
        assert(isinstance(data_type_pipelines, dict))
        assert(isinstance(exp_ctx, NetworkSerializationContext))
        assert(isinstance(exp_io, DefaultNetworkIOUtils))
        assert(isinstance(doc_ops, DocumentOperations))
        assert(isinstance(vectorizers, dict))
        assert(isinstance(balance, bool))
        assert(isinstance(save_embedding, bool))
        super(NetworksInputSerializerExperimentIteration, self).__init__()

        self.__data_type_pipelines = data_type_pipelines
        self.__exp_ctx = exp_ctx
        self.__exp_io = exp_io
        self.__doc_ops = doc_ops
        self.__save_labels_func = save_labels_func
        self.__vectorizers = vectorizers
        self.__balance = balance
        self.__save_embedding = save_embedding

    # region protected methods

    @staticmethod
    def __add_term_embedding(dict_data, term, emb_vector):
        if term in dict_data:
            return
        dict_data[term] = emb_vector

    def __handle_iteration(self, data_type, pipeline, rows_provider):
        assert(isinstance(data_type, DataType))
        assert(isinstance(pipeline, BasePipeline))

        # Perform data serialization.
        InputDataSerializationHelper.serialize(
            pipeline=pipeline,
            exp_io=self.__exp_io,
            iter_doc_ids_func=lambda dtype: self.__doc_ops.iter_doc_ids(dtype),
            keep_labels_func=lambda dtype: self.__save_labels_func(data_type),
            balance=self.__balance,
            data_type=data_type,
            sample_rows_provider=rows_provider)

    # endregion

    def on_iteration(self, iter_index):
        """ Performing data serialization for a particular iteration
        """

        term_embedding_pairs = collections.OrderedDict()

        text_terms_mapper = StringWithEmbeddingNetworkTermMapping(
            vectorizers=self.__vectorizers,
            string_entities_formatter=self.__exp_ctx.StringEntityFormatter)

        text_provider = NetworkSingleTextProvider(
            text_terms_mapper=text_terms_mapper,
            pair_handling_func=lambda pair: self.__add_term_embedding(
                dict_data=term_embedding_pairs,
                term=pair[0],
                emb_vector=pair[1]))

        rows_provider = NetworkSampleRowProvider(
            label_provider=self.__exp_ctx.LabelProvider,
            text_provider=text_provider,
            frames_connotation_provider=self.__exp_ctx.FramesConnotationProvider,
            frame_role_label_scaler=self.__exp_ctx.FrameRolesLabelScaler,
            pos_terms_mapper=PosTermsMapper(self.__exp_ctx.PosTagger))

        for data_type, pipeline in self.__data_type_pipelines.items():
            self.__handle_iteration(pipeline=pipeline, data_type=data_type, rows_provider=rows_provider)

        if not self.__save_embedding:
            return

        # Save embedding information additionally.
        term_embedding = Embedding.from_word_embedding_pairs_iter(iter(term_embedding_pairs.items()))
        embedding_matrix = create_term_embedding_matrix(term_embedding=term_embedding)
        vocab = list(TermsEmbeddingOffsets.extract_vocab(words_embedding=term_embedding))

        # Save embedding matrix
        self.__exp_io.save_embedding(data=embedding_matrix)
        self.__exp_io.save_vocab(data=vocab)

        del embedding_matrix

    # endregion
