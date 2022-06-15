import collections

from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.handler import ExperimentIterationHandler
from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.experiment_rusentrel.model_io.tf_networks import RuSentRelExperimentNetworkIOUtils
from arekit.contrib.networks.core.input.ctx_serialization import NetworkSerializationContext
from arekit.contrib.networks.core.input.embedding.matrix import create_term_embedding_matrix
from arekit.contrib.networks.core.input.embedding.offsets import TermsEmbeddingOffsets
from arekit.contrib.networks.core.input.formatters.pos_mapper import PosTermsMapper
from arekit.contrib.networks.core.input.providers.sample import NetworkSampleRowProvider
from arekit.contrib.networks.core.input.providers.text import NetworkSingleTextProvider
from arekit.contrib.networks.core.input.terms_mapping import StringWithEmbeddingNetworkTermMapping
from arekit.contrib.networks.embeddings.base import Embedding
from arekit.contrib.utils.serializer import InputDataSerializationHelper


class NetworksInputSerializerExperimentIteration(ExperimentIterationHandler):

    def __init__(self, pipeline, exp_ctx, exp_io, doc_ops, balance):
        """ pipeline:
                doc_id -> parsed_news -> annot -> opinion linkages
                for example, function: sentiment_attitude_extraction_default_pipeline
        """
        assert(isinstance(pipeline, BasePipeline))
        assert(isinstance(exp_ctx, NetworkSerializationContext))
        assert(isinstance(exp_io, RuSentRelExperimentNetworkIOUtils))
        assert(isinstance(doc_ops, DocumentOperations))
        assert(isinstance(balance, bool))
        super(NetworksInputSerializerExperimentIteration, self).__init__()

        self.__exp_ctx = exp_ctx
        self.__exp_io = exp_io
        self.__doc_ops = doc_ops
        self.__balance = balance
        self.__pipeline = pipeline

    # region protected methods

    @staticmethod
    def __add_term_embedding(dict_data, term, emb_vector):
        if term in dict_data:
            return
        dict_data[term] = emb_vector

    def __handle_iteration(self, data_type, rows_provider):

        # Perform data serialization.
        InputDataSerializationHelper.serialize(
            pipeline=self.__pipeline,
            exp_io=self.__exp_io,
            iter_doc_ids_func=lambda dtype: self.__doc_ops.iter_doc_ids(dtype),
            balance=self.__balance,
            data_type=data_type,
            sample_rows_provider=rows_provider)

    # endregion

    def on_iteration(self, iter_index):
        """ Performing data serialization for a particular iteration
        """

        term_embedding_pairs = collections.OrderedDict()

        text_terms_mapper = StringWithEmbeddingNetworkTermMapping(
            predefined_embedding=self.__exp_ctx.WordEmbedding,
            string_entities_formatter=self.__exp_ctx.StringEntityFormatter,
            string_emb_entity_formatter=self.__exp_ctx.StringEntityEmbeddingFormatter)

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

        for data_type in self.__exp_ctx.DataFolding.iter_supported_data_types():
            self.__handle_iteration(data_type=data_type, rows_provider=rows_provider)

        # Save embedding information additionally.
        term_embedding = Embedding.from_word_embedding_pairs_iter(iter(term_embedding_pairs.items()))
        embedding_matrix = create_term_embedding_matrix(term_embedding=term_embedding)
        vocab = list(TermsEmbeddingOffsets.extract_vocab(words_embedding=term_embedding))

        # Save embedding matrix
        self.__exp_io.save_embedding(data=embedding_matrix)
        self.__exp_io.save_vocab(data=vocab)

        del embedding_matrix

    # endregion
