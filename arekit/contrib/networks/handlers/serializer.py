import collections

from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.api.ops_opin import OpinionOperations
from arekit.common.experiment.handler import ExperimentIterationHandler
from arekit.common.text.parser import BaseTextParser
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

    def __init__(self, exp_ctx, exp_io, doc_ops, opin_ops, value_to_group_id_func, text_parser, balance):
        assert(callable(value_to_group_id_func))
        assert(isinstance(exp_ctx, NetworkSerializationContext))
        assert(isinstance(exp_io, RuSentRelExperimentNetworkIOUtils))
        assert(isinstance(doc_ops, DocumentOperations))
        assert(isinstance(opin_ops, OpinionOperations))
        assert(isinstance(text_parser, BaseTextParser))
        assert(isinstance(balance, bool))
        super(NetworksInputSerializerExperimentIteration, self).__init__()

        self.__exp_ctx = exp_ctx
        self.__exp_io = exp_io
        self.__doc_ops = doc_ops
        self.__opin_ops = opin_ops
        self.__value_to_group_id_func = value_to_group_id_func
        self.__balance = balance
        self.__text_parser = text_parser

    # region protected methods

    @staticmethod
    def __add_term_embedding(dict_data, term, emb_vector):
        if term in dict_data:
            return
        dict_data[term] = emb_vector

    def __handle_iteration(self, data_type, rows_provider):

        # Perform data serialization.
        InputDataSerializationHelper.serialize(
            exp_io=self.__exp_io,
            exp_ctx=self.__exp_ctx,
            doc_ops=self.__doc_ops,
            opin_ops=self.__opin_ops,
            terms_per_context=self.__exp_ctx.TermsPerContext,
            balance=self.__balance,
            value_to_group_id_func=self.__value_to_group_id_func,
            text_parser=self.__text_parser,
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
