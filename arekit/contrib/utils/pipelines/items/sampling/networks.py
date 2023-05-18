from arekit.common.experiment.data_type import DataType
from arekit.common.folding.base import BaseDataFolding
from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.contrib.networks.input.embedding.matrix import create_term_embedding_matrix
from arekit.contrib.networks.input.embedding.offsets import TermsEmbeddingOffsets
from arekit.contrib.networks.embedding import Embedding
from arekit.contrib.networks.input.providers.sample import NetworkSampleRowProvider
from arekit.contrib.utils.io_utils.embedding import NpEmbeddingIO
from arekit.contrib.utils.io_utils.samples import SamplesIO
from arekit.contrib.utils.utils_folding import folding_iter_states
from arekit.contrib.utils.serializer import InputDataSerializationHelper


class NetworksInputSerializerPipelineItem(BasePipelineItem):

    def __init__(self, save_labels_func, rows_provider, samples_io,
                 emb_io, balance_func, storage, save_embedding=True):
        """ This pipeline item allows to perform a data preparation for neural network models.

            considering a list of the whole data_types with the related pipelines,
            which are supported and required in a handler. It is necessary to know
            data_types in advance as it allows to create a complete vocabulary of input terms,
            with the related embeddings.

            balance: bool
                declares whethere there is a need to balance Train samples

            save_labels_func: function
                data_type -> bool

            save_embedding: bool
                save embedding and all the related information to it.
        """
        assert(isinstance(samples_io, SamplesIO))
        assert(isinstance(emb_io, NpEmbeddingIO))
        assert(isinstance(rows_provider, NetworkSampleRowProvider))
        assert(isinstance(save_embedding, bool))
        assert(callable(save_labels_func))
        assert(callable(balance_func))
        super(NetworksInputSerializerPipelineItem, self).__init__()

        self.__emb_io = emb_io
        self.__samples_io = samples_io
        self.__save_embedding = save_embedding
        self.__save_labels_func = save_labels_func
        self.__balance_func = balance_func
        self.__storage = storage
        self.__rows_provider = rows_provider

    def __serialize_iteration(self, data_type, pipeline, rows_provider, data_folding):
        assert(isinstance(data_type, DataType))
        assert(isinstance(pipeline, BasePipeline))

        repos = {
            "sample": InputDataSerializationHelper.create_samples_repo(
                keep_labels=self.__save_labels_func(data_type),
                rows_provider=rows_provider,
                storage=self.__storage),
        }

        writer_and_targets = {
            "sample": (self.__samples_io.Writer,
                       self.__samples_io.create_target(
                           data_type=data_type, data_folding=data_folding)),
        }

        for description, repo in repos.items():
            InputDataSerializationHelper.fill_and_write(
                repo=repo,
                pipeline=pipeline,
                doc_ids_iter=data_folding.fold_doc_ids_set()[data_type],
                do_balance=self.__balance_func(data_type),
                desc="{desc} [{data_type}]".format(desc=description, data_type=data_type),
                writer=writer_and_targets[description][0],
                target=writer_and_targets[description][1])

    def __handle_iteration(self, data_type_pipelines, data_folding):
        """ Performing data serialization for a particular iteration
        """
        assert(isinstance(data_type_pipelines, dict))
        assert(isinstance(data_folding, BaseDataFolding))

        # Prepare for the present iteration.
        self.__rows_provider.clear_embedding_pairs()

        for data_type, pipeline in data_type_pipelines.items():
            self.__serialize_iteration(pipeline=pipeline,
                                       data_type=data_type,
                                       rows_provider=self.__rows_provider,
                                       data_folding=data_folding)

        if not (self.__save_embedding and self.__rows_provider.HasEmbeddingPairs):
            return

        # Save embedding information additionally.
        term_embedding = Embedding.from_word_embedding_pairs_iter(self.__rows_provider.iter_term_embedding_pairs())
        embedding_matrix = create_term_embedding_matrix(term_embedding=term_embedding)
        vocab = list(TermsEmbeddingOffsets.extract_vocab(words_embedding=term_embedding))

        # Save embedding matrix
        self.__emb_io.save_embedding(data=embedding_matrix, data_folding=data_folding)
        self.__emb_io.save_vocab(data=vocab, data_folding=data_folding)

        del embedding_matrix

    # endregion

    def apply_core(self, input_data, pipeline_ctx):
        """
            data_type_pipelines: dict of, for example:
                {
                    DataType.Train: BasePipeline,
                    DataType.Test: BasePipeline
                }

                pipeline: doc_id -> parsed_news -> annot -> opinion linkages
                    for example, function: sentiment_attitude_extraction_default_pipeline
        """
        assert(isinstance(pipeline_ctx, PipelineContext))
        assert("data_type_pipelines" in pipeline_ctx)
        assert("data_folding" in pipeline_ctx)

        data_folding = pipeline_ctx.provide("data_folding")
        for _ in folding_iter_states(data_folding):
            self.__handle_iteration(data_type_pipelines=pipeline_ctx.provide("data_type_pipelines"),
                                    data_folding=data_folding)
