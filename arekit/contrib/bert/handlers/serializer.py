from arekit.common.data.input.pipeline import text_opinions_iter_pipeline
from arekit.common.data.input.providers.columns.opinion import OpinionColumnsProvider
from arekit.common.data.input.providers.columns.sample import SampleColumnsProvider
from arekit.common.data.input.providers.opinions import InputTextOpinionProvider
from arekit.common.data.input.providers.rows.opinions import BaseOpinionsRowProvider
from arekit.common.data.input.repositories.opinions import BaseInputOpinionsRepository
from arekit.common.data.input.repositories.sample import BaseInputSamplesRepository
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.api.ops_opin import OpinionOperations
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.handler import ExperimentIterationHandler
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.bert.samplers.factory import create_bert_sample_provider


class BertExperimentInputSerializerIterationHandler(ExperimentIterationHandler):

    def __init__(self, exp_io, exp_ctx, doc_ops, opin_ops,
                 sample_labels_fmt, annot_labels_fmt, value_to_group_id_func,
                 sample_provider_type, entity_formatter, balance_train_samples):
        assert(isinstance(exp_io, BaseIOUtils))
        assert(isinstance(doc_ops, DocumentOperations))
        assert(isinstance(opin_ops, OpinionOperations))
        assert(isinstance(sample_labels_fmt, StringLabelsFormatter))
        assert(isinstance(annot_labels_fmt, StringLabelsFormatter))
        assert(callable(value_to_group_id_func))
        super(BertExperimentInputSerializerIterationHandler, self).__init__()

        self.__value_to_group_id_func = value_to_group_id_func
        self.__entity_formatter = entity_formatter
        self.__sample_provider_type = sample_provider_type
        self.__balance_train_samples = balance_train_samples
        self.__sample_label_formatter = sample_labels_fmt
        self.__annot_label_formatter = annot_labels_fmt
        self.__exp_io = exp_io
        self.__exp_ctx = exp_ctx
        self.__doc_ops = doc_ops
        self.__opin_ops = opin_ops

    # region private methods

    def __handle_iteration(self, data_type):
        assert(isinstance(data_type, DataType))

        # Create samples formatter.
        sample_rows_provider = create_bert_sample_provider(
            text_b_labels_fmt=self.__sample_label_formatter,
            provider_type=self.__sample_provider_type,
            label_scaler=self.__exp_ctx.LabelsScaler,
            entity_formatter=self.__entity_formatter)

        # Create repositories
        opinions_repo = BaseInputOpinionsRepository(
            columns_provider=OpinionColumnsProvider(),
            rows_provider=BaseOpinionsRowProvider(),
            storage=BaseRowsStorage())
        samples_repo = BaseInputSamplesRepository(
            columns_provider=SampleColumnsProvider(store_labels=True),
            rows_provider=sample_rows_provider,
            storage=BaseRowsStorage())

        # TODO. #250. Expand this pipeline with the annotation (in advance).
        # TODO. Check out the same comment at NetworkInputHelper.
        pipeline = text_opinions_iter_pipeline(
            parse_news_func=lambda doc_id: self.__doc_ops.parse_doc(doc_id),
            value_to_group_id_func=self.__value_to_group_id_func,
            iter_doc_opins=lambda doc_id: self.__opin_ops.iter_opinions_for_extraction(
                doc_id=doc_id, data_type=data_type),
            terms_per_context=self.__exp_ctx.TermsPerContext)

        # Create opinion provider
        opinion_provider = InputTextOpinionProvider(pipeline)

        # Populate repositories
        opinions_repo.populate(opinion_provider=opinion_provider,
                               doc_ids=list(self.__doc_ops.iter_doc_ids(data_type)),
                               desc="opinion")

        samples_repo.populate(opinion_provider=opinion_provider,
                              doc_ids=list(self.__doc_ops.iter_doc_ids(data_type)),
                              desc="sample")

        if self.__exp_io.balance_samples(data_type=data_type, balance=self.__balance_train_samples):
            samples_repo.balance()

        # Save repositories
        samples_repo.write(
            target=self.__exp_io.create_samples_writer_target(data_type),
            writer=self.__exp_io.create_samples_writer())

        opinions_repo.write(
            target=self.__exp_io.create_opinions_writer_target(data_type),
            writer=self.__exp_io.create_opinions_writer())

    # endregion

    # region protected methods

    def on_iteration(self, iter_index):
        """ Performing data serialization for a particular iteration
        """
        for data_type in self.__exp_ctx.DataFolding.iter_supported_data_types():
            self.__handle_iteration(data_type)

    def on_before_iteration(self):
        for data_type in self.__exp_ctx.DataFolding.iter_supported_data_types():

            # TODO. #250. A part of the further pipeline.
            # TODO. This might be included in InputTextOpinionProvider, as an initial operation
            # TODO. In a whole pipeline. This code duplicates the one in NetworkInputHelper.
            collections_it = self.__opin_ops.iter_annot_collections(
                exp_ctx=self.__exp_ctx, doc_ops=self.__doc_ops, data_type=data_type)

            for doc_id, collection in collections_it:

                target = self.__exp_io.create_opinion_collection_target(
                    doc_id=doc_id,
                    data_type=data_type)

                self.__exp_io.write_opinion_collection(
                    collection=collection,
                    target=target,
                    labels_formatter=self.__annot_label_formatter)

    # endregion
