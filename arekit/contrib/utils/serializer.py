import logging

from arekit.common.data.input.providers.columns.opinion import OpinionColumnsProvider
from arekit.common.data.input.providers.columns.sample import SampleColumnsProvider
from arekit.common.data.input.providers.opinions import InputTextOpinionProvider
from arekit.common.data.input.providers.rows.opinions import BaseOpinionsRowProvider
from arekit.common.data.input.repositories.opinions import BaseInputOpinionsRepository
from arekit.common.data.input.repositories.sample import BaseInputSamplesRepository
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.pipeline.base import BasePipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class InputDataSerializationHelper(object):

    @staticmethod
    def serialize(pipeline, exp_io, iter_doc_ids_func, keep_labels_func, balance, data_type, sample_rows_provider):
        """ pipeline:
                note, it is important to provide a pipeline which results in linked opinions iteration
                for a particular document.
                    document (id, instance) -> ... -> linked opinion list
            keep_labels_func: function
                data_type -> bool
            iter_doc_ids:
                func(data_type)
        """
        assert(isinstance(pipeline, BasePipeline))
        assert(callable(iter_doc_ids_func))
        assert(callable(keep_labels_func))
        assert(isinstance(balance, bool))

        opinions_repo = BaseInputOpinionsRepository(
            columns_provider=OpinionColumnsProvider(),
            rows_provider=BaseOpinionsRowProvider(),
            storage=BaseRowsStorage())

        samples_repo = BaseInputSamplesRepository(
            columns_provider=SampleColumnsProvider(store_labels=keep_labels_func(data_type)),
            rows_provider=sample_rows_provider,
            storage=BaseRowsStorage())

        opinion_provider = InputTextOpinionProvider(pipeline)

        # Populate repositories
        opinions_repo.populate(opinion_provider=opinion_provider,
                               doc_ids=list(iter_doc_ids_func(data_type)),
                               desc="opinion")

        samples_repo.populate(opinion_provider=opinion_provider,
                              doc_ids=list(iter_doc_ids_func(data_type)),
                              desc="sample")

        if exp_io.balance_samples(data_type=data_type, balance=balance):
            samples_repo.balance()

        # Write repositories
        samples_repo.write(writer=exp_io.create_samples_writer(),
                           target=exp_io.create_samples_writer_target(data_type=data_type))

        opinions_repo.write(writer=exp_io.create_opinions_writer(),
                            target=exp_io.create_opinions_writer_target(data_type=data_type))
