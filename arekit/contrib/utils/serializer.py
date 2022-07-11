import collections
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
    def serialize(pipeline, doc_ids_iter, keep_labels, do_balance, sample_rows_provider,
                  samples_writer, samples_target, opinions_writer, opinions_target):
        """ pipeline:
                note, it is important to provide a pipeline which results in linked opinions iteration
                for a particular document.
                    document (id, instance) -> ... -> linked opinion list
        """
        assert(isinstance(pipeline, BasePipeline))
        assert(isinstance(doc_ids_iter, collections.Iterable))
        assert(isinstance(keep_labels, bool))
        assert(isinstance(do_balance, bool))

        opinions_repo = BaseInputOpinionsRepository(
            columns_provider=OpinionColumnsProvider(),
            rows_provider=BaseOpinionsRowProvider(),
            storage=BaseRowsStorage())

        samples_repo = BaseInputSamplesRepository(
            columns_provider=SampleColumnsProvider(store_labels=keep_labels),
            rows_provider=sample_rows_provider,
            storage=BaseRowsStorage())

        opinion_provider = InputTextOpinionProvider(pipeline)

        doc_ids = list(doc_ids_iter)

        # Populate repositories
        opinions_repo.populate(opinion_provider=opinion_provider,
                               doc_ids=doc_ids,
                               desc="opinion")

        samples_repo.populate(opinion_provider=opinion_provider,
                              doc_ids=doc_ids,
                              desc="sample")

        if do_balance:
            samples_repo.balance()

        # Write repositories
        samples_repo.write(writer=samples_writer, target=samples_target())
        opinions_repo.write(writer=opinions_writer, target=opinions_target())
