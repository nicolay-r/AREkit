import collections
import logging

from arekit.common.data.input.providers.columns.sample import SampleColumnsProvider
from arekit.common.data.input.providers.opinions import InputTextOpinionProvider
from arekit.common.data.input.providers.rows.base import BaseRowProvider
from arekit.common.data.input.repositories.base import BaseInputRepository
from arekit.common.data.input.repositories.sample import BaseInputSamplesRepository
from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.utils.data.storages.pandas_based import PandasBasedRowsStorage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class InputDataSerializationHelper(object):

    @staticmethod
    def create_samples_repo(keep_labels, rows_provider):
        assert(isinstance(rows_provider, BaseRowProvider))
        assert(isinstance(keep_labels, bool))
        return BaseInputSamplesRepository(
            columns_provider=SampleColumnsProvider(store_labels=keep_labels),
            rows_provider=rows_provider,
            storage=PandasBasedRowsStorage())

    @staticmethod
    def fill_and_write(pipeline, repo, target, writer, doc_ids_iter, desc="", do_balance=False):
        assert(isinstance(pipeline, BasePipeline))
        assert(isinstance(doc_ids_iter, collections.Iterable))
        assert(isinstance(repo, BaseInputRepository))
        assert(isinstance(do_balance, bool))

        doc_ids = list(doc_ids_iter)
        repo.populate(opinion_provider=InputTextOpinionProvider(pipeline),
                      doc_ids=doc_ids,
                      desc=desc)

        # hack related to a particular type check.
        if do_balance and isinstance(repo, BaseInputSamplesRepository):
            repo.balance()

        repo.write(writer=writer, target=target)
