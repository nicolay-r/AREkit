import collections
import logging

from arekit.common.data import const

from arekit.common.data.input.providers.columns.sample import SampleColumnsProvider
from arekit.common.data.input.providers.rows.base import BaseRowProvider
from arekit.common.data.input.repositories.base import BaseInputRepository
from arekit.common.data.input.repositories.sample import BaseInputSamplesRepository
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.utils.data.contents.opinions import InputTextOpinionProvider
from arekit.contrib.utils.data.readers.csv_pd import PandasCsvReader
from arekit.contrib.utils.data.service.balance import PandasBasedStorageBalancing

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class InputDataSerializationHelper(object):

    @staticmethod
    def create_samples_repo(keep_labels, rows_provider, storage):
        assert(isinstance(rows_provider, BaseRowProvider))
        assert(isinstance(keep_labels, bool))
        assert(isinstance(storage, BaseRowsStorage))
        return BaseInputSamplesRepository(
            columns_provider=SampleColumnsProvider(store_labels=keep_labels),
            rows_provider=rows_provider,
            storage=storage)

    @staticmethod
    def fill_and_write(pipeline, repo, target, writer, doc_ids_iter, desc="", do_balance=False):
        assert(isinstance(pipeline, BasePipeline))
        assert(isinstance(doc_ids_iter, collections.Iterable))
        assert(isinstance(repo, BaseInputRepository))
        assert(isinstance(do_balance, bool))

        doc_ids = list(doc_ids_iter)

        repo.populate(contents_provider=InputTextOpinionProvider(pipeline),
                      doc_ids=doc_ids,
                      desc=desc,
                      writer=writer,
                      target=target)

        repo.push(writer=writer, target=target)

        if do_balance:

            # We perform a complete and clean data reading from scratch.
            reader = PandasCsvReader()
            balanced_storage = PandasBasedStorageBalancing.create_balanced_from(
                storage=reader.read(target=target), column_name=const.LABEL, free_origin=True)

            # Initializing the new repository instance.
            repo = BaseInputSamplesRepository(columns_provider=repo._columns_provider,
                                              rows_provider=repo._rows_provider,
                                              storage=balanced_storage)

            repo.push(writer=writer, target=target)
