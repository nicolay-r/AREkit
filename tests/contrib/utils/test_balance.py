# We perform a complete and clean data reading from scratch.
import unittest
from os.path import join, dirname

from arekit.common.data import const
from arekit.contrib.utils.data.readers.csv_pd import PandasCsvReader
from arekit.contrib.utils.data.service.balance import PandasBasedStorageBalancing


class TestBalancing(unittest.TestCase):

    __output_dir = join(dirname(__file__), "out")

    def test(self):
        reader = PandasCsvReader()

        balanced_storage = PandasBasedStorageBalancing.create_balanced_from(
            storage=reader.read(target=join(self.__output_dir, "sample-train-0.csv")),
            column_name=const.LABEL_UINT,
            free_origin=True)

        print(balanced_storage.DataFrame)
