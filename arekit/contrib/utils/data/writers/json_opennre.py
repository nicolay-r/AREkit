import json
import logging
import os
from os.path import dirname

from arekit.common.data import const
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage
from arekit.contrib.utils.data.writers.base import BaseWriter

logger = logging.getLogger(__name__)


class OpenNREJsonWriter(BaseWriter):
    """ This is a bag-based writer for the samples.
        Project page: https://github.com/thunlp/OpenNRE

        Every bag presented as follows:
            {
              'text' or 'token': ...,
              'h': {'pos': [start, end], 'id': ... },
              't': {'pos': [start, end], 'id': ... }
              'id': "id_of_the_text_opinion"
            }

        In terms of the linked opinions (i0, i1, etc.) we consider id of the first opinion in linkage.
        During the dataset reading stage via OpenNRE, these linkages automaticaly groups into bags.
    """

    EXTRA_KEYS_TEMPLATE = "_{}"

    def __init__(self, text_columns, encoding="utf-8"):
        """ text_columns: list
                column names that expected to be joined into a single (token) column.
        """
        assert(isinstance(text_columns, list))
        assert(isinstance(encoding, str))
        self.__text_columns = text_columns
        self.__encoding = encoding
        self.__target_f = None

    @staticmethod
    def __format_row(row, text_columns):
        """ Formatting that is compatible with the OpenNRE.
        """

        sample_id = row[const.ID]
        s_ind = int(row[const.S_IND])
        t_ind = int(row[const.T_IND])
        bag_id = sample_id[0:sample_id.find('_i')]

        # Gather tokens.
        tokens = []
        for text_col in text_columns:
            if text_col in row:
                tokens.extend(row[text_col].split())

        # Filtering JSON row.
        formatted_data = {
            "id": bag_id,
            "id_orig": sample_id,
            "token": tokens,
            "h": {"pos": [s_ind, s_ind + 1], "id": str(bag_id + "s")},
            "t": {"pos": [t_ind, t_ind + 1], "id": str(bag_id + "t")},
            "relation": str(int(row[const.LABEL])) if const.LABEL in row else "NA"
        }

        # Register extra fields.
        for key, value in row.items():
            if key not in formatted_data and key not in text_columns:
                formatted_data[OpenNREJsonWriter.EXTRA_KEYS_TEMPLATE.format(key)] = value

        return formatted_data

    def open_target(self, target):
        os.makedirs(dirname(target), exist_ok=True)
        self.__target_f = open(target, "w")
        pass

    def close_target(self):
        self.__target_f.close()

    def commit_line(self, storage):
        assert(isinstance(storage, RowCacheStorage))

        # Collect existed columns.
        row_data = {}
        for col_name in storage.iter_column_names():
            if col_name not in storage.RowCache:
                continue
            row_data[col_name] = storage.RowCache[col_name]

        self.__write_bag(bag=self.__format_row(row_data, text_columns=self.__text_columns),
                         json_file=self.__target_f)

    @staticmethod
    def __write_bag(bag, json_file):
        assert(isinstance(bag, dict))
        json.dump(bag, json_file, separators=(",", ":"), ensure_ascii=False)
        json_file.write("\n")

    def write_all(self, storage, target):
        assert(isinstance(storage, BaseRowsStorage))
        assert(isinstance(target, str))

        logger.info("Saving... {rows}: {filepath}".format(rows=(len(storage)), filepath=target))

        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding=self.__encoding) as json_file:
            for row_index, row in storage:
                self.__write_bag(bag=self.__format_row(row, text_columns=self.__text_columns),
                                 json_file=json_file)

        logger.info("Saving completed!")
