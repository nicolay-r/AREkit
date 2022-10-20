import json
import logging
import os

from arekit.common.data import const
from arekit.common.data.storages.base import BaseRowsStorage
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

    def __init__(self, text_columns, encoding="utf-8"):
        assert(isinstance(encoding, str))
        self.__encoding = encoding
        self.__text_columns = text_columns

    @staticmethod
    def __write_bag(bag, json_file):
        json.dump(bag, json_file, separators=(",", ":"), ensure_ascii=False)
        json_file.write("\n")

    def write(self, storage, target):
        assert(isinstance(storage, BaseRowsStorage))
        assert(isinstance(target, str))

        logger.info("Saving... {rows}: {filepath}".format(rows=(len(storage)), filepath=target))

        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding=self.__encoding) as json_file:

            for row_index, row in storage:

                sample_id = row[const.ID]
                s_ind = int(row[const.S_IND])
                t_ind = int(row[const.T_IND])
                bag_id = sample_id[0:sample_id.find('_i')]

                # Gather tokens.
                tokens = []
                for text_col in self.__text_columns:
                    tokens.extend(row[text_col].split())

                # Fillring JSON row.
                json_row = {
                    "id": bag_id,
                    "id_orig": sample_id,
                    "token": tokens,
                    "h": {"pos": [s_ind, s_ind + 1], "id": str(bag_id + "s")},
                    "t": {"pos": [t_ind, t_ind + 1], "id": str(bag_id + "t")},
                    "relation": str(int(row[const.LABEL])) if const.LABEL in row else "NA"
                }

                self.__write_bag(json_row, json_file=json_file)

        logger.info("Saving completed!")
