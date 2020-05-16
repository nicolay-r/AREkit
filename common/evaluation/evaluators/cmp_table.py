import pandas as pd

from arekit.common.evaluation.evaluators.utils import label_to_str
from arekit.common.labels.base import Label


class DocumentCompareTable:

    C_WHO = 'who'
    C_TO = 'to'
    C_ORIG = 'how_orig'
    C_RES = 'how_results'
    C_CMP = 'comparison'

    def __init__(self, cmp_table):
        assert(isinstance(cmp_table, pd.DataFrame))
        self.__cmp_table = cmp_table

    def __filter_by_label(self, col_name, label):
        assert(isinstance(col_name, str))
        assert(isinstance(label, Label))
        label_str = label_to_str(label)
        return DocumentCompareTable(cmp_table=self.__cmp_table[(self.__cmp_table[col_name] == label_str)])

    @staticmethod
    def create_template_df():
        return pd.DataFrame(columns=[DocumentCompareTable.C_WHO,
                                     DocumentCompareTable.C_TO,
                                     DocumentCompareTable.C_ORIG,
                                     DocumentCompareTable.C_RES,
                                     DocumentCompareTable.C_CMP])

    @classmethod
    def load(cls, filepath):
        assert(isinstance(filepath, unicode))
        return cls(cmp_table=pd.DataFrame.from_csv(filepath))

    def save(self, filepath):
        assert(isinstance(filepath, unicode))
        self.__cmp_table.to_csv(filepath)

    def filter_result_column_by_label(self, label):
        return self.__filter_by_label(col_name=self.C_RES, label=label)

    def filter_original_column_by_label(self, label):
        return self.__filter_by_label(col_name=self.C_ORIG, label=label)

    def filter_comparison_true(self):
        return DocumentCompareTable(cmp_table=self.__cmp_table[(self.__cmp_table[self.C_CMP] == True)])

    def __len__(self):
        return len(self.__cmp_table)
