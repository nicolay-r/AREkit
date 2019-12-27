import zipfile
from os import path
from os.path import dirname

import pandas as pd

from arekit.source.lexicons.lexicon import Lexicon


class RuSentiLexLexicon(Lexicon):
    """
    RuSentiLex Lexicon wrapper for csv file stored in /data folder.
    """

    __INNER_PATH = 'rusentilex.csv'

    @property
    def ToneKey(self):
        return 'tone'

    @property
    def TermKey(self):
        return 'term'

    @staticmethod
    def __get_data_root():
        return path.join(dirname(__file__), u"../../data/")

    @staticmethod
    def __get_archive_filepath():
        return path.join(RuSentiLexLexicon.__get_data_root(), u"rusentilex.zip")

    @classmethod
    def from_zip(cls):
        with zipfile.ZipFile(cls.__get_archive_filepath(), "r") as zip_ref:
            with zip_ref.open(cls.__INNER_PATH, mode='r') as csv_file:
                df = pd.read_csv(csv_file, sep=',')
                return cls(df)
