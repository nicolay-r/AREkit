import importlib
import zipfile
from os import path


from arekit.contrib.source.zip_utils import ZipArchiveUtils
from arekit.contrib.utils.lexicons.lexicon import Lexicon


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
    def __get_archive_filepath():
        return path.join(ZipArchiveUtils.get_data_root(), "rusentilex.zip")

    @classmethod
    def from_zip(cls):
        """ Using Pandas API to read lexicon.
        """
        pd = importlib.import_module("pandas")
        with zipfile.ZipFile(cls.__get_archive_filepath(), "r") as zip_ref:
            with zip_ref.open(cls.__INNER_PATH, mode='r') as csv_file:
                df = pd.read_csv(csv_file, sep=',')
                return cls(df)
