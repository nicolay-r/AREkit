from arekit.common.data import const
from arekit.common.data.storages.base import BaseRowsStorage


class GoogleBertOutputStorage(BaseRowsStorage):
    """ This output assumes to be provided with only labels by default proposed here:
        https://github.com/google-research/bert
    """

    def apply_samples_view(self, row_ids, doc_ids):
        """
        In addition to such output we provide the following parameters via samples_view instance:
        - id -- is a row identifier, which is compatible with row_inds in serialized opinions.
        - doc_id -- is, towards which the output corresponds to.
        """
        assert(len(row_ids) == len(doc_ids) == len(self.DataFrame))

        df = self.DataFrame

        # Providing the latter into output.
        df.insert(0, const.ID, row_ids)
        df.insert(1, const.DOC_ID, doc_ids)

        # Providing columns
        df.set_index(const.ID)

        df.columns = [str(c) for c in df.columns]

    @classmethod
    def from_tsv(cls, filepath, sep='\t', compression='infer', encoding='utf-8', header=None):
        return super(GoogleBertOutputStorage, cls).from_tsv(filepath=filepath, sep=sep, compression=compression, encoding=encoding, header=header)

    # endregion
