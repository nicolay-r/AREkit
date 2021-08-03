import collections
from arekit.common.evaluation.cmp_opinions import OpinionCollectionsToCompare


class OpinionCollectionsToCompareUtils:

    def __init__(self):
        pass

    @staticmethod
    def iter_comparable_collections(doc_ids,
                                    read_result_collection_func,
                                    read_etalon_collection_func):
        assert(isinstance(doc_ids, collections.Iterable))

        for doc_id in doc_ids:
            yield OpinionCollectionsToCompare(doc_id=doc_id,
                                              read_result_collection_func=read_result_collection_func,
                                              read_etalon_collection_func=read_etalon_collection_func)
