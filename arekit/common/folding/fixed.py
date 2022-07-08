from arekit.common.folding.base import BaseDataFolding


class FixedFolding(BaseDataFolding):

    def __init__(self, doc_to_dtype_func, doc_ids_to_fold, supported_data_types):
        assert(callable(doc_to_dtype_func))

        super(FixedFolding, self).__init__(doc_ids_to_fold=doc_ids_to_fold,
                                           supported_data_types=supported_data_types)

        self.__doc_to_dtype_func = doc_to_dtype_func

    @property
    def Name(self):
        return "fixed"

    @classmethod
    def from_parts(cls, parts):
        """ parts: dict
                dictionary of {data_type: [doc_ids]}
        """
        assert(isinstance(parts, dict))

        doc_to_type = {}
        for data_type, doc_ids in parts.items():
            for doc_id in doc_ids:
                assert(doc_id not in doc_to_type)
                doc_to_type[doc_id] = data_type

        return cls(doc_to_dtype_func=lambda doc_id: doc_to_type[doc_id],
                   doc_ids_to_fold=doc_to_type.keys(),
                   supported_data_types=list(parts.keys()))

    def fold_doc_ids_set(self):

        folded = {}
        for d_type in self._supported_data_types:
            folded[d_type] = []

        for doc_id in self._doc_ids_to_fold_set:
            d_type = self.__doc_to_dtype_func(doc_id)
            folded[d_type].append(doc_id)

        return folded
